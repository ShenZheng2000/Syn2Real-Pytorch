# --- Imports --- #
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.train_data import TrainData
from data.val_data import ValData
from modeling.model import DeRain_v2
from modeling.GP import GPStruct
from modeling.fpn import *
from utils import *
from torchvision.models import vgg16
from modeling.perceptual import LossNetwork
import os
import sys
import numpy as np
import random
from torch.autograd import Variable

plt.switch_backend('agg')


def SegLoss(input_train, out_train, device):
    # TODO: Seg is too large, consider scale with 0.1
    # Build Segmentation model
    num_of_SegClass = 21
    seg = fpn(num_of_SegClass).to(device)
    seg_criterion = FocalLoss(gamma=2).to(device)

    # Build and clip residual image
    out_train = torch.clamp(input_train - out_train, 0., 1.)

    # Build and dmean seg. input (maybe clip image before)
    seg_input = out_train.data.cpu().numpy()
    for n in range(out_train.size()[0]):
        seg_input[n, :, :, :] = rgb_demean(seg_input[n, :, :, :])

    # send seg. input to cuda
    seg_input = Variable(torch.from_numpy(seg_input).to(device))

    # build seg. output
    seg_output = seg(seg_input)

    # build seg. target
    target = (get_NoGT_target(seg_output)).data.cpu()
    target_ = resize_target(target, seg_output.size(2))
    target_ = torch.from_numpy(target_).long().to(device)

    # calculate seg. loss
    seg_loss = seg_criterion(seg_output, target_).to(device)

    # freeze seg. backpropagation
    for param in seg.parameters():
        param.requires_grad = False

    return seg_loss


class Trainer():
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.crop_size = args.crop_size
        self.train_batch_size = args.train_batch_size
        self.epoch_start = args.epoch_start
        self.lambda_loss = args.lambda_loss
        self.val_batch_size = args.val_batch_size
        self.category = args.category
        self.version = args.version  # version1 is GP model used in conference paper and version2 is GP model (feature level GP) used in journal paper
        self.kernel_type = args.kernel_type
        self.exp_name = args.exp_name
        self.lambgp = args.lambda_GP
        self.use_GP_inlblphase = False  # indication whether or not to use GP during labeled phase
        self.lambseg = args.lambda_SEG

        self.labeled_name = args.labeled_name
        self.unlabeled_name = args.unlabeled_name
        self.val_filename = args.val_filename

        self.device_ids = [Id for Id in range(torch.cuda.device_count())]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.ghost = args.ghost
        self.mix = args.mix

        self.net = DeRain_v2(ghost=self.ghost, mix=self.mix).to(self.device)
        self.num_epochs, self.train_data_dir, self.val_data_dir = self.get_type(self.category)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # --- Load training data and validation/test data --- #
        self.unlbl_train_data_loader = DataLoader(TrainData(self.crop_size, self.train_data_dir, self.unlabeled_name),
                                                  batch_size=self.train_batch_size, shuffle=True, num_workers=0)
        self.lbl_train_data_loader = DataLoader(TrainData(self.crop_size, self.train_data_dir, self.labeled_name),
                                                batch_size=self.train_batch_size,
                                                shuffle=True, num_workers=0)
        self.val_data_loader = DataLoader(ValData(self.val_data_dir, self.val_filename),
                                          batch_size=self.val_batch_size,
                                          shuffle=False,
                                          num_workers=0)

        self.num_labeled = self.train_batch_size * len(self.lbl_train_data_loader)  # number of labeled images
        self.num_unlabeled = self.train_batch_size * len(self.unlbl_train_data_loader)  # number of unlabeled images

        self.gp_struct = GPStruct(self.num_labeled, self.num_unlabeled, self.train_batch_size, self.version,
                                  self.kernel_type)

    def percep_net(self):
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(self.device)
        # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
        for param in vgg_model.parameters():
            param.requires_grad = False
        loss_network = LossNetwork(vgg_model)
        loss_network.eval()
        return loss_network

    def get_seed(self, seed):
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            print('Seed:\t{}'.format(seed))

    def get_type(self, category):
        if category == 'derain':
            num_epochs = 200
            train_data_dir = './data/train/derain/'
            val_data_dir = './data/test/derain/'
        elif category == 'dehaze':
            num_epochs = 10
            train_data_dir = './data/train/dehaze/'
            val_data_dir = './data/test/dehaze/'
        else:
            raise Exception('Wrong image category. Set it to derain or dehaze dateset.')

        return num_epochs, train_data_dir, val_data_dir

    def load_train_weight(self, exp_name, net, category):
        if os.path.exists('./{}/'.format(exp_name)) == False:
            os.mkdir('./{}/'.format(exp_name))
        try:
            net.load_state_dict(torch.load('./{}/{}_best'.format(exp_name, category)))
            print('--- weight loaded ---')
        except:
            print('--- no weight loaded ---')

    def cal_params(self, net):
        pass
        # pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        # print("Total_params: {}".format(pytorch_total_params))

    def print_params(self):
        print('--- Hyper-parameters for training ---')
        print(
            'learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}\ncategory: {}'.format(
                self.learning_rate, self.crop_size,
                self.train_batch_size, self.val_batch_size, self.lambda_loss, self.category))

    def train_label(self, epoch):
        if self.lambgp != 0 and self.use_GP_inlblphase == True:
            self.gp_struct.gen_featmaps(self.lbl_train_data_loader, self.net, self.device)

        for batch_id, train_data in enumerate(self.lbl_train_data_loader):

            input_image, gt, imgid = train_data
            input_image = input_image.to(self.device)
            gt = gt.to(self.device)

            # --- Zero the parameter gradients --- #
            self.optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            self.net.train()
            pred_image, zy_in = self.net(input_image)

            smooth_loss = F.smooth_l1_loss(pred_image, gt)
            perceptual_loss = self.loss_network(pred_image, gt)
            gp_loss = 0
            if self.lambgp != 0 and self.use_GP_inlblphase == True:
                gp_loss = self.gp_struct.compute_gploss(zy_in, imgid, batch_id, 1)
            loss = smooth_loss + self.lambda_loss * perceptual_loss + self.lambgp * gp_loss

            loss.backward()
            self.optimizer.step()

            # --- To calculate average PSNR --- #
            self.psnr_list.extend(to_psnr(pred_image, gt))

            if not (batch_id % 100):
                print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

        # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(self.psnr_list) / len(self.psnr_list)

        # --- Save the network parameters --- #
        torch.save(self.net.state_dict(), './{}/{}'.format(self.exp_name, self.category))

        self.evaluate(epoch=epoch, train_psnr=train_psnr)

        if self.lambgp != 0:
            self.gp_struct.gen_featmaps(self.lbl_train_data_loader, self.net, self.device)

    def train_unlabel(self, epoch):
        if self.lambgp != 0:
            self.gp_struct.gen_featmaps_unlbl(self.unlbl_train_data_loader, self.net, self.device)

            for batch_id, train_data in enumerate(self.unlbl_train_data_loader):

                input_image, gt, imgid = train_data
                input_image = input_image.to(self.device)
                gt = gt.to(self.device)

                # --- Zero the parameter gradients --- #
                self.optimizer.zero_grad()

                # --- Forward + Backward + Optimize --- #
                self.net.train()
                pred_image, zy_in = self.net(input_image)
                gp_loss = 0
                seg_loss = 0

                # TODO: Unsuperivsed Segmentation
                assert pred_image.size() == input_image.size()
                if self.lambseg != 0:
                    seg_loss = SegLoss(input_image, pred_image, device=self.device)
                    #print("seg_loss is", seg_loss)

                if self.lambgp != 0:
                    gp_loss = self.gp_struct.compute_gploss(zy_in, imgid, batch_id, 0)
                    #print("gp_loss is", gp_loss)

                loss = self.lambgp * gp_loss + self.lambseg * seg_loss
                if loss != 0:
                    loss.backward()
                    self.optimizer.step()

                # --- To calculate average PSNR --- #
                self.psnr_list.extend(to_psnr(pred_image, gt))

                if not (batch_id % 100):
                    print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

            # --- Calculate the average training PSNR in one epoch --- #
            train_psnr = sum(self.psnr_list) / len(self.psnr_list)

            # --- Save the network parameters --- #
            torch.save(self.net.state_dict(), './{}/{}'.format(self.exp_name, self.category))

            self.evaluate(epoch=epoch, train_psnr=train_psnr)

    def evaluate(self, epoch, train_psnr):
        # --- Use the evaluation model in testing --- #
        self.net.eval()

        val_psnr, val_ssim = validation(self.net, self.val_data_loader, self.device, self.category, self.exp_name)
        one_epoch_time = time.time() - self.start_time
        print_log(epoch + 1, self.num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, self.category,
                  self.exp_name)

        # --- update the network weight --- #
        if val_psnr >= self.old_val_psnr:
            torch.save(self.net.state_dict(), './{}/{}_best'.format(self.exp_name, self.category))
            print('model saved')
            self.old_val_psnr = val_psnr

    def train(self):

        # set seed
        self.get_seed(args.seed)

        # print hyperparameters
        self.print_params()

        # --- Define the perceptual loss network --- #
        self.loss_network = self.percep_net()

        # --- Load the network weight --- #
        self.load_train_weight(self.exp_name, self.net, self.category)

        # --- Calculate all trainable parameters in network --- #
        print_network(self.net)

        # --- Previous PSNR and SSIM in testing --- #
        self.net.eval()
        self.old_val_psnr, self.old_val_ssim = validation(self.net, self.val_data_loader, self.device, self.category,
                                                          self.exp_name)
        print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(self.old_val_psnr, self.old_val_ssim))

        self.net.train()
        for epoch in range(self.epoch_start, self.num_epochs):
            self.psnr_list = []
            self.start_time = time.time()
            adjust_learning_rate(self.optimizer, epoch, category=self.category)

            # -------------------------------------------------------------------------------------------------------------
            # Labeled phase
            self.train_label(epoch)

            # -------------------------------------------------------------------------------------------------------------
            # Unlabeled Phase
            self.train_unlabel(epoch)


if __name__ == "__main__":
    # --- Parse hyper-parameters  --- #
    parser = argparse.ArgumentParser(description='Hyper-parameters for network')
    parser.add_argument('-learning_rate', default=2e-4, type=float, help='Set the learning rate', )
    parser.add_argument('-crop_size', default=[256, 256], nargs='+', type=int, help='Set the crop_size')
    parser.add_argument('-train_batch_size', default=18, type=int, help='Set the training batch size')
    parser.add_argument('-epoch_start', default=0, type=int, help='Starting epoch number of the training')
    parser.add_argument('-lambda_loss', default=0.04, type=float, help='Set the lambda in loss function')
    parser.add_argument('-val_batch_size', default=1, type=int, help='Set the validation/test batch size')
    parser.add_argument('-category', default='derain', type=str, help='Set image category (derain or dehaze?)')
    parser.add_argument('-version', default='version1', type=str, help='Set the GP model (version1 or version2?)')
    parser.add_argument('-kernel_type', default='Linear', type=str,
                        help='Set the GP model (Linear or Squared_exponential or Rational_quadratic?)')
    parser.add_argument('-exp_name', type=str, help='directory for saving the networks of the experiment')
    parser.add_argument('-lambda_GP', default=0.0015, type=float, help='Set the lambda_GP for gploss in loss function')
    parser.add_argument('-seed', default=19, type=int, help='set random seed')

    parser.add_argument('-labeled_name', default='real_input_split1.txt', type=str)
    parser.add_argument('-unlabeled_name', default='real_input_split1.txt', type=str)
    parser.add_argument('-val_filename', default='SIRR_test.txt', type=str)

    parser.add_argument('-lambda_SEG', default=0.0001, type=float)

    parser.add_argument('-ghost', default=1, type=int)  # ghost coef. (mid_ch = lambda_ghost * in_ch)
    parser.add_argument('-mix', default=False, type=bool)  # ghost coef. (mid_ch = lambda_ghost * in_ch)

    args = parser.parse_args()

    t = Trainer(args)
    t.train()
