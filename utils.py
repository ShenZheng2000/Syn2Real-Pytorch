# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
import torch.nn as nn
import numpy as np
import cv2


def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

    return ssim_list


def validation(net, val_data_loader, device, category, exp_name, save_tag=False):
    """
    :param net: Gatepred_imageNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: derain or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, image_name = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image,zy_in = net(input_im)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(pred_image, gt))
        #print(image_name,psnr_list[-1],ssim_list[-1])

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, image_name, category, exp_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(pred_image, image_name, category, exp_name):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)
    
    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        utils.save_image(pred_image_images[ind], './{}_results/{}/{}'.format(category, exp_name, image_name_1[:-3] + 'png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category, exp_name):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_{}_log.txt'.format(category, exp_name), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)



def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 100 if category == 'derain' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))

def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))

def gradient(y):
    gradient_h=y[:, :, :, :-1] - y[:, :, :, 1:]
    gradient_v=y[:, :, :-1, :] - y[:, :, 1:, :]

    return gradient_h, gradient_v

def TV(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_v=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_v

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class FocalLoss(nn.Module):

    # def __init__(self, device, gamma=0, eps=1e-7, size_average=True):
    def __init__(self, gamma=0, eps=1e-7, size_average=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.reduce = reduce
        # self.device = device

    def forward(self, input, target):
        # y = one_hot(target, input.size(1), self.device)
        y = one_hot(target, input.size(1))
        probs = F.softmax(input, dim=1)
        probs = (probs * y).sum(1)  # dimension ???
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.reduce:
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
        else:
            loss = batch_loss
        return loss

def one_hot(index, classes):
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]

    # mask = torch.Tensor(size).fill_(0).to(device)
    if torch.cuda.is_available():
        mask = torch.Tensor(size).fill_(0).cuda()
    else:
        mask = torch.Tensor(size).fill_(0)
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)


def get_NoGT_target(inputs):
    sfmx_inputs = F.log_softmax(inputs, dim=1)
    target = torch.argmax(sfmx_inputs, dim=1)
    return target

def rgb_demean(inputs):
    rgb_mean = np.array([0.48109378172, 0.4575245789, 0.4078705409]).reshape((3, 1, 1))
    inputs = inputs - rgb_mean  # inputs in [0,1]
    return inputs

def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_CUBIC)
    return new_target


