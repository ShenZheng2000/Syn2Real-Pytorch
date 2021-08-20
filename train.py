# --- Imports --- #
import argparse
from torch.utils.data import DataLoader
from data.val_data import ValData
from modeling.model import DeRain_v2
from utils import *
import os
import numpy as np
import random


class Tester():
    def __init__(self, args):
        self.lambda_loss = args.lambda_loss
        self.val_batch_size = args.val_batch_size
        self.category = args.category
        self.exp_name = args.exp_name
        self.device_ids = [Id for Id in range(torch.cuda.device_count())]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_seed(self, seed):
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            print('Seed:\t{}'.format(seed))

    def test(self):
        print('--- Hyper-parameters for testing ---')
        print('val_batch_size: {}\nlambda_loss: {}\ncategory: {}'
              .format(self.val_batch_size, self.lambda_loss, self.category))

        # --- Set category-specific hyper-parameters  --- #
        if self.category == 'derain':
            val_data_dir = './data/test/derain/'
        elif self.category == 'dehaze':
            val_data_dir = './data/test/dehaze/'
        else:
            raise Exception('Wrong image category. Set it to derain or dehaze dateset.')

        # --- Validation data loader --- #
        val_filename = args.val_filename
        val_data_loader = DataLoader(ValData(val_data_dir, val_filename),
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     num_workers=0)

        # --- Define the network --- #
        net = DeRain_v2().to(self.device)

        # --- Load the network weight --- #
        net.load_state_dict(torch.load('./{}/{}_best'.format(self.exp_name, self.category)))

        # --- Use the evaluation model in testing --- #
        net.eval()
        if os.path.exists('./{}_results/{}/'.format(self.category, self.exp_name)) == False:
            os.mkdir('./{}_results/{}/'.format(self.category, self.exp_name))
            os.mkdir('./{}_results/{}/rain/'.format(self.category, self.exp_name))

        print('--- Testing starts! ---')
        start_time = time.time()
        val_psnr, val_ssim = validation(net, val_data_loader, self.device, self.category, self.exp_name, save_tag=True)
        end_time = time.time() - start_time
        print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
        print('validation time is {0:.4f}'.format(end_time))



if __name__ == "__main__":

    # --- Parse hyper-parameters  --- #
    parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
    parser.add_argument('-lambda_loss', default=0.04, type=float, help='Set the lambda in loss function',)
    parser.add_argument('-val_batch_size', default=1, type=int, help='Set the validation/test batch size')
    parser.add_argument('-exp_name', type=str, help='directory for saving the networks of the experiment')
    parser.add_argument('-category', default='derain', type=str, help='Set image category (derain or dehaze?)')
    parser.add_argument('-val_filename', default='SIRR_test.txt', type=str, help='dataset for testing (real_input_split1.txt or SIRR_test.txt)',)
    parser.add_argument('-seed', help='set random seed', default=19, type=int)
    args = parser.parse_args()

    t = Tester(args)
    t.test()




