import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import TrainDatasetFromFolder2, ValDatasetFromFolder, display_transform
from model import CSNet_Color

import torch.nn as nn
import torch
import torchvision.transforms as transforms

gray = transforms.Gray()

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--pre_epochs', default=70, type=int, help='pre train generator epoch number')
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')

parser.add_argument('--batchSize', default=8, type=int, help='train epoch number')
parser.add_argument('--sub_rate', default=0.5, type=float, help='sampling sub rate')
parser.add_argument('--rate_control', default=0, type=float, help='rate control')
parser.add_argument('--meas_rate_control', default=1.0, type=float, help='bitdepth for the SQ')
parser.add_argument('--blocksize', default=32, type=int, help='bitdepth for the SQ')

parser.add_argument('--loadEpoch', default=61, type=int, help='load epoch number')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
# parser.add_argument('--generatorWeights', type=str, default='pre_epochs_rate_control_0_SQ_subrate_0.5_meas_rate_control_1.0/netG_epoch_1_70.pth')
parser.add_argument('--discriminatorWeights', type=str, default='',
                    help="path to discriminator weights (to continue training)")

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
NUM_EPOCHS = opt.num_epochs
PRE_EPOCHS = opt.pre_epochs
RATE_CONTROL = opt.rate_control
MEAS_RATE_CONTROL = opt.meas_rate_control
LOAD_EPOCH = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_set = TrainDatasetFromFolder2('../Datasets/VOC2012/train', crop_size=CROP_SIZE,
                                   upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolder('../Datasets/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = CSNet_Color(UPSCALE_FACTOR, opt.sub_rate)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

if opt.generatorWeights != '':
    netG.load_state_dict(torch.load(opt.generatorWeights))
    # PRE_EPOCHS = 0
    LOAD_EPOCH = opt.loadEpoch
print netG

mse_loss = nn.MSELoss()

if torch.cuda.is_available():
    netG.cuda()
    mse_loss.cuda()

optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=20, gamma=0.5)

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

############ pre training generator network for PRE-EPOCHS #############

for epoch in range(LOAD_EPOCH+1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0, 'meas_loss': 0 }

    netG.train()
    schedulerG.step()
    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        running_results['batch_sizes'] += batch_size

        secret = Variable(target)
        if torch.cuda.is_available():
            secret = secret.cuda()
        cover = Variable(data)
        if torch.cuda.is_available():
            cover = cover.cuda()
        if epoch < PRE_EPOCHS:
            meas, fake_img = netG(cover, secret, 0)   # the last parameter is 0: training the SMG module.
            optimizerG.zero_grad()
            g_loss = mse_loss(fake_img, secret)

            g_loss.backward()

            optimizerG.step()

            running_results['g_loss'] += g_loss.data[0] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.4f' % (
                epoch, PRE_EPOCHS, running_results['g_loss'] / running_results['batch_sizes']))

        else:
            # no more training the sampling layer.
            
            for param2 in netG.parameters():
                param2.requires_grad = False
                break


            meas, cover_o, meas_o = netG(cover, secret, 1)  # The last parameter is 1: training the Steg module.
            
            optimizerG.zero_grad()
            g_loss = mse_loss(cover_o, cover)
            meas_real = Variable(torch.zeros(meas.shape).cuda()+meas.data)
            meas_loss = mse_loss(meas_o, meas_real)
            total_loss = g_loss + MEAS_RATE_CONTROL*meas_loss

            total_loss.backward()

            optimizerG.step()

            running_results['g_loss'] += g_loss.data[0] * batch_size
            running_results['meas_loss'] += meas_loss.data[0] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.5f Loss_Meas: %.5f' % (
                epoch, NUM_EPOCHS, running_results['g_loss'] / running_results['batch_sizes'], running_results['meas_loss'] / running_results['batch_sizes']))



    ############## testing generator network ###############

    # save model parameters
    save_dir = 'epochs_Color' + '_subrate_' + str(opt.sub_rate)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch%5 == 0:
        torch.save(netG.state_dict(), save_dir + '/netG_epoch_%d.pth' % (epoch))
