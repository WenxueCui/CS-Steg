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

import pytorch_ssim
from data_utils import TrainDatasetFromFolder2, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import CSNet, CSNet3

import torch.nn as nn
import torch
import torchvision.transforms as transforms

gray = transforms.Gray()

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--g_trigger_threshold', default=0.8, type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help='generator update trigger threshold')
parser.add_argument('--g_without_trigger_threshold', default=0.6, type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help='generator update without trigger threshold and it must less than g_trigger_threshold')
parser.add_argument('--g_update_number', default=3, type=int, choices=[1, 2, 3, 4, 5],
                    help='generator update number')
parser.add_argument('--d_update_number', default=3, type=int, choices=[1, 2, 3, 4, 5],
                    help='generator update number')
parser.add_argument('--pre_epochs', default=0, type=int, help='pre train generator epoch number')
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')

parser.add_argument('--batchSize', default=8, type=int, help='train epoch number')
parser.add_argument('--sub_rate', default=0.8, type=float, help='sampling sub rate')
parser.add_argument('--rate_control', default=0, type=float, help='rate control')
parser.add_argument('--meas_rate_control', default=1.2, type=float, help='bitdepth for the SQ')
parser.add_argument('--blocksize', default=32, type=int, help='bitdepth for the SQ')

parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
# parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--generatorWeights', type=str, default='pre_epochs_rate_control_0_SQ_subrate_0.8_meas_rate_control_1.2/netG_epoch_1_195.pth')
parser.add_argument('--enhancedWeights', type=str, default='')
# parser.add_argument('--enhancedWeights', type=str, default='pre_epochs_rate_control_0_EN_BN_subrate_0.2_meas_rate_control_0.8/netG_epoch_1_40.pth')
parser.add_argument('--discriminatorWeights', type=str, default='',
                    help="path to discriminator weights (to continue training)")

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
PRE_EPOCHS = opt.pre_epochs
G_TRIGGER_THRESHOLD = opt.g_trigger_threshold
G_WITHOUT_TRIGGER_THRESHOLD = opt.g_without_trigger_threshold
G_UPDATE_NUMBER = opt.g_update_number
D_UPDATE_NUMBER = opt.d_update_number
RATE_CONTROL = opt.rate_control
MEAS_RATE_CONTROL = opt.meas_rate_control
LOAD_EPOCH = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_set = TrainDatasetFromFolder2('../Datasets/VOC2012/train', crop_size=CROP_SIZE,
                                   upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolder('../Datasets/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = CSNet(UPSCALE_FACTOR, opt.sub_rate)
netE = CSNet3(UPSCALE_FACTOR, opt.sub_rate)

print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

if opt.enhancedWeights != '':
    netE.load_state_dict(torch.load(opt.enhancedWeights))

if opt.generatorWeights != '':
    netG.load_state_dict(torch.load(opt.generatorWeights))
    # PRE_EPOCHS = 0
    LOAD_EPOCH = opt.loadEpoch
# print netG

blocks = ['sampling','upsampling1','upsampling2','block11','block12','block13','block14','block15','block21','block22','block23','block24','block25','block26','block27','block28','block31','block32','block33','block34','block35','block36','block37','downsample1','downsample2'] 

for i in range(0,len(blocks)):
    block_temp = blocks[i]
    attr = getattr(netG, block_temp)
    for param2 in attr.parameters():
        param2.requires_grad = False
        # print param2
        # print param2_data
        # break
    setattr(netE, block_temp, attr)

netG = 0
#     print netG.block_temp

mse_loss = nn.MSELoss()

if torch.cuda.is_available():
    netE.cuda()
    mse_loss.cuda()

optimizerG = optim.Adam(netE.Enhanced.parameters(), lr=0.00009)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.6)

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

############ pre training generator network for PRE-EPOCHS #############

for epoch in range(LOAD_EPOCH+1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0 }

    netE.train()
    schedulerG.step()
    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        data_temp = torch.FloatTensor(batch_size, 1, opt.crop_size, opt.crop_size)
        target_temp = torch.FloatTensor(batch_size, 1, opt.crop_size, opt.crop_size)

        # convert the rgb images to the gray scale images.
        for j in range(batch_size):
            data_temp[j] = gray(data[j])
            target_temp[j] = gray(target[j])
        data = data_temp
        target = target_temp
        running_results['batch_sizes'] += batch_size

        secret = Variable(target)
        if torch.cuda.is_available():
            secret = secret.cuda()
        cover = Variable(data)
        if torch.cuda.is_available():
            cover = cover.cuda()
        if epoch < PRE_EPOCHS:
            meas, fake_img = netE(cover, secret, 0)
            optimizerG.zero_grad()
            g_loss = mse_loss(fake_img, secret)
            meas_real = Variable(torch.zeros(meas.shape).cuda())
            rate_control_loss = mse_loss(meas, meas_real)
            total_loss = g_loss + RATE_CONTROL*rate_control_loss

            total_loss.backward()

            optimizerG.step()

            running_results['g_loss'] += g_loss.data[0] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.4f' % (
                epoch, PRE_EPOCHS, running_results['g_loss'] / running_results['batch_sizes']))

        else:
            # print netG
            # for param2 in netG.parameters():
            #     param2.requires_grad = False
            #     # print param2
            #     # print param2_data
            #     break


            secret_o = netE(cover, secret, 2)

            optimizerG.zero_grad()
            g_loss = mse_loss(secret_o, secret)

            g_loss.backward()

            optimizerG.step()

            running_results['g_loss'] += g_loss.data[0] * batch_size
            # running_results['meas_loss'] += meas_loss.data[0] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.5f' % (
                epoch, NUM_EPOCHS, running_results['g_loss'] / running_results['batch_sizes']))

            

    ############## testing generator network ###############

    # save model parameters
    save_dir = 'pre_epochs_rate_control_' + str(RATE_CONTROL) + '_EN_BN' + '_subrate_' + str(opt.sub_rate) + '_meas_rate_control_' + str(MEAS_RATE_CONTROL)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch%5 == 0:
        torch.save(netE.state_dict(), save_dir + '/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

