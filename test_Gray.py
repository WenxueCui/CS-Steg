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
from model import CSNet3

import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
gray = transforms.Gray()

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
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
parser.add_argument('--pre_epochs', default=1, type=int, help='pre train generator epoch number')
parser.add_argument('--num_epochs', default=1, type=int, help='train epoch number')

parser.add_argument('--batchSize', default=1, type=int, help='train epoch number')
parser.add_argument('--sub_rate', default=0.6, type=float, help='sampling sub rate')
parser.add_argument('--rate_control', default=0, type=float, help='rate control for the distribution of measurements')
parser.add_argument('--meas_rate_control', default=1.0, type=float, help='offset of measurement rate')
parser.add_argument('--blocksize', default=32, type=int, help='bitdepth for the SQ')

parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--generatorWeights', type=str, default='pre_epochs_rate_control_0_EN_BN_subrate_0.6_meas_rate_control_1.0/netG_epoch_1_200.pth')
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_set = TrainDatasetFromFolder2('../Datasets/LIVE1', crop_size=CROP_SIZE,
                                   upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolder('../Datasets/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = CSNet3(UPSCALE_FACTOR, opt.sub_rate)
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


results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

############ pre training generator network for PRE-EPOCHS #############

stage = 1

save_dir = 'results_rate_control_' + str(RATE_CONTROL) + '_SQ' + '_subrate_' + str(opt.sub_rate) + '_meas_rate_control_' + str(MEAS_RATE_CONTROL)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_id = 0

for epoch in range(LOAD_EPOCH+1, LOAD_EPOCH+2):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0, 'meas_loss': 0 }
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    valing_results2 = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    netG.eval()
    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        valing_results['batch_sizes'] += batch_size
        valing_results2['batch_sizes'] += batch_size
        data_temp = torch.FloatTensor(batch_size, 1, opt.crop_size, opt.crop_size)
        target_temp = torch.FloatTensor(batch_size, 1, opt.crop_size, opt.crop_size)

        # convert the rgb images to the gray scale images.
        for j in range(batch_size):
            data_temp[j] = gray(data[j])
            target_temp[j] = gray(target[j])
        data = data_temp
        target = target_temp
        running_results['batch_sizes'] += batch_size
        
        # print data.shape, target.shape
        cover_ = ToPILImage()(data[0,:,:,:])
        cover_.save(save_dir+'/'+str(img_id)+'_cover'+'.png')
        secret_ = ToPILImage()(target[0,:,:,:])
        secret_.save(save_dir+'/'+str(img_id)+'_secret'+'.png')

        secret = Variable(target)
        if torch.cuda.is_available():
            secret = secret.cuda()
        cover = Variable(data)
        if torch.cuda.is_available():
            cover = cover.cuda()
        if stage == 0:
            meas, fake_img = netG(cover, secret, 0)
            g_loss = mse_loss(fake_img, secret)
            meas_real = Variable(torch.zeros(meas.shape).cuda())
            rate_control_loss = mse_loss(meas, meas_real)

            running_results['g_loss'] += rate_control_loss.data[0] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.4f' % (
                epoch, PRE_EPOCHS, running_results['g_loss'] / running_results['batch_sizes']))

        elif stage == 1:
            # print netG
            # for param2 in netG.parameters():
            #     param2.requires_grad = False
            #     # print param2
            #     # print param2_data
            #     break


            meas, cover_o, meas_o = netG(cover, secret, 1)
            
            batch_mse = ((cover_o - cover) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(cover_o, cover).data[0]
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            print 'cover:  ', valing_results['psnr'], valing_results['ssim']            

            # print meas, cover_o, meas_o
            g_loss = mse_loss(cover_o, cover)
            meas_real = Variable(torch.zeros(meas.shape).cuda()+meas.data)
            meas_loss = mse_loss(meas_o, meas_real)

            running_results['g_loss'] += g_loss.data[0] * batch_size
            running_results['meas_loss'] += meas_loss.data[0] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.5f Loss_Meas: %.5f' % (
                epoch, NUM_EPOCHS, running_results['g_loss'] / running_results['batch_sizes'], running_results['meas_loss'] / running_results['batch_sizes']))

            cover_o_ = ToPILImage()(cover_o.cpu()[0,:,:,:].data)
            cover_o_.save(save_dir+'/'+str(img_id)+'_container'+'.png')
            cover_o = ToPILImage()(torch.abs(cover_o.cpu()[0,:,:,:].data - data[0,:,:,:])*10)
            cover_o.save(save_dir+'/'+str(img_id)+'_diff_cover_container'+'.png')

        # elif stage == 2:

            secret_o = netG(cover, secret, 2)

            batch_mse = ((secret_o - secret) ** 2).data.mean()
            valing_results2['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(secret_o, secret).data[0]
            valing_results2['ssims'] += batch_ssim * batch_size
            valing_results2['psnr'] = 10 * log10(1 / (valing_results2['mse'] / valing_results2['batch_sizes']))
            valing_results2['ssim'] = valing_results2['ssims'] / valing_results2['batch_sizes']
            print 'secret:  ', valing_results2['psnr'], valing_results2['ssim']

            g_loss = mse_loss(secret_o, secret)

            running_results['g_loss'] += g_loss.data[0] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.5f' % (
                epoch, NUM_EPOCHS, running_results['g_loss'] / running_results['batch_sizes']))

            secret_o_ = ToPILImage()(secret_o.cpu()[0,:,:,:].data)
            secret_o_.save(save_dir+'/'+str(img_id)+'_revealed'+'.png')
            secret_o = ToPILImage()(torch.abs(secret_o.cpu()[0,:,:,:].data - target[0,:,:,:]))
            secret_o.save(save_dir+'/'+str(img_id)+'_diff_revealed_secret'+'.png')


        else:
            print 'stage is wrong!'
            exit()

        img_id += 1

