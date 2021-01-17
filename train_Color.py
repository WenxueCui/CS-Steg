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
from model import CSNet_Color

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

        # data_temp = torch.FloatTensor(batch_size, 1, opt.crop_size, opt.crop_size)
        # target_temp = torch.FloatTensor(batch_size, 1, opt.crop_size, opt.crop_size)

        # convert the rgb images to the gray scale images.
        # for j in range(batch_size):
        #     data_temp[j] = gray(data[j])
        #     target_temp[j] = gray(target[j])
        # data = data_temp
        # target = target_temp
        running_results['batch_sizes'] += batch_size

        secret = Variable(target)
        if torch.cuda.is_available():
            secret = secret.cuda()
        cover = Variable(data)
        if torch.cuda.is_available():
            cover = cover.cuda()
        if epoch < PRE_EPOCHS:
            meas, fake_img = netG(cover, secret, 0)
            optimizerG.zero_grad()
            g_loss = mse_loss(fake_img, secret)
            # meas_real = Variable(torch.zeros(meas.shape).cuda())
            # rate_control_loss = mse_loss(meas, meas_real)
            # total_loss = g_loss + RATE_CONTROL*rate_control_loss

            g_loss.backward()

            optimizerG.step()

            running_results['g_loss'] += g_loss.data[0] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.4f' % (
                epoch, PRE_EPOCHS, running_results['g_loss'] / running_results['batch_sizes']))

        else:
            # print netG
            for param2 in netG.parameters():
                param2.requires_grad = False
                # print param2
                # print param2_data
                break


            meas, cover_o, meas_o = netG(cover, secret, 1)
            # print meas, cover_o, meas_o
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
    save_dir = 'pre_epochs_rate_control_' + str(RATE_CONTROL) + '_SQ_Color' + '_subrate_' + str(opt.sub_rate) + '_meas_rate_control_' + str(MEAS_RATE_CONTROL)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch%5 == 0:
        torch.save(netG.state_dict(), save_dir + '/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))


############ trainging CSGAN for NUM_EPOCHS ##############

generator_criterion = GeneratorLoss()
adversarial_criterion = nn.BCELoss()

netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

if opt.discriminatorWeights != '':
    netD.load_state_dict(torch.load(opt.discriminatorWeights))
print netD

if torch.cuda.is_available():
    netD.cuda()
    generator_criterion.cuda()
    adversarial_criterion.cuda()
optimizerD = optim.Adam(netD.parameters(), lr=0.0001)
scheluderD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.3)

for epoch in range(LOAD_EPOCH + 1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    schedulerG.step()
    scheluderD.step()
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

        g_update_first = True
        d_update_first = True
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()

        index = 1
        while ((real_out.data[0] - fake_out.data[0] < G_WITHOUT_TRIGGER_THRESHOLD) or d_update_first) and (
                index <= D_UPDATE_NUMBER):
            netD.zero_grad()
            target_real = Variable(torch.rand(batch_size, 1) * 0.4 + 0.8).cuda()
            target_fake = Variable(torch.rand(batch_size, 1) * 0.2).cuda()
            d_loss = adversarial_criterion(netD(real_img), target_real) + \
                     adversarial_criterion(netD(fake_img), target_fake)
            d_loss.backward(retain_graph=True)
            optimizerD.step()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_update_first = False
            index += 1

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss
        ###########################
        # if (real_out.data[0] - fake_out.data[0] < G_WITHOUT_TRIGGER_THRESHOLD):
        # g_update_first = False
        index = 1
        while ((real_out.data[0] - fake_out.data[0] > G_TRIGGER_THRESHOLD) or g_update_first) and (
                index <= G_UPDATE_NUMBER):
            netG.zero_grad()
            fake_label = netD(fake_img)
            g_loss = generator_criterion(fake_label, fake_img, real_img, batch_size)
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            g_update_first = False
            index += 1

        fake_label = netD(fake_img)
        g_loss = generator_criterion(fake_label, fake_img, real_img, batch_size)
        running_results['g_loss'] += g_loss.data[0] * batch_size
        d_loss = 1 - real_out + fake_out
        running_results['d_loss'] += d_loss.data[0] * batch_size
        running_results['d_score'] += real_out.data[0] * batch_size
        running_results['g_score'] += fake_out.data[0] * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

    ############## tesing generator network ################

    netG.eval()
    out_path = 'training_results_rate_control_' + str(RATE_CONTROL) + '_SQ' + '_subrate_' + str(opt.sub_rate) + '_bitdepth_' + str(
        opt.bitdepth) + '/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    for val_lr, val_hr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)

        # convert the rgb images to the gray scale images.
        val_lr_temp = torch.FloatTensor(batch_size, 1, val_lr.size(2), val_lr.size(3))
        val_hr_restore_temp = torch.FloatTensor(batch_size, 1, val_hr_restore.size(2), val_hr_restore.size(3))
        val_hr_temp = torch.FloatTensor(batch_size, 1, val_hr.size(2), val_hr.size(3))

        # convert the rgb images to the gray scale images.
        for j in range(batch_size):
            val_lr_temp[j] = gray(val_lr[j])
            val_hr_restore_temp[j] = gray(val_hr_restore[j])
            val_hr_temp[j] = gray(val_hr[j])
        val_lr = val_lr_temp
        val_hr_restore = val_hr_restore_temp
        val_hr = val_hr_temp

        valing_results['batch_sizes'] += batch_size
        lr = Variable(val_lr, volatile=True)
        hr = Variable(val_hr, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).data[0]
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))

        val_images.extend(
            [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
             display_transform()(sr.data.cpu().squeeze(0))])
    val_images = torch.stack(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // 15)
    val_save_bar = tqdm(val_images, desc='[saving training results]')
    index = 1
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        index += 1

    # save model parameters
    save_dir = 'epochs_rate_control_' + str(RATE_CONTROL) + '_SQ' + '_subrate_' + str(opt.sub_rate) + '_bitdepth_' + str(opt.bitdepth)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(netG.state_dict(), save_dir + '/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), save_dir + '/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    # if epoch % 10 == 0 and epoch != 0:
    #     out_path = 'statistics/'
    #     data_frame = pd.DataFrame(
    #         data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
    #               'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
    #         index=range(1, epoch + 1))
    #     data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
