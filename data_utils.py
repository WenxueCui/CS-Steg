from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Mod_Crop, Crop

import random
from torch.autograd import Variable
from model import Generator_SQ
import torch

import torchvision.transforms as transforms

gray = transforms.Gray()
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def calculate_valid_crop_size_2(crop_size, blocksize):
    return crop_size - (crop_size % blocksize)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


def get_mask(global_block, local_block, sides):
    temp = global_block - local_block
    res = []
    if temp >= 2 * (sides - 1):
        y = temp - 2 * (sides - 1)
        x = sides - 1 - y
        res1 = np.arange(0, 2*x+1, 2)
        res2 = np.arange(2*x+3, global_block-1, 3)
        res = np.append(res1, res2)
    elif temp - sides + 1 <= sides - 1:
        y = temp - sides + 1
        x = sides - 1 - y
        res1 = np.arange(0, x+1, 1)
        res2 = np.arange(x+2, global_block-1, 2)
        res = np.append(res1, res2)

    return res


def get_masks(global_block, local_block, sides):
    temp = global_block - local_block
    res = []
    if temp >= 2 * (sides - 1):
        y = temp - 2 * (sides - 1)
        x = sides - 1 - y
        res1 = np.arange(0, 2*x+1, 2)
        res2 = np.arange(2*x+local_block, global_block-local_block+1, 3)
        res = np.append(res1, res2)
    elif temp - sides + 1 <= sides - 1:
        y = temp - sides + 1
        x = sides - 1 - y
        res1 = np.arange(0, x+1, 1)
        res2 = np.arange(x+2, global_block-local_block+1, 2)
        res = np.append(res1, res2)

    return res


def get_masks1(global_block, local_block, sides):
    temp = global_block - local_block
    res = []
    if temp >= 2 * (sides - 1):
        y = temp - 2 * (sides - 1)
        x = sides - 1 - y
        res1 = np.arange(0, 2*x+1, 2)
        res2 = np.arange(2*x+local_block+1, global_block-local_block+1, 3)
        res = np.append(res1, res2)
    elif temp - sides + 1 <= sides - 1:
        y = temp - sides + 1
        x = sides - 1 - y
        res1 = np.arange(0, x+1, 1)
        res2 = np.arange(x+2, global_block-local_block+1, 2)
        res = np.append(res1, res2)

    return res

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class TrainDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder2, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        rand = random.randint(1,len(self.image_filenames))
        index2 = index + rand
        index2 = index2%len(self.image_filenames)
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.hr_transform(Image.open(self.image_filenames[index2]))
        # lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class TrainDatasetFromFolder3(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder3, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        rand = random.randint(1,len(self.image_filenames))
        index2 = index + rand
        index2 = index2%len(self.image_filenames)
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.hr_transform(Image.open(self.image_filenames[index2]))
        # lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)



# this class is for getting the training datasets from the Model
class TrainDatasetFromModel(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, generatorWeights):
        super(TrainDatasetFromModel, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.crop_size = crop_size
        blocksize = 32
        subrate = 0.1
        bitdepth = 3
        self.model = Generator_SQ(upscale_factor, blocksize, subrate, bitdepth=bitdepth)
        self.model.load_state_dict(torch.load(generatorWeights))
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size_2(min(w, h), 32)

        if crop_size > self.crop_size:
            hr_image_ = CenterCrop(crop_size)(hr_image)
            hr_image = ToTensor()(hr_image_)
            hr_image_gray = gray(hr_image)
            hr_image = Variable(hr_image_gray)
            hr_image = hr_image.unsqueeze(0)
            # if torch.cuda.is_available():
            #     hr_image = hr_image.cuda()

            output = self.model(hr_image)
            output = output.squeeze(0)
            output = output.data
            output = output.cpu()
            output = ToPILImage(output)

            x = random.randint(0, crop_size - self.crop_size)
            y = random.randint(0, crop_size - self.crop_size)
            Crop_Image = Crop(x, y, self.crop_size)
            hr_image = Crop_Image(hr_image_gray)
            lr_image = Crop_Image(output)
            return ToTensor()(lr_image).unsqueeze(0).unsqueeze(1), ToTensor()(hr_image).unsqueeze(0).unsqueeze(1)
        else:
            hr_image = Image.open(self.image_filenames[1])
            w, h = hr_image.size
            crop_size = calculate_valid_crop_size_2(min(w, h), 32)

            hr_image_ = CenterCrop(crop_size)(hr_image)
            hr_image = ToTensor()(hr_image_)
            hr_image = Variable(hr_image)
            # if torch.cuda.is_available():
            #     hr_image = hr_image.cuda()

            output = self.model(hr_image)
            output = output.data
            output = output.cpu()
            output = ToPILImage(output)

            x = random.randint(0, crop_size - self.crop_size)
            y = random.randint(0, crop_size - self.crop_size)
            Crop_Image = Crop(x, y, self.crop_size)
            hr_image = Crop_Image(hr_image_)
            lr_image = Crop_Image(output)
            return ToTensor()(lr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size_2(w, 32)
        crop_size_2 = calculate_valid_crop_size_2(h, 32)
        # lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        # hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        # mod_crop = Mod_Crop()
        hr_image = CenterCrop((crop_size_2, crop_size))(hr_image)
        # lr_image = lr_scale(hr_image)
        # hr_restore_img = hr_scale(lr_image)
        # hr_image = mod_crop(hr_image)
        # lr_image = mod_crop(lr_image)
        # hr_restore_img = mod_crop(hr_restore_img)
        return ToTensor()(hr_image), ToTensor()(hr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder2, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size_2(min(w, h), 32)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        # mod_crop = Mod_Crop()
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        # hr_image = mod_crop(hr_image)
        # lr_image = mod_crop(lr_image)
        # hr_restore_img = mod_crop(hr_restore_img)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


# this file is get the val dataset from the model
class ValDatasetFromModel(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromModel, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size_2(min(w, h), 32)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        # mod_crop = Mod_Crop()
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        # hr_image = mod_crop(hr_image)
        # lr_image = mod_crop(lr_image)
        # hr_restore_img = mod_crop(hr_restore_img)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        # print self.image_filenames[index] # output the file name
        w, h = hr_image.size
        # crop_size = calculate_valid_crop_size_2(min(w, h), 32)
        w = int(np.floor(w/32)*32)
        h = int(np.floor(h/32)*32)
        crop_size = (h, w)
        # lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        # hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        # mod_crop = Mod_Crop()
        hr_image = CenterCrop(crop_size)(hr_image)
        # lr_image = lr_scale(hr_image)
        # hr_restore_img = hr_scale(lr_image)
        # hr_image = mod_crop(hr_image)
        # lr_image = mod_crop(lr_image)
        # hr_restore_img = mod_crop(hr_restore_img)
        return ToTensor()(hr_image), ToTensor()(hr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


# class TestDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(TestDatasetFromFolder, self).__init__()
#         self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
#         self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
#         self.upscale_factor = upscale_factor
#         self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
#         self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]
#
#     def __getitem__(self, index):
#         image_name = self.lr_filenames[index].split('/')[-1]
#         lr_image = Image.open(self.lr_filenames[index])
#         w, h = lr_image.size
#         hr_image = Image.open(self.hr_filenames[index])
#         hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
#         hr_restore_img = hr_scale(lr_image)
#         return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
#
#     def __len__(self):
#         return len(self.lr_filenames)
