import math

import torch.nn.functional as F
from torch import nn
# import torch.legacy.nn as nn2
# from torch.legacy.nn import SpatialSoftMax

from torch.autograd import Variable
import torch

import numpy as np
import collections
import time

from torchvision.transforms import ToPILImage
from PIL import Image

import os

from math import sqrt

from torchvision.transforms import ToTensor, ToPILImage

#my class for the reshape + concat layer

class Reshape_Concat(torch.autograd.Function):

    blocksize = 32

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        data = input_

        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros(data.shape).cuda()
        output = output.view(b_, c_/Reshape_Concat.blocksize/Reshape_Concat.blocksize, w_*Reshape_Concat.blocksize, h_*Reshape_Concat.blocksize)
        for i in range(0, w_):
            for j in range(0, h_):
                data_t = input_[:, :, i, j]
                data_temp = torch.zeros(data_t.shape).cuda() + data_t
                #data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_/Reshape_Concat.blocksize/Reshape_Concat.blocksize, Reshape_Concat.blocksize, Reshape_Concat.blocksize))
                output[:, :, i*Reshape_Concat.blocksize:(i+1)*Reshape_Concat.blocksize, j*Reshape_Concat.blocksize:(j+1)*Reshape_Concat.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.data

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros(grad_input.shape).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_t = grad_input[:, :, i*Reshape_Concat.blocksize:(i+1)*Reshape_Concat.blocksize, j*Reshape_Concat.blocksize:(j+1)*Reshape_Concat.blocksize]
                data_temp = torch.zeros(data_t.shape).cuda() + data_t
                #data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += data_temp

        return Variable(output)


def My_Reshape(input):
    return Reshape_Concat.apply(input)


class Reshape_Concat_Color(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        data = input_
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        # output = data.view(b_, c_/32/32, w_*32, h_*32)
        output = torch.FloatTensor(b_, c_/32/32, w_*32, h_*32)
        # output.cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = input_[:, :, i, j]
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_/32/32, 32, 32))
                output[:, :, i*32:(i+1)*32, j*32:(j+1)*32] = data_temp

        return output.cuda()

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.data

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        # output = grad_input.view(b_, c_, w_, h_)
        output = torch.FloatTensor(b_, c_, w_, h_)
        # output.cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i*32:(i+1)*32, j*32:(j+1)*32]
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, j, i] = data_temp

        return Variable(output.cuda())


def My_Reshape_Color(input):
    return Reshape_Concat_Color.apply(input)



# my code for the reshape + concat layer for the compressed sensing

class Reshape_Concat_Adap(torch.autograd.Function):

    blocksize = 0

    def __init__(self, block_size):
        #super(Reshape_Concat_Adap, self).__init__()
        Reshape_Concat_Adap.blocksize = block_size

    @staticmethod
    def forward(ctx, input_,):
        ctx.save_for_backward(input_)
        data = input_
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros(data.shape).cuda()
        output = output.view(b_, c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize, w_ * Reshape_Concat_Adap.blocksize, h_ * Reshape_Concat_Adap.blocksize)
        for i in range(0, w_):
            for j in range(0, h_):
                data_t = input_[:, :, i, j]
                data_temp = torch.zeros(data_t.shape).cuda() + data_t
                # data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize, j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.data

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros(grad_input.shape).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_t = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize, j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                data_temp = torch.zeros(data_t.shape).cuda() + data_t
                # data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += data_temp

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)

# the branches layer for the pytorch framework


class Branch(torch.autograd.Function):
    ratios = 20/21.0

    def __init__(self, ratios=20/21.0):
        Branch.ratios = ratios

    @staticmethod
    def forward(ctx, input_,):
        data = input_
        data1 = torch.zeros(data.shape).cuda() + data
        data2 = torch.zeros(data.shape).cuda() + data
        return data1, data2

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        grad_output1_t = grad_output1.data
        grad_output2_t = grad_output2.data

        output = torch.zeros(grad_output1_t.shape).cuda()

        output = output + grad_output1_t * Branch.ratios

        output = output + grad_output2_t * (1 - Branch.ratios)

        return Variable(output)

def My_Branch(input, ratios):
    return Branch(ratios).apply(input)

# my code for the scalar Q       round(measurements/stepsize)*stepsize

class SQ(torch.autograd.Function):
    # bitdepth = 0

    def __init__(self, bitdepth):
        # super(Reshape_Concat_Adap, self).__init__()
        SQ.bitdepth = bitdepth

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)
        data = input_
        # maxs = torch.max(data)
        # mins = torch.min(data)
        # stepsize = (maxs-mins)/(pow(2, SQ.bitdepth))

        channels = data.shape[0]
        yq = torch.zeros(data.shape).cuda()
        for i in range(0, channels):
            maxs = torch.max(data[i, :, :, :])
            mins = torch.min(data[i, :, :, :])
            stepsize_ = (maxs - mins) / pow(2, SQ.bitdepth)
            if stepsize_ < 0.00001:
                stepsize_ = 0.0001

            # print stepsize_
            yq[i, :, :, :] = yq[i, :, :, :] + torch.round(data[i, :, :, :] / stepsize_)
            yq[i,:,:,:] = yq[i,:,:,:] * stepsize_
            # stepsize_t = torch.Tensor([stepsize_]).cuda()


        # yq = torch.zeros(data.shape).cuda()
        # yq = yq + torch.round(data/stepsize)
        # compute the bpp.   sp: just for the test process.
        if(False):
            bpp = SQ.Measurement_Entropy(yq.cpu(), 0.15)
            print 'bpp: ', bpp
        # yq = yq*stepsize

        return yq

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.data

        output = torch.zeros(input_.shape).cuda()
        output = output + grad_input

        return Variable(output)

    @staticmethod
    def Measurement_Entropy(y, subrate):
        y_view = y.view(1, -1)
        y_list = y_view.numpy().tolist()
        y_list = y_list[0]
        length = float(len(y_list))
        d = collections.Counter(y_list)
        H = 0.0
        for k in d:
            p = d[k]/length
            H = H + (-p*math.log(p, 2))

        return H*length/(length/round(1024*subrate)*1024)


def My_SQ(input, bitdepth):
    return SQ(bitdepth).apply(input)

# this class if only for the scalar Q.   measurement/stepsize

class SQ_ONLY(torch.autograd.Function):
    # bitdepth = 0

    def __init__(self, bitdepth):
        # super(Reshape_Concat_Adap, self).__init__()
        SQ.bitdepth = bitdepth

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)
        data = input_
        # maxs = torch.max(data)
        # mins = torch.min(data)
        # print maxs
        # print mins
        # stepsize = (maxs-mins)/pow(2, SQ.bitdepth)
        # if the you want the stepsize be the constant, you need to change this code
        stepsize = SQ.bitdepth

        # print SQ.bitdepth

        stepsize_t = torch.Tensor([stepsize]).cuda()
        yq = torch.zeros(data.shape).cuda()
        # stepsize_t = torch.zeros(1) + stepsize
        yq = yq + torch.round(data/stepsize)
        # compute the bpp.   sp: just for the test process.

        # for writing the txt file.
        if(False):
            filename = 'test-results/' + 'lenna_0.6_1_199'+'.txt'
            w = yq.shape[2]
            h = yq.shape[3]
            for i in range(0, w):
                for j in range(0, h):
                    temp = yq[:, :, i, j]
                    temp = temp.cpu().numpy().tolist()
                    temp = temp[0]
                    temp = [int(a) for a in temp]
                    temp = str(temp)
                    temp = temp[1:-1]
                    temp = temp.replace(',', '')
                    with open(filename, 'a') as f:
                        f.write(temp)
                        f.write("\n")


        if(False):
            bpp = SQ_ONLY.Measurement_Entropy(yq.cpu(), 0.1)
            print 'bpp: ', bpp
        # yq = yq*stepsize

        if(True):
            c = yq.shape[1]
            subrate = c/(32.0*32.0)
            print 'subrate', subrate
            bpp = SQ_ONLY.Measurement_Entropy(yq.cpu(), subrate)
            print 'bpp', bpp
            filename = '../results/' + 'rd' + '.txt'
            filename_temp = '../results/' + 'temp.txt'
            with open(filename, 'a') as f:
                f.write(bpp.__str__())
                f.write('  ')
            if True:
                with open(filename_temp, 'w') as f:
                    f.write(bpp.__str__())

        return yq, stepsize_t

    @staticmethod
    def backward(ctx, grad_output, grad_stepsize):
        input_, = ctx.saved_tensors
        # data = input_
        # maxs = torch.max(data)
        # mins = torch.min(data)
        # stepsize = (maxs - mins) / pow(2, SQ.bitdepth)
        stepsize = SQ.bitdepth
        # print SQ.bitdepth
        grad_input = grad_output.data


        output = torch.zeros(input_.shape).cuda()
        output = output + grad_input/stepsize

        return Variable(output)

    # compute the entropy for the measurements from the compressed sensing with blocksize 32x32
    @staticmethod
    def Measurement_Entropy(y, subrate):
        y_view = y.view(1, -1)
        y_list = y_view.numpy().tolist()
        y_list = y_list[0]
        length = float(len(y_list))
        d = collections.Counter(y_list)
        H = 0.0
        for k in d:
            p = d[k]/length
            H = H + (-p*math.log(p, 2))

        return H*length/(length/round(32*32*subrate)*32*32)


def My_SQ_ONLY(input, bitdepth):
    return SQ_ONLY(bitdepth).apply(input)


class SQ_ONLY2(torch.autograd.Function):
    # bitdepth = 0

    def __init__(self, bitdepth):
        # super(Reshape_Concat_Adap, self).__init__()
        SQ.bitdepth = bitdepth

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)
        data = input_

        # print maxs
        # print mins

        channels = data.shape[0]
        stepsize_temp = torch.zeros(channels).cuda()
        yq = torch.zeros(data.shape).cuda()
        for i in range(0,channels):
            maxs = torch.max(data[i,:,:,:])
            mins = torch.min(data[i,:,:,:])
            stepsize_ = (maxs-mins)/pow(2, SQ.bitdepth)
            # print stepsize_
            yq[i,:,:,:] = yq[i,:,:,:] + torch.round(data[i,:,:,:] / stepsize_)
            # stepsize_t = torch.Tensor([stepsize_]).cuda()
            stepsize_temp[i] = stepsize_temp[i] + stepsize_

        # stepsize = (maxs-mins)/pow(2, SQ.bitdepth)
        # if the you want the stepsize be the constant, you need to change this code
        # stepsize = SQ.bitdepth

        # print SQ.bitdepth

        # stepsize_t = torch.Tensor([stepsize]).cuda()

        # stepsize_t = torch.zeros(1) + stepsize

        # compute the bpp.   sp: just for the test process.

        # for writing the txt file.
        if(False):
            filename = 'test-results/' + 'lenna_0.6_1_199'+'.txt'
            w = yq.shape[2]
            h = yq.shape[3]
            for i in range(0, w):
                for j in range(0, h):
                    temp = yq[:, :, i, j]
                    temp = temp.cpu().numpy().tolist()
                    temp = temp[0]
                    temp = [int(a) for a in temp]
                    temp = str(temp)
                    temp = temp[1:-1]
                    temp = temp.replace(',', '')
                    with open(filename, 'a') as f:
                        f.write(temp)
                        f.write("\n")


        if(False):
            bpp = SQ_ONLY.Measurement_Entropy(yq.cpu(), 0.2)
            print 'bpp: ', bpp
        # yq = yq*stepsize

        return yq, stepsize_temp

    @staticmethod
    def backward(ctx, grad_output, grad_stepsize):
        input_, = ctx.saved_tensors
        data = input_
        # maxs = torch.max(data)
        # mins = torch.min(data)
        # stepsize = (maxs - mins) / pow(2, SQ.bitdepth)
        # stepsize = SQ.bitdepth
        # print SQ.bitdepth

        channels = data.shape[0]

        output = torch.zeros(input_.shape).cuda()
        grad_input = grad_output.data
        for i in range(0, channels):
            maxs = torch.max(data[i, :, :, :])
            mins = torch.min(data[i, :, :, :])
            stepsize_ = (maxs - mins) / pow(2, SQ.bitdepth)
            output[i,:,:,:] = output[i,:,:,:] + grad_input[i,:,:,:] / stepsize_
            # stepsize_t = torch.Tensor([stepsize_]).cuda()



        return Variable(output)

    # compute the entropy for the measurements from the compressed sensing with blocksize 32x32
    @staticmethod
    def Measurement_Entropy(y, subrate):
        y_view = y.view(1, -1)
        y_list = y_view.numpy().tolist()
        y_list = y_list[0]
        length = float(len(y_list))
        d = collections.Counter(y_list)
        H = 0.0
        for k in d:
            p = d[k]/length
            H = H + (-p*math.log(p, 2))

        return H*length/(length/round(32*32*subrate)*32*32)


def My_SQ_ONLY2(input, bitdepth):
    return SQ_ONLY2(bitdepth).apply(input)


class SQ_ONLY_TEST(torch.autograd.Function):
    # bitdepth = 0

    def __init__(self, bitdepth):
        # super(Reshape_Concat_Adap, self).__init__()
        SQ.bitdepth = bitdepth

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)
        data = input_
        # maxs = torch.max(data)
        # mins = torch.min(data)
        # print maxs
        # print mins
        # stepsize = (maxs-mins)/pow(2, SQ.bitdepth)
        # if the you want the stepsize be the constant, you need to change this code
        stepsize = SQ.bitdepth

        # print SQ.bitdepth
        bpp = 0

        stepsize_t = torch.Tensor([stepsize]).cuda()
        yq = torch.zeros(data.shape).cuda()
        # stepsize_t = torch.zeros(1) + stepsize
        yq = yq + torch.round(data/stepsize)
        # compute the bpp.   sp: just for the test process.

        # for writing the txt file.
        if(False):
            filename = 'test-results/' + 'lenna_0.6_1_199'+'.txt'
            w = yq.shape[2]
            h = yq.shape[3]
            for i in range(0, w):
                for j in range(0, h):
                    temp = yq[:, :, i, j]
                    temp = temp.cpu().numpy().tolist()
                    temp = temp[0]
                    temp = [int(a) for a in temp]
                    temp = str(temp)
                    temp = temp[1:-1]
                    temp = temp.replace(',', '')
                    with open(filename, 'a') as f:
                        f.write(temp)
                        f.write("\n")


        if(True):
            bpp = SQ_ONLY.Measurement_Entropy(yq.cpu(), 0.1)
            print 'bpp: ', bpp
        # yq = yq*stepsize

        return yq, stepsize_t, bpp

    @staticmethod
    def backward(ctx, grad_output, grad_stepsize, grad_bpp):
        input_, = ctx.saved_tensors
        # data = input_
        # maxs = torch.max(data)
        # mins = torch.min(data)
        # stepsize = (maxs - mins) / pow(2, SQ.bitdepth)
        stepsize = SQ.bitdepth
        # print SQ.bitdepth
        grad_input = grad_output.data


        output = torch.zeros(input_.shape).cuda()
        output = output + grad_input/stepsize

        return Variable(output)

    # compute the entropy for the measurements from the compressed sensing with blocksize 32x32
    @staticmethod
    def Measurement_Entropy(y, subrate):
        y_view = y.view(1, -1)
        y_list = y_view.numpy().tolist()
        y_list = y_list[0]
        length = float(len(y_list))
        d = collections.Counter(y_list)
        H = 0.0
        for k in d:
            p = d[k]/length
            H = H + (-p*math.log(p, 2))

        return H*length/(length/round(32*32*subrate)*32*32)


def My_SQ_ONLY_TEST(input, bitdepth):
    return SQ_ONLY_TEST(bitdepth).apply(input)


# the scalar Q for the DPCM algorithm.

class SQ_DPCM(torch.autograd.Function):
    bitdepth = 0

    def __init__(self, bitdepth):
        # super(Reshape_Concat_Adap, self).__init__()
        SQ_DPCM.bitdepth = bitdepth

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)
        data = input_
        maxs = torch.max(data)
        mins = torch.min(data)
        stepsize = (maxs-mins)/pow(2, SQ.bitdepth)
        stepsize = 0.1
        y_hat = torch.zeros(1).cuda()
        yq = torch.zeros(data.shape).cuda()
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]
        for b in range(0, b_):
            for w in range(0, w_):
                for h in range(0, h_):
                    for c in range(0, c_):
                        d = data[b, c, w, h] - y_hat
                        yq_temp = torch.round(d/stepsize)*stepsize
                        y_hat = y_hat + yq_temp
                        yq[b:b+1, c:c+1, w:w+1, h:h+1] = y_hat
                    y_hat = y_hat * 0.0

        return yq

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.data

        output = torch.zeros(input_.shape).cuda()
        output = output + grad_input

        return Variable(output)


def My_DPCM_SQ(input, bitdepth):
    return SQ_DPCM(bitdepth).apply(input)

class SQ_LSMM(torch.autograd.Function):
    bitdepth = 0

    def __init__(self, bitdepth):
        # super(Reshape_Concat_Adap, self).__init__()
        SQ_LSMM.bitdepth = bitdepth

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)
        data = input_
        # maxs = torch.max(data)
        # mins = torch.min(data)
        # stepsize = (maxs-mins)/pow(2, SQ.bitdepth)
        stepsize = 0.1
        avg = torch.round(0.5/stepsize)*stepsize
        y_hat = torch.zeros(1).cuda() + avg
        yq = torch.zeros(data.shape).cuda()
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]
        for b in range(0, b_):
            for w in range(0, w_):
                for h in range(0, h_):
                    for c in range(0, c_):
                        d = data[b, c, w, h] - y_hat
                        yq_temp = torch.round(d/stepsize)*stepsize
                        y_hat = y_hat + yq_temp
                        yq[b:b+1, c:c+1, w:w+1, h:h+1] = y_hat
                    y_hat = y_hat * 0.0 + avg

        return yq

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.data

        output = torch.zeros(input_.shape).cuda()
        output = output + grad_input

        return Variable(output)


def My_LSMM_SQ(input, bitdepth):
    return SQ_LSMM(bitdepth).apply(input)

class SQ_LSMM_2(torch.autograd.Function):
    bitdepth = 0

    def __init__(self, bitdepth):
        # super(Reshape_Concat_Adap, self).__init__()
        SQ_LSMM_2.bitdepth = bitdepth

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)
        data = input_
        # maxs = torch.max(data)
        # mins = torch.min(data)
        # stepsize = (maxs-mins)/pow(2, SQ.bitdepth)
        stepsize = 0.1
        mid = torch.zeros(1).cuda()+0.5
        avg = torch.round(mid/stepsize)*stepsize
        y_hat = torch.zeros(1).cuda() + avg
        yq = torch.zeros(data.shape).cuda()
        yq_= torch.zeros(data.shape).cuda()
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        DC = torch.zeros(w_, h_).cuda()
        DC[0:1,0:1] = DC[0:1,0:1] + avg
        # floatTensor_temp = torch.zeros(1).cuda()
        num_temp = 0
        for b in range(0, b_):
            for w in range(0, w_):
                for h in range(0, h_):
                    for c in range(0, c_):
                        if num_temp is 0:

                            # d = floatTensor_temp + (data[b, c, w, h] - DC[w, h])
                            d = (data[b:b+1, c:c+1, w:w+1, h:h+1] - DC[w:w+1, h:h+1])
                            # print d
                            y_hat = DC[w:w+1,h:h+1]
                        else:
                            # d = floatTensor_temp + (data[b, c, w, h] - y_hat)
                            d = (data[b:b+1, c:c+1, w:w+1, h:h+1] - y_hat)
                            # print d
                        yq_temp = torch.round(d/stepsize)*stepsize
                        yq_[b:b+1, c:c+1, w:w+1, h:h+1] = torch.round(d/stepsize)
                        y_hat = y_hat + yq_temp
                        if num_temp is 0:
                            DC[w:w+1,h:h+1] = y_hat
                            if h < h_-1:
                                DC[w:w+1,h+1:h+2] = DC[w:w+1,h:h+1]
                            else:
                                if w < w_-1:
                                    DC[w+1:w+2,0:1] = DC[w:w+1,0:1]
                        yq[b:b+1, c:c+1, w:w+1, h:h+1] = y_hat
                        num_temp = num_temp + 1
                    y_hat = y_hat * 0.0 + avg
                    num_temp = 0
                num_temp = 0
            num_temp = 0
            DC[0:1,0:1] = DC[0:1,0:1]*0.0 + avg

        if (True):
            bpp = SQ_LSMM_2.Measurement_Entropy(yq_.cpu(), 0.2)
            print 'bpp: ', bpp
        # yq = yq*stepsize

        return yq

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.data

        output = torch.zeros(input_.shape).cuda()
        output = output + grad_input

        return Variable(output)

    # compute the entropy for the measurements from the compressed sensing with blocksize 32x32
    @staticmethod
    def Measurement_Entropy(y, subrate):
        y_view = y.view(1, -1)
        y_list = y_view.numpy().tolist()
        y_list = y_list[0]
        length = float(len(y_list))
        d = collections.Counter(y_list)
        H = 0.0
        for k in d:
            p = d[k] / length
            H = H + (-p * math.log(p, 2))

        return H * length / (length / round(16 * 16 * subrate) * 16 * 16)


def My_LSMM_SQ_2(input, bitdepth):
    return SQ_LSMM_2(bitdepth).apply(input)

def swish(x):
    return x * F.sigmoid(x)

# for the CSNet the blocksize can be changed according to the scale_factor.
class CSNet(nn.Module):
    def __init__(self, scale_factor, subrate):
        #upsample_block_num = int(math.log(scale_factor, 2))

        super(CSNet, self).__init__()

        # for sampling
        self.blocksize = 32/scale_factor
        self.sampling = nn.Conv2d(1, int(np.round(32*32*subrate)), 32, stride=32, padding=0)

        self.upsampling1 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)
        self.upsampling2 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)

        self.block11 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
 
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block15 = (nn.Conv2d(64, 1, kernel_size=5, padding=2))


        self.block21 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.block22 = ResidualBlock(64)
        self.block23 = ResidualBlock(64)
        self.block24 = ResidualBlock(64)
        self.block25 = ResidualBlock(64)
        self.block26 = ResidualBlock(64)
        self.block27 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block28 = (nn.Conv2d(64, 1, kernel_size=9, padding=4))
        
        self.block31 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block32 = ResidualBlock(64)
        self.block33 = ResidualBlock(64)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size=4, stride=4, padding=0),
            nn.PReLU()
        )
        self.block34 = ResidualBlock(256)
        self.block35 = ResidualBlock(256)
        self.block36 = ResidualBlock(256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, int(np.round(self.blocksize*self.blocksize*subrate*2)), kernel_size=8, stride=8, padding=0),
            nn.PReLU()
        )
        self.block37 = (nn.Conv2d(int(np.round(self.blocksize*self.blocksize*subrate*2)), int(np.round(self.blocksize*self.blocksize*subrate)), kernel_size=3, padding=1))


        # for test extra layers
        # self.con_first = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.con = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        # self.con_end = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, cover, x, flag):
        measures = self.sampling(x)
        # x = swish(self.upsampling1(measures))

        if flag==0:
            rands = torch.rand(measures.shape).cuda()
            rands = rands/torch.max(rands)/1000
            # print rands
            measures.data = measures.data + rands
            x = self.upsampling1(measures)
            # rands = torch.rand(x.size).cuda()
            # rands = Variable(torch.rand(measures.shape).cuda())

            # print rands
            x = My_Reshape_Adap(x, self.blocksize)
            block11 = self.block11(x)
            block12 = self.block12(block11)
            block13 = self.block13(block12)
            block14 = self.block14(block13+block11)
            block15 = self.block15(block14)
            return measures, block15
        
        if flag==1:
            x = self.upsampling2(measures)
            x = My_Reshape_Adap(x, self.blocksize)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)

            return measures, block29, block37


class CSNet_Color_Gray(nn.Module):
    def __init__(self, scale_factor, subrate):
        #upsample_block_num = int(math.log(scale_factor, 2))

        super(CSNet_Color_Gray, self).__init__()

        # for sampling
        self.blocksize = 32/scale_factor
        self.sampling = nn.Conv2d(1, int(np.round(32*32*subrate)), 32, stride=32, padding=0)

        self.upsampling1 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)
        self.upsampling2 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)

        self.block11 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
 
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block15 = (nn.Conv2d(64, 1, kernel_size=5, padding=2))


        self.block21 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.block22 = ResidualBlock(64)
        self.block23 = ResidualBlock(64)
        self.block24 = ResidualBlock(64)
        self.block25 = ResidualBlock(64)
        self.block26 = ResidualBlock(64)
        self.block27 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block28 = (nn.Conv2d(64, 3, kernel_size=9, padding=4))
        
        self.block31 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block32 = ResidualBlock(64)
        self.block33 = ResidualBlock(64)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size=4, stride=4, padding=0),
            nn.PReLU()
        )
        self.block34 = ResidualBlock(256)
        self.block35 = ResidualBlock(256)
        self.block36 = ResidualBlock(256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, int(np.round(self.blocksize*self.blocksize*subrate*4)), kernel_size=8, stride=8, padding=0),
            nn.PReLU()
        )
        self.block37 = (nn.Conv2d(int(np.round(self.blocksize*self.blocksize*subrate*4)), int(np.round(self.blocksize*self.blocksize*subrate)), kernel_size=3, padding=1))


        # for test extra layers
        # self.con_first = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.con = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        # self.con_end = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, cover, x, flag):
        measures = self.sampling(x)
        # x = swish(self.upsampling1(measures))
        

        if flag==0:
            rands = torch.rand(measures.shape).cuda()
            rands = rands/torch.max(rands)/1000
            # print rands
            measures.data = measures.data + rands
            x = self.upsampling1(measures)
            # rands = torch.rand(x.size).cuda()
            # rands = Variable(torch.rand(measures.shape).cuda())

            # print rands
            x = My_Reshape_Adap(x, self.blocksize)
            block11 = self.block11(x)
            block12 = self.block12(block11)
            block13 = self.block13(block12)
            block14 = self.block14(block13+block11)
            block15 = self.block15(block14)
            return measures, block15
        
        if flag==1:
            x = self.upsampling2(measures)
            x = My_Reshape_Adap(x, self.blocksize)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)

            return measures, block29, block37




class CSNet_Color(nn.Module):
    def __init__(self, scale_factor, subrate):
        #upsample_block_num = int(math.log(scale_factor, 2))

        super(CSNet_Color, self).__init__()

        # for sampling
        self.blocksize = 32/scale_factor
        self.sampling = nn.Conv2d(3, int(np.round(32*32*3*subrate)), 32, stride=32, padding=0)

        self.upsampling1 = nn.Conv2d(int(np.round(32*32*3*subrate)), self.blocksize*self.blocksize*3, 1, stride=1, padding=0)
        self.upsampling2 = nn.Conv2d(int(np.round(32*32*3*subrate)), self.blocksize*self.blocksize*3, 1, stride=1, padding=0)

        self.block11 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
 
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block15 = (nn.Conv2d(64, 3, kernel_size=5, padding=2))


        self.block21 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.block22 = ResidualBlock(64)
        self.block23 = ResidualBlock(64)
        self.block24 = ResidualBlock(64)
        self.block25 = ResidualBlock(64)
        self.block26 = ResidualBlock(64)
        self.block27 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block28 = (nn.Conv2d(64, 3, kernel_size=9, padding=4))
        
        self.block31 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block32 = ResidualBlock(64)
        self.block33 = ResidualBlock(64)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size=4, stride=4, padding=0),
            nn.PReLU()
        )
        self.block34 = ResidualBlock(256)
        self.block35 = ResidualBlock(256)
        self.block36 = ResidualBlock(256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, int(np.round(self.blocksize*self.blocksize*subrate*3*4)), kernel_size=8, stride=8, padding=0),
            nn.PReLU()
        )
        self.block37 = (nn.Conv2d(int(np.round(self.blocksize*self.blocksize*subrate*3*4)), int(np.round(self.blocksize*self.blocksize*3*subrate)), kernel_size=3, padding=1))


        # for test extra layers
        # self.con_first = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.con = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        # self.con_end = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, cover, x, flag):
        measures = self.sampling(x)
        # x = swish(self.upsampling1(measures))
        # print torch.max(x)
        # print torch.max(measures)
        # print torch.min(measures)

        if flag==0:
            rands = torch.rand(measures.shape).cuda()
            rands = rands/torch.max(rands)/10000
            # print rands
            measures.data = measures.data + rands
            x = self.upsampling1(measures)
            # rands = torch.rand(x.size).cuda()
            # rands = Variable(torch.rand(measures.shape).cuda())

            # print rands
            x = My_Reshape_Color(x)
            block11 = self.block11(x)
            block12 = self.block12(block11)
            block13 = self.block13(block12)
            block14 = self.block14(block13+block11)
            block15 = self.block15(block14)
            return measures, block15
        
        if flag==1:
            x = self.upsampling2(measures)
            x = My_Reshape_Color(x)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)

            return measures, block29, block37



class CSNet2(nn.Module):
    def __init__(self, scale_factor, subrate):
        #upsample_block_num = int(math.log(scale_factor, 2))

        super(CSNet2, self).__init__()

        # for sampling
        self.blocksize = 32/scale_factor
        self.sampling = nn.Conv2d(1, int(np.round(32*32*subrate)), 32, stride=32, padding=0)

        self.upsampling1 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)
        self.upsampling2 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)

        self.block11 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
 
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block15 = (nn.Conv2d(64, 1, kernel_size=5, padding=2))


        self.block21 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.block22 = ResidualBlock(64)
        self.block23 = ResidualBlock(64)
        self.block24 = ResidualBlock(64)
        self.block25 = ResidualBlock(64)
        self.block26 = ResidualBlock(64)
        self.block27 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block28 = (nn.Conv2d(64, 1, kernel_size=9, padding=4))
        
        self.block31 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block32 = ResidualBlock(64)
        self.block33 = ResidualBlock(64)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size=4, stride=4, padding=0),
            nn.PReLU()
        )
        self.block34 = ResidualBlock(256)
        self.block35 = ResidualBlock(256)
        self.block36 = ResidualBlock(256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, int(np.round(self.blocksize*self.blocksize*subrate*2)), kernel_size=8, stride=8, padding=0),
            nn.PReLU()
        )
        self.block37 = (nn.Conv2d(int(np.round(self.blocksize*self.blocksize*subrate*2)), int(np.round(self.blocksize*self.blocksize*subrate)), kernel_size=3, padding=1))
       
 
        self.upsampling3 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)
        self.block41 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.block42 = ResidualBlock(64)
        self.block43 = ResidualBlock(64)
        self.block44 = ResidualBlock(64)
        self.block45 = ResidualBlock(64)
        self.block46 = ResidualBlock(64)
        self.block47 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block48 = (nn.Conv2d(64, 1, kernel_size=9, padding=4))
        

        # for test extra layers
        # self.con_first = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.con = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        # self.con_end = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, cover, x, flag):
        measures = self.sampling(x)
        # x = swish(self.upsampling1(measures))

        if flag==0:
            rands = torch.rand(measures.shape).cuda()
            rands = rands/torch.max(rands)/100
            # print rands
            measures.data = measures.data + rands
            x = self.upsampling1(measures)
            # rands = torch.rand(x.size).cuda()
            # rands = Variable(torch.rand(measures.shape).cuda())

            # print rands
            x = My_Reshape_Adap(x, self.blocksize)
            block11 = self.block11(x)
            block12 = self.block12(block11)
            block13 = self.block13(block12)
            block14 = self.block14(block13+block11)
            block15 = self.block15(block14)
            return measures, block15
        
        if flag==1:
            x = self.upsampling2(measures)
            x = My_Reshape_Adap(x, self.blocksize)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)

            return measures, block29, block37

        if flag==2:
            x = self.upsampling2(measures)
            x = My_Reshape_Adap(x, self.blocksize)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)
           
            x = self.upsampling3(block37)
            x = My_Reshape_Adap(x, self.blocksize)
            block41 = self.block41(x)
            block42 = self.block42(block41)
            block43 = self.block43(block42)
            block44 = self.block44(block43)
            block45 = self.block45(block44)
            block46 = self.block46(block45)
            block47 = self.block47(block46)
            block48 = self.block48(block41 + block47)
            block49 = (F.tanh(block48)+1)/2

            return block49

class CSNet3(nn.Module):
    def __init__(self, scale_factor, subrate):
        #upsample_block_num = int(math.log(scale_factor, 2))

        super(CSNet3, self).__init__()

        # for sampling
        self.blocksize = 32/scale_factor
        self.sampling = nn.Conv2d(1, int(np.round(32*32*subrate)), 32, stride=32, padding=0)

        self.upsampling1 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)
        self.upsampling2 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)

        self.block11 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
 
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block15 = (nn.Conv2d(64, 1, kernel_size=5, padding=2))


        self.block21 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.block22 = ResidualBlock(64)
        self.block23 = ResidualBlock(64)
        self.block24 = ResidualBlock(64)
        self.block25 = ResidualBlock(64)
        self.block26 = ResidualBlock(64)
        self.block27 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block28 = (nn.Conv2d(64, 1, kernel_size=9, padding=4))
        
        self.block31 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block32 = ResidualBlock(64)
        self.block33 = ResidualBlock(64)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size=4, stride=4, padding=0),
            nn.PReLU()
        )
        self.block34 = ResidualBlock(256)
        self.block35 = ResidualBlock(256)
        self.block36 = ResidualBlock(256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, int(np.round(self.blocksize*self.blocksize*subrate*2)), kernel_size=8, stride=8, padding=0),
            nn.PReLU()
        )
        self.block37 = (nn.Conv2d(int(np.round(self.blocksize*self.blocksize*subrate*2)), int(np.round(self.blocksize*self.blocksize*subrate)), kernel_size=3, padding=1))
       
        self.Enhanced = Enhanced(scale_factor, subrate)

    def forward(self, cover, x, flag):
        measures = self.sampling(x)
        # x = swish(self.upsampling1(measures))

        if flag==0:
            rands = torch.rand(measures.shape).cuda()
            rands = rands/torch.max(rands)/100
            # print rands
            measures.data = measures.data + rands
            x = self.upsampling1(measures)
            # rands = torch.rand(x.size).cuda()
            # rands = Variable(torch.rand(measures.shape).cuda())

            # print rands
            x = My_Reshape_Adap(x, self.blocksize)
            block11 = self.block11(x)
            block12 = self.block12(block11)
            block13 = self.block13(block12)
            block14 = self.block14(block13+block11)
            block15 = self.block15(block14)
            return measures, block15
        
        if flag==1:
            x = self.upsampling2(measures)
            x = My_Reshape_Adap(x, self.blocksize)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)

            return measures, block29, block37

        if flag==2:
            x = self.upsampling2(measures)
            x = My_Reshape_Adap(x, self.blocksize)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)
          
            res = self.Enhanced(block37)
 
            return res



class CS_Steg_Gray(nn.Module):
    def __init__(self, subrate):

        super(CS_Steg_Gray, self).__init__()

        # compressed sensing block size
        self.blocksize = 32 
		
		##################### SMG Module #######################
		
		# for image sampling
        self.sampling = nn.Conv2d(1, int(np.round(32*32*subrate)), 32, stride=32, padding=0)

        # for upsampling in SMG module.
        self.upsampling1 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)

        self.block11 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
 
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block15 = (nn.Conv2d(64, 1, kernel_size=5, padding=2))

        ##################### Steg Module #######################
		
		# Hiding Network
		
        self.upsampling2 = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)
        self.block21 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.block22 = ResidualBlock(64)
        self.block23 = ResidualBlock(64)
        self.block24 = ResidualBlock(64)
        self.block25 = ResidualBlock(64)
        self.block26 = ResidualBlock(64)
        self.block27 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block28 = (nn.Conv2d(64, 1, kernel_size=9, padding=4))
        
		# Distillation Network
		
        self.block31 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block32 = ResidualBlock(64)
        self.block33 = ResidualBlock(64)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size=4, stride=4, padding=0),
            nn.PReLU()
        )
        self.block34 = ResidualBlock(256)
        self.block35 = ResidualBlock(256)
        self.block36 = ResidualBlock(256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, int(np.round(self.blocksize*self.blocksize*subrate*2)), kernel_size=8, stride=8, padding=0),
            nn.PReLU()
        )
        self.block37 = (nn.Conv2d(int(np.round(self.blocksize*self.blocksize*subrate*2)), int(np.round(self.blocksize*self.blocksize*subrate)), kernel_size=3, padding=1))
       
	   
	    ##################### Recon Module #######################
		
        self.Enhanced = Enhanced(scale_factor, subrate)

    def forward(self, cover, x, flag):
        measures = self.sampling(x)
        # x = swish(self.upsampling1(measures))

        if flag==0:
            rands = torch.rand(measures.shape).cuda()
            rands = rands/torch.max(rands)/1000
            # print rands
            measures.data = measures.data + rands
            x = self.upsampling1(measures)
            # rands = torch.rand(x.size).cuda()
            # rands = Variable(torch.rand(measures.shape).cuda())

            # print rands
            x = My_Reshape_Adap(x, self.blocksize)
            block11 = self.block11(x)
            block12 = self.block12(block11)
            block13 = self.block13(block12)
            block14 = self.block14(block13+block11)
            block15 = self.block15(block14)
            return measures, block15
        
        if flag==1:
            x = self.upsampling2(measures)
            x = My_Reshape_Adap(x, self.blocksize)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)

            return measures, block29, block37

        if flag==2:
            x = self.upsampling2(measures)
            x = My_Reshape_Adap(x, self.blocksize)
            x = torch.cat([cover, x], 1)
            block21 = self.block21(x)
            block22 = self.block22(block21)
            block23 = self.block23(block22)
            block24 = self.block24(block23)
            block25 = self.block25(block24)
            block26 = self.block26(block25)
            block27 = self.block27(block26)
            block28 = self.block28(block21 + block27)
            block29 = (F.tanh(block28)+1)/2
            
            block31 = self.block31(block29)
            block32 = self.block32(block31)
            block33 = self.block33(block32)
            block33 = self.downsample1(block33+block31)
            block34 = self.block34(block33)
            block35 = self.block35(block34)
            block36 = self.block36(block35)
            block36 = self.downsample2(block36+block33)
            block37 = self.block37(block36)
          
            res = self.Enhanced(block37)
 
            return res


class Enhanced_temp(nn.Module):
    def __init__(self, channels, has_BN = False):
        super(Enhanced_temp, self).__init__()
        self.has_BN = has_BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        if self.has_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if self.has_BN:
            residual = self.bn2(residual)

        return x + residual




class Enhanced(nn.Module):
    def __init__(self, scale_factor, subrate):
        #upsample_block_num = int(math.log(scale_factor, 2))

        super(Enhanced, self).__init__()

        # for sampling
        self.blocksize = 32/scale_factor

        self.upsampling = nn.Conv2d(int(np.round(32*32*subrate)), self.blocksize*self.blocksize, 1, stride=1, padding=0)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64, has_BN=True)
        self.block3 = ResidualBlock(64, has_BN=True)
        self.block4 = ResidualBlock(64, has_BN=True)
        self.block5 = ResidualBlock(64, has_BN=True)
        self.block6 = ResidualBlock(64, has_BN=True)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block8 = (nn.Conv2d(64, 1, kernel_size=7, padding=3))
        

    def forward(self, x):
        # measures = self.sampling(x)
        # x = swish(self.upsampling1(measures))

        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7 + block1)
        res = (F.tanh(block8)+1)/2
            
        return res



class Generator_Adap(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_Adap, self).__init__()

        # conv blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

        # for test extra layers
        # self.con_first = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.con = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        # self.con_end = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        # x = My_SQ(x, self.bitdepth)
        batch_size = x.shape[0]
        channels = x.shape[1]
        sides = channels ** 0.5
        sides = int(sides)
        wides = x.shape[2]
        highs = x.shape[3]
        temp = []
        temp2 = []
        for i in range(0, batch_size):
            for h in range(0, highs):
                if h == 1:
                    temp2 = temp
                elif h > 1:
                    temp2 = torch.cat((temp2, temp), 0)
                for w in range(0, wides):
                    if w == 0:
                        temp = x[i, :, w, h].contiguous().view(sides, sides)
                    else:
                        temp = torch.cat((temp, x[i, :, w, h].contiguous().view(sides, sides)), 1)
            temp2 = torch.cat((temp2, temp), 0)




            # for j in range(0, wides):
            #     for k in range(0, highs):
            #         if k == 0:
            #             if j == 1:
            #                 temp2 = temp
            #             elif j > 1:
            #                 temp2 = torch.cat((temp2, temp), 0)
            #             temp = x[i, :, j, k].contiguous().view(sides, sides)
            #         else:
            #             temp = torch.cat((temp, x[i, :, j, k].contiguous().view(sides, sides)), 1)
            # temp2 = torch.cat((temp2, temp), 0)


            temp2 = temp2.unsqueeze(0)
            temp2 = ToPILImage()(temp2.data.cpu())
            temp2.save('temps.png', 'PNG', quality=100)
            os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q ' + str(
                        self.bitdepth) + ' -o temps.bpg ./temps.png' + ' > /dev/null 2>&1')
            os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temps.png ./temps.bpg' + ' > /dev/null 2>&1')
            temp = Image.open('temps.png')
            temp = ToTensor()(temp)
            temp = temp[0, :, :]
            temp = temp.squeeze()

            for j in range(0, highs):
                for k in range(0, wides):
                    d = temp[j*sides:j*sides+sides, k*sides:k*sides+sides]
                    x.data[i,:,k,j] = d.contiguous().view(-1).cuda()


            # for j in range(0, wides):
            #     for k in range(0, highs):
            #         d = temp[j*sides:j*sides+sides, k*sides:k*sides+sides]
            #         x.data[i,:,j,k] = d.contiguous().view(-1).cuda()


        x = self.upsampling(x)
        x = My_Reshape(x)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2

class Generator_temp(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        # x = My_SQ(x, self.bitdepth)
        batch_size = x.shape[0]
        channels = x.shape[1]
        sides = channels ** 0.5
        sides = int(sides)
        wides = x.shape[2]
        highs = x.shape[3]
        temp = []
        temp2 = []
        # for i in range(0, batch_size):
        #     for j in range(0, wides):
        #         for k in range(0, highs):
        #             if k == 0:
        #                 if j == 1:
        #                     temp2 = temp
        #                 elif j > 1:
        #                     temp2 = torch.cat((temp2, temp), 0)
        #                 temp = x[i, :, j, k].contiguous().view(sides, sides)
        #             else:
        #                 temp = torch.cat((temp, x[i, :, j, k].contiguous().view(sides, sides)), 1)
        #     temp2 = torch.cat((temp2, temp), 0)

        for i in range(0, batch_size):
            for h in range(0, highs):
                if h == 1:
                    temp2 = temp
                elif h > 1:
                    temp2 = torch.cat((temp2, temp), 0)
                for w in range(0, wides):
                    if w == 0:
                        temp = x[i, :, w, h].contiguous().view(sides, sides)
                    else:
                        temp = torch.cat((temp, x[i, :, w, h].contiguous().view(sides, sides)), 1)
            temp2 = torch.cat((temp2, temp), 0)

            temp2 = temp2.unsqueeze(0)
            temp2 = ToPILImage()(temp2.data.cpu())
            temp2.save('temps_temps.png', 'PNG', quality=100)
            os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q ' + str(
                        self.bitdepth) + ' -o temps_temps.bpg ./temps_temps.png' + ' > /dev/null 2>&1')
            os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temps_temps.png ./temps_temps.bpg' + ' > /dev/null 2>&1')
            temp = Image.open('temps_temps.png')
            temp = ToTensor()(temp)
            temp = temp[0, :, :]
            temp = temp.squeeze()

            for j in range(0, highs):
                for k in range(0, wides):
                    d = temp[j*sides:j*sides+sides, k*sides:k*sides+sides]
                    x.data[i,:,k,j] = d.contiguous().view(-1).cuda()

            # for j in range(0, wides):
            #     for k in range(0, highs):
            #         d = temp[j*sides:j*sides+sides, k*sides:k*sides+sides]
            #         x.data[i,:,j,k] = d.contiguous().view(-1).cuda()


        x = self.upsampling(x)
        x = My_Reshape(x)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_temp_temp(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_temp_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)

        self.upsampling_block1 = nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), blocksize * blocksize * 8, 1, stride=1,
                                    padding=0)
        self.upsampling_block2 = nn.Conv2d(blocksize * blocksize * 8, blocksize * blocksize * 8, 1,
                                           stride=1,
                                           padding=0)
        self.upsampling_block3 = nn.Conv2d(blocksize * blocksize * 8, blocksize * blocksize * 8, 1,
                                           stride=1,
                                           padding=0)
        self.upsampling = nn.Conv2d(blocksize*blocksize*8, blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        # x = My_SQ(x, self.bitdepth)
        batch_size = x.shape[0]
        channels = x.shape[1]
        sides = channels ** 0.5
        sides = int(sides)
        wides = x.shape[2]
        highs = x.shape[3]
        temp = []
        temp2 = []
        for i in range(0, batch_size):
            for j in range(0, wides):
                for k in range(0, highs):
                    if k == 0:
                        if j == 1:
                            temp2 = temp
                        elif j > 1:
                            temp2 = torch.cat((temp2, temp), 0)
                        temp = x[i, :, j, k].contiguous().view(sides, sides)
                    else:
                        temp = torch.cat((temp, x[i, :, j, k].contiguous().view(sides, sides)), 1)
            temp2 = torch.cat((temp2, temp), 0)


            temp2 = temp2.unsqueeze(0)
            temp2 = ToPILImage()(temp2.data.cpu())
            temp2.save('temps_temp_temps.png', 'PNG', quality=100)
            os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q ' + str(
                        self.bitdepth) + ' -o temps_temp_temps.bpg ./temps_temp_temps.png' + ' > /dev/null 2>&1')
            os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temps_temp_temps.png ./temps_temp_temps.bpg' + ' > /dev/null 2>&1')
            temp = Image.open('temps_temp_temps.png')
            temp = ToTensor()(temp)
            temp = temp[0, :, :]
            temp = temp.squeeze()

            for j in range(0, wides):
                for k in range(0, highs):
                    d = temp[j*sides:j*sides+sides, k*sides:k*sides+sides]
                    x.data[i,:,j,k] = d.contiguous().view(-1).cuda()

        x = self.upsampling_block1(x)
        x = self.upsampling_block2(x)
        x = self.upsampling_block3(x)
        x = self.upsampling(x)
        x = My_Reshape(x)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_test(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_test, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        # x = My_SQ(x, self.bitdepth)
        batch_size = x.shape[0]
        channels = x.shape[1]
        sides = channels ** 0.5
        sides = int(sides)
        wides = x.shape[2]
        highs = x.shape[3]
        temp = []
        temp2 = []
        file_size = 0
        for i in range(0, batch_size):
            for h in range(0, highs):
                if h == 1:
                    temp2 = temp
                elif h > 1:
                    temp2 = torch.cat((temp2, temp), 0)
                for w in range(0, wides):
                    if w == 0:
                        temp = x[i, :, w, h].contiguous().view(sides, sides)
                    else:
                        temp = torch.cat((temp, x[i, :, w, h].contiguous().view(sides, sides)), 1)
            temp2 = torch.cat((temp2, temp), 0)
        # for i in range(0, batch_size):
        #     for j in range(0, wides):
        #         for k in range(0, highs):
        #             if k == 0:
        #                 if j == 1:
        #                     temp2 = temp
        #                 elif j > 1:
        #                     temp2 = torch.cat((temp2, temp), 0)
        #                 temp = x[i, :, j, k].contiguous().view(sides, sides)
        #             else:
        #                 temp = torch.cat((temp, x[i, :, j, k].contiguous().view(sides, sides)), 1)
        #     temp2 = torch.cat((temp2, temp), 0)


            temp2 = temp2.unsqueeze(0)
            temp2 = ToPILImage()(temp2.data.cpu())
            temp2.save('test.png', 'PNG', quality=100)
            os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q ' + str(
                        self.bitdepth) + ' -o test.bpg ./test.png' + ' > /dev/null 2>&1')
            os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./test.png ./test.bpg' + ' > /dev/null 2>&1')
            temp = Image.open('test.png')
            file_size = os.path.getsize('test.bpg')
            temp = ToTensor()(temp)
            temp = temp[0, :, :]
            temp = temp.squeeze()

            for j in range(0, highs):
                for k in range(0, wides):
                    d = temp[j*sides:j*sides+sides, k*sides:k*sides+sides]
                    x.data[i,:,k,j] = d.contiguous().view(-1).cuda()


        x = self.upsampling(x)
        x = My_Reshape(x)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        bpp = file_size * 8.0/32/32/wides/highs
        print bpp

        return (F.tanh(block8) + 1) / 2, bpp

class Generator_rate_control_loss(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_rate_control_loss, self).__init__()
        self.blocksize = blocksize
        self.bitdepth = bitdepth
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # for the SQ offset computation
        # channels = int(np.round(blocksize * blocksize * subrate))
        # self.block1_sq = ResidualBlock(channels, has_BN=True)
        # self.block2_sq = ResidualBlock(channels, has_BN=True)
        # self.block3_sq = ResidualBlock(channels, has_BN=True)
        # self.block4_sq = ResidualBlock(channels, has_BN=True)

        # self.block1_sq = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), int(np.round(blocksize*blocksize*subrate)), 3, stride=1, padding=1)
        # self.block2_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block3_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block4_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())

        # self.block5_sq = nn.Conv2d(channels, channels, 3, stride=1,
        #                           padding=1)

        # for the upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        meas = x
        # x = swish(self.upsampling(x))
        x = My_SQ(x, self.bitdepth)
        # sq = self.block1_sq(x)
        # sq = self.block2_sq(sq)
        # sq = self.block3_sq(sq)
        # sq = self.block4_sq(sq)
        # sq = self.block5_sq(sq + x)
        #
        # if(False):
        #     sq_temp = sq.data.cpu()
        #     ToPILImage()(sq_temp).show()

        # sq = sq + stepsize
        # x = x*sq
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return meas, (F.tanh(block8) + 1) / 2


class Generator_SQ(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_SQ, self).__init__()
        self.blocksize = blocksize
        self.bitdepth = bitdepth
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # for the SQ offset computation
        channels = int(np.round(blocksize * blocksize * subrate))
        self.block1_sq = ResidualBlock(channels, has_BN=True)
        self.block2_sq = ResidualBlock(channels, has_BN=True)
        self.block3_sq = ResidualBlock(channels, has_BN=True)
        self.block4_sq = ResidualBlock(channels, has_BN=True)

        # self.block1_sq = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), int(np.round(blocksize*blocksize*subrate)), 3, stride=1, padding=1)
        # self.block2_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block3_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block4_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())

        self.block5_sq = nn.Conv2d(channels, channels, 3, stride=1,
                                   padding=1)

        # for the upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        x, stepsize = My_SQ_ONLY(x, self.bitdepth)
        sq = self.block1_sq(x)
        sq = self.block2_sq(sq)
        sq = self.block3_sq(sq)
        sq = self.block4_sq(sq)
        sq = self.block5_sq(sq + x)

        if(False):
            sq_temp = sq.data.cpu()
            ToPILImage()(sq_temp).show()

        sq = sq + stepsize
        x = x*sq
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_SQ_rate_control_loss(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_SQ_rate_control_loss, self).__init__()
        self.blocksize = blocksize
        self.bitdepth = bitdepth
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # for the SQ offset computation
        channels = int(np.round(blocksize * blocksize * subrate))
        self.block1_sq = ResidualBlock(channels, has_BN=True)
        self.block2_sq = ResidualBlock(channels, has_BN=True)
        self.block3_sq = ResidualBlock(channels, has_BN=True)
        self.block4_sq = ResidualBlock(channels, has_BN=True)

        # self.block1_sq = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), int(np.round(blocksize*blocksize*subrate)), 3, stride=1, padding=1)
        # self.block2_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block3_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block4_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())

        self.block5_sq = nn.Conv2d(channels, channels, 3, stride=1,
                                   padding=1)

        # for the upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        meas = x
        # x = swish(self.upsampling(x))
        x, stepsize = My_SQ_ONLY(x, self.bitdepth)
        sq = self.block1_sq(x)
        sq = self.block2_sq(sq)
        sq = self.block3_sq(sq)
        sq = self.block4_sq(sq)
        sq = self.block5_sq(sq + x)

        if(False):
            sq_temp = sq.data.cpu()
            ToPILImage()(sq_temp).show()

        sq = sq + stepsize
        x = x*sq
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return meas, (F.tanh(block8) + 1) / 2

        # return (F.tanh(block8) + 1) / 2


class Generator_SQ_rate_control_loss2(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_SQ_rate_control_loss2, self).__init__()
        self.blocksize = blocksize
        self.bitdepth = bitdepth
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # for the SQ offset computation
        channels = int(np.round(blocksize * blocksize * subrate))
        self.block1_sq = ResidualBlock(channels, has_BN=True)
        self.block2_sq = ResidualBlock(channels, has_BN=True)
        self.block3_sq = ResidualBlock(channels, has_BN=True)
        self.block4_sq = ResidualBlock(channels, has_BN=True)

        # self.block1_sq = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), int(np.round(blocksize*blocksize*subrate)), 3, stride=1, padding=1)
        # self.block2_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block3_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block4_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())

        self.block5_sq = nn.Conv2d(channels, channels, 3, stride=1,
                                   padding=1)

        # for the upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        meas = x
        # x = swish(self.upsampling(x))
        x, stepsize = My_SQ_ONLY2(x, self.bitdepth)
        sq = self.block1_sq(x)
        sq = self.block2_sq(sq)
        sq = self.block3_sq(sq)
        sq = self.block4_sq(sq)
        sq = self.block5_sq(sq + x)

        if(False):
            sq_temp = sq.data.cpu()
            ToPILImage()(sq_temp).show()

        # sq = sq + stepsize

        channels = sq.shape[0]
        for i in range(0,channels):
            sq[i,:,:,:] = sq[i,:,:,:] + stepsize[i]

        x = x*sq
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return meas, (F.tanh(block8) + 1) / 2

        # return (F.tanh(block8) + 1) / 2


class Generator_SQ_rate_control_loss_test(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_SQ_rate_control_loss_test, self).__init__()
        self.blocksize = blocksize
        self.bitdepth = bitdepth
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # for the SQ offset computation
        channels = int(np.round(blocksize * blocksize * subrate))
        self.block1_sq = ResidualBlock(channels, has_BN=True)
        self.block2_sq = ResidualBlock(channels, has_BN=True)
        self.block3_sq = ResidualBlock(channels, has_BN=True)
        self.block4_sq = ResidualBlock(channels, has_BN=True)

        # self.block1_sq = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), int(np.round(blocksize*blocksize*subrate)), 3, stride=1, padding=1)
        # self.block2_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block3_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block4_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())

        self.block5_sq = nn.Conv2d(channels, channels, 3, stride=1,
                                   padding=1)

        # for the upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        meas = x
        # x = swish(self.upsampling(x))
        x, stepsize, bpp = My_SQ_ONLY_TEST(x, self.bitdepth)
        sq = self.block1_sq(x)
        sq = self.block2_sq(sq)
        sq = self.block3_sq(sq)
        sq = self.block4_sq(sq)
        sq = self.block5_sq(sq + x)

        if(False):
            sq_temp = sq.data.cpu()
            ToPILImage()(sq_temp).show()

        sq = sq + stepsize
        x = x*sq
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return meas, (F.tanh(block8) + 1) / 2, bpp


class Generator_SQ_2(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_SQ_2, self).__init__()
        self.blocksize = blocksize
        self.bitdepth = bitdepth
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize * blocksize * subrate)), blocksize, stride=blocksize,
                                  padding=0, bias=False)
        # for the sq computation
        channels = int(np.round(blocksize * blocksize * subrate))
        self.block1_sq = ResidualBlock(channels, has_BN=True)
        self.block2_sq = ResidualBlock(channels, has_BN=True)
        self.block3_sq = ResidualBlock(channels, has_BN=True)
        self.block4_sq = ResidualBlock(channels, has_BN=True)

        # self.block1_sq = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), int(np.round(blocksize*blocksize*subrate)), 3, stride=1, padding=1)
        # self.block2_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block3_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())
        # self.block4_sq = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), int(np.round(blocksize * blocksize * subrate)), 3, stride=1,
        #               padding=1),
        #     nn.PReLU())

        self.block5_sq = nn.Conv2d(channels, channels, 3, stride=1,
                                   padding=1)

        # for the upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), blocksize * blocksize, 1, stride=1,
                                    padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

        # for the enhance blocks in the networks
        self.block_e1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block_e2 = ResidualBlock(64)
        self.block_e3 = ResidualBlock(64)
        self.block_e4 = ResidualBlock(64)
        self.block_e5 = ResidualBlock(64)
        self.block_e6 = ResidualBlock(64)
        self.block_e7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block_e8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block_e8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block_e8 = nn.Sequential(*block_e8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        x, stepsize = My_SQ_ONLY(x, self.bitdepth)
        sq = self.block1_sq(x)
        sq = self.block2_sq(sq)
        sq = self.block3_sq(sq)
        sq = self.block4_sq(sq)
        sq = self.block5_sq(sq + x)

        sq = sq + stepsize
        x = x * sq
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        temp = (F.tanh(block8) + 1) / 2

        temp1, temp2 = My_Branch(temp, 24.0/25.0)

        # for the enhance blocks

        block_e1 = self.block_e1(temp2)
        block_e2 = self.block_e2(block_e1)
        block_e3 = self.block_e3(block_e2)
        block_e4 = self.block_e4(block_e3)
        block_e5 = self.block_e5(block_e4)
        block_e6 = self.block_e6(block_e5)
        block_e7 = self.block_e7(block_e6)
        block_e8 = self.block_e8(block_e1 + block_e7)

        return temp1, (F.tanh(block_e8) + 1) / 2


class Generator_DPCM(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_DPCM, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        x = My_DPCM_SQ(x, self.bitdepth)
        x = self.upsampling(x)
        # x = My_Reshape(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_LSMM(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_LSMM, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        x = My_LSMM_SQ(x, self.bitdepth)
        x = self.upsampling(x)
        # x = My_Reshape(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_LSMM_2(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_LSMM_2, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        x = My_LSMM_SQ_2(x, self.bitdepth)
        x = self.upsampling(x)
        # x = My_Reshape(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD(nn.Module):
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_STD, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # x = swish(self.upsampling(x))
        x = My_LSMM_SQ_2(x, self.bitdepth)
        x = self.upsampling(x)
        # x = My_Reshape(x)
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2



class Generator_STD2(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD2, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 3) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]

        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_JPEG(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_JPEG, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 3) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp.png', 'PNG', quality=100)
                os.system('../openjpeg-master/build/bin/opj_compress -i ./temp.png -o ./temp.j2k -r 1' + ' > /dev/null 2>&1')
                os.system('../openjpeg-master/build/bin/opj_decompress -i ./temp.j2k -o ./temp.png' + ' > /dev/null 2>&1')
                temp = Image.open('temp.png')
                temp = ToTensor()(temp)
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_BPG(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor
        super(Generator_STD_BPG, self).__init__()
        upsample = [blocksize]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize/2]
        else:
            upsample = [2, 2, 2]
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, upsample[_]) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                # temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp.png', 'PNG', quality=100)
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o temp.bpg ./temp.png' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temp.png ./temp.bpg' + ' > /dev/null 2>&1')
                temp = Image.open('temp.png')
                temp = ToTensor()(temp)
                temp = temp[0, :, :]
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_BPG_ReCon(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor
        super(Generator_STD_BPG_ReCon, self).__init__()
        upsample = [blocksize]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize/2]
        else:
            upsample = [2, 2, 2]
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        kernel_size = 3
        padding = 1
        features = 64
        channels = 1
        num_of_layers = 17
        layers = []
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)

        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                # temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp_ReCon.png', 'PNG', quality=100)
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o temp_ReCon.bpg ./temp_ReCon.png' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temp_ReCon.png ./temp_ReCon.bpg' + ' > /dev/null 2>&1')
                temp = Image.open('temp_ReCon.png')
                temp = ToTensor()(temp)
                temp = temp[0, :, :]
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        out = self.dncnn(x)
        return out


class Generator_STD_BPG_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor
        super(Generator_STD_BPG_temp, self).__init__()
        upsample = [blocksize]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize/2]
        else:
            upsample = [2, 2, 2]
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, upsample[_]) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                # temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp_temp.png', 'PNG', quality=100)
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o temp_temp.bpg ./temp_temp.png' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temp_temp.png ./temp_temp.bpg' + ' > /dev/null 2>&1')
                temp = Image.open('temp_temp.png')
                temp = ToTensor()(temp)
                temp = temp[0, :, :]
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_BPG_ReCon_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor
        super(Generator_STD_BPG_ReCon_temp, self).__init__()
        upsample = [blocksize]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize/2]
        else:
            upsample = [2, 2, 2]
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        kernel_size = 3
        padding = 1
        features = 64
        channels = 1
        num_of_layers = 17
        layers = []
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)

        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            # layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                # temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp_ReCon_temp.png', 'PNG', quality=100)
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o temp_ReCon_temp.bpg ./temp_ReCon_temp.png' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temp_ReCon_temp.png ./temp_ReCon_temp.bpg' + ' > /dev/null 2>&1')
                temp = Image.open('temp_ReCon_temp.png')
                temp = ToTensor()(temp)
                temp = temp[0, :, :]
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        out = self.dncnn(x)
        return out


class Generator_STD_BPG_temp_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor
        super(Generator_STD_BPG_temp_temp, self).__init__()
        upsample = [blocksize]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize/2]
        else:
            upsample = [2, 2, 2]
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, upsample[_]) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                # temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp_temp_temp.png', 'PNG', quality=100)
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o temp_temp_temp.bpg ./temp_temp_temp.png' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temp_temp_temp.png ./temp_temp_temp.bpg' + ' > /dev/null 2>&1')
                temp = Image.open('temp_temp_temp.png')
                temp = ToTensor()(temp)
                temp = temp[0, :, :]
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_HEVC(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_HEVC, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        # self.spatialSoftmax = nn.Softmax()
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        if False:
            batch_size = x.shape[0]
            channels = x.shape[1]
            # print channels
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('results1/' + str(j) + '.png', 'PNG', quality=100)
                w = x.shape[2]
                h = x.shape[3]
                os.system('ffmpeg -i results1/%d.png -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -y result.yuv' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/HM-16.18/bin/TAppEncoderStatic -c cfg/encoder_intra_main_rext.cfg -c cfg/test_400_8bit.cfg -wdt ' + str(h) + ' -hgt ' + str(w) + ' -q ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                os.system('ffmpeg -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -i rec.yuv -y results2/%d.png' + ' > /dev/null 2>&1')
                for j in range(0, channels):
                    temp = Image.open('results2/' + str(j+1) + '.png')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()
                    # print x.shape, temp.shape
                    x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        # block1_ = self.spatialSoftmax(block1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_HEVC_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_HEVC_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        if False:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('results3/' + str(j) + '.png', 'PNG', quality=100)
                w = x.shape[2]
                h = x.shape[3]
                os.system('ffmpeg -i results3/%d.png -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -y result_temp.yuv' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/HM-16.18/bin/TAppEncoderStatic -c cfg/encoder_lowdelay_main_rext_temp.cfg -c cfg/test_400_8bit_temp.cfg -wdt ' + str(h) + ' -hgt ' + str(w) + ' -q ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                os.system('ffmpeg -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -i rec_temp.yuv -y results4/%d.png' + ' > /dev/null 2>&1')

                for j in range(0, channels):
                    temp = Image.open('results4/' + str(j+1) + '.png')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_HEVC_temp_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_HEVC_temp_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 3) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                # temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('results5/' + str(j) + '.png', 'PNG', quality=100)
            w = x.shape[2]
            h = x.shape[3]
            os.system('ffmpeg -i results5/%d.png -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -y result_temp_temp.yuv' + ' > /dev/null 2>&1')
            os.system('/home/zyd/cuiwenxue/HM-16.18/bin/TAppEncoderStatic -c cfg/encoder_lowdelay_main_rext_temp_temp.cfg -c cfg/test_400_8bit_temp_temp.cfg -wdt ' + str(h) + ' -hgt ' + str(w) + ' -q ' + str(self.bitdepth) + ' > /dev/null 2>&1')
            os.system('ffmpeg -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -i rec_temp_temp.yuv -y results6/%d.png' + ' > /dev/null 2>&1')

            for j in range(0, channels):
                temp = Image.open('results6/' + str(j+1) + '.png')
                temp = ToTensor()(temp)
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_BPG_Local_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_BPG_Local_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.form  = 1
        self.upscale = 32/(2 ** scale_factor)
        # for sampling
        self.sampling = nn.Conv2d(1, 1, blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.UpsamplingBilinear2d(size=(64, 64), scale_factor=None)
        # nn.UpsamplingBilinear2d(())
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x, sides):
        x = self.sampling(x)
        w_temp = x.shape[2]
        h_temp = x.shape[3]
        w_temp = w_temp/sides*self.upscale
        h_temp = h_temp/sides*self.upscale
        # upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        # x = upsampling(x)
        # img = x[1, 1, :, :]

        if self.form == 1:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    #   print temp.size
                    temp.save('./temp/2.bmp', 'bmp', quality=100)
                    os.system('../openjpeg/build/bin/opj_compress2 -i ./temp/2.bmp -o ./temp/2.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                    os.system('../openjpeg/build/bin/opj_decompress2 -i ./temp/2.j2k -o ./temp/12.bmp' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/12.bmp')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()

                    x.data[i, j, :, :] = temp.cuda()


        if self.form == 0:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('./temp/2.png', 'PNG', quality=100)
                    # fz = os.path.getsize('./temp/7.png')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o ./temp/2.bpg ./temp/2.png' + ' > /dev/null 2>&1')
                    # fz = os.path.getsize('./temp/2.bpg')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgdec -o ./temp/12.png ./temp/2.bpg' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/12.png')
                    temp = ToTensor()(temp)
                    temp = temp[0, :, :]
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()




        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        x = upsampling(x)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_BPG_Local_temp_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_BPG_Local_temp_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.form = 1
        self.upscale = 32/(2 ** scale_factor)
        # for sampling
        self.sampling = nn.Conv2d(1, 1, blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.UpsamplingBilinear2d(size=(64, 64), scale_factor=None)
        # nn.UpsamplingBilinear2d(())
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x, sides):
        x = self.sampling(x)
        w_temp = x.shape[2]
        h_temp = x.shape[3]
        w_temp = w_temp/sides*self.upscale
        h_temp = h_temp/sides*self.upscale
        # upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        # x = upsampling(x)
        # img = x[1, 1, :, :]

        if self.form == 1:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    #   print temp.size
                    temp.save('./temp/3.bmp', 'bmp', quality=100)
                    os.system('../openjpeg/build/bin/opj_compress2 -i ./temp/3.bmp -o ./temp/3.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                    os.system('../openjpeg/build/bin/opj_decompress2 -i ./temp/3.j2k -o ./temp/13.bmp' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/13.bmp')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()

                    x.data[i, j, :, :] = temp.cuda()


        if self.form == 0:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('./temp/3.png', 'PNG', quality=100)
                    # fz = os.path.getsize('./temp/7.png')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o ./temp/3.bpg ./temp/3.png' + ' > /dev/null 2>&1')
                    # fz = os.path.getsize('./temp/3.bpg')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgdec -o ./temp/13.png ./temp/3.bpg' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/13.png')
                    temp = ToTensor()(temp)
                    temp = temp[0, :, :]
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()




        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        x = upsampling(x)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_BPG_Local_temp_temp_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_BPG_Local_temp_temp_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.form = 1
        self.upscale = 32/(2 ** scale_factor)
        # for sampling
        self.sampling = nn.Conv2d(1, 1, blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.UpsamplingBilinear2d(size=(64, 64), scale_factor=None)
        # nn.UpsamplingBilinear2d(())
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x, sides):
        x = self.sampling(x)
        w_temp = x.shape[2]
        h_temp = x.shape[3]
        w_temp = w_temp/sides*self.upscale
        h_temp = h_temp/sides*self.upscale
        # upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        # x = upsampling(x)
        # img = x[1, 1, :, :]

        if self.form == 1:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    #   print temp.size
                    temp.save('./temp/4.bmp', 'bmp', quality=100)
                    os.system('../openjpeg/build/bin/opj_compress2 -i ./temp/4.bmp -o ./temp/4.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                    os.system('../openjpeg/build/bin/opj_decompress2 -i ./temp/4.j2k -o ./temp/14.bmp' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/14.bmp')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()

                    x.data[i, j, :, :] = temp.cuda()


        if self.form == 0:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('./temp/4.png', 'PNG', quality=100)
                    # fz = os.path.getsize('./temp/7.png')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o ./temp/4.bpg ./temp/4.png' + ' > /dev/null 2>&1')
                    # fz = os.path.getsize('./temp/4.bpg')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgdec -o ./temp/14.png ./temp/4.bpg' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/14.png')
                    temp = ToTensor()(temp)
                    temp = temp[0, :, :]
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()




        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        x = upsampling(x)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_BPG_Local_temp_temp_temp_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_BPG_Local_temp_temp_temp_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.form = 1
        self.upscale = 32/(2 ** scale_factor)
        # for sampling
        self.sampling = nn.Conv2d(1, 1, blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.UpsamplingBilinear2d(size=(64, 64), scale_factor=None)
        # nn.UpsamplingBilinear2d(())
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x, sides):
        x = self.sampling(x)
        w_temp = x.shape[2]
        h_temp = x.shape[3]
        w_temp = w_temp/sides*self.upscale
        h_temp = h_temp/sides*self.upscale
        # upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        # x = upsampling(x)
        # img = x[1, 1, :, :]

        if self.form == 1:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    #   print temp.size
                    temp.save('./temp/5.bmp', 'bmp', quality=100)
                    os.system('../openjpeg/build/bin/opj_compress2 -i ./temp/5.bmp -o ./temp/5.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                    os.system('../openjpeg/build/bin/opj_decompress2 -i ./temp/5.j2k -o ./temp/15.bmp' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/15.bmp')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()

                    x.data[i, j, :, :] = temp.cuda()


        if self.form == 0:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('./temp/5.png', 'PNG', quality=100)
                    # fz = os.path.getsize('./temp/7.png')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o ./temp/5.bpg ./temp/5.png' + ' > /dev/null 2>&1')
                    # fz = os.path.getsize('./temp/5.bpg')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgdec -o ./temp/15.png ./temp/5.bpg' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/15.png')
                    temp = ToTensor()(temp)
                    temp = temp[0, :, :]
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()




        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        x = upsampling(x)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_STD_BPG_Local_temp_temp_temp_temp_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_BPG_Local_temp_temp_temp_temp_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.form = 1
        self.upscale = 32/(2 ** scale_factor)
        # for sampling
        self.sampling = nn.Conv2d(1, 1, blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.UpsamplingBilinear2d(size=(64, 64), scale_factor=None)
        # nn.UpsamplingBilinear2d(())
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x, sides):
        x = self.sampling(x)
        w_temp = x.shape[2]
        h_temp = x.shape[3]
        w_temp = w_temp/sides*self.upscale
        h_temp = h_temp/sides*self.upscale
        # upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        # x = upsampling(x)
        # img = x[1, 1, :, :]

        if self.form == 1:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    #   print temp.size
                    temp.save('./temp/6.bmp', 'bmp', quality=100)
                    os.system('../openjpeg/build/bin/opj_compress2 -i ./temp/6.bmp -o ./temp/6.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                    os.system('../openjpeg/build/bin/opj_decompress2 -i ./temp/6.j2k -o ./temp/16.bmp' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/16.bmp')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()

                    x.data[i, j, :, :] = temp.cuda()


        if self.form == 0:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('./temp/6.png', 'PNG', quality=100)
                    # fz = os.path.getsize('./temp/7.png')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o ./temp/6.bpg ./temp/6.png' + ' > /dev/null 2>&1')
                    # fz = os.path.getsize('./temp/6.bpg')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgdec -o ./temp/16.png ./temp/6.bpg' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/16.png')
                    temp = ToTensor()(temp)
                    temp = temp[0, :, :]
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()




        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        x = upsampling(x)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2



class Generator_STD_BPG_Local(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_BPG_Local, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.form = 1
        self.upscale = 32/(2 ** scale_factor)
        # for sampling
        self.sampling = nn.Conv2d(1, 1, blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.UpsamplingBilinear2d(size=(64, 64), scale_factor=None)
        # nn.UpsamplingBilinear2d(())
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x, sides):
        x = self.sampling(x)
        w_temp = x.shape[2]
        h_temp = x.shape[3]
        w_temp = w_temp/sides*self.upscale
        h_temp = h_temp/sides*self.upscale
        # upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        # x = upsampling(x)
        # img = x[1, 1, :, :]

        if self.form == 1:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    #   print temp.size
                    temp.save('./temp/1.bmp', 'bmp', quality=100)
                    os.system('../openjpeg/build/bin/opj_compress2 -i ./temp/1.bmp -o ./temp/1.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                    os.system('../openjpeg/build/bin/opj_decompress2 -i ./temp/1.j2k -o ./temp/11.bmp' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/11.bmp')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()

                    x.data[i, j, :, :] = temp.cuda()


        if self.form == 0:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('./temp/1.png', 'PNG', quality=100)
                    # fz = os.path.getsize('./temp/7.png')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o ./temp/1.bpg ./temp/1.png' + ' > /dev/null 2>&1')
                    # fz = os.path.getsize('./temp/1.bpg')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgdec -o ./temp/11.png ./temp/1.bpg' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/11.png')
                    temp = ToTensor()(temp)
                    temp = temp[0, :, :]
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()





        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        x = upsampling(x)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2




class Generator_STD_HEVC_temp_temp_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_HEVC_temp_temp_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.form = 0
        self.upscale = 32/(2 ** scale_factor)
        # for sampling
        self.sampling = nn.Conv2d(1, 1, blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.UpsamplingBilinear2d(size=(64, 64), scale_factor=None)
        # nn.UpsamplingBilinear2d(())
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x, sides):
        x = self.sampling(x)
        w_temp = x.shape[2]
        h_temp = x.shape[3]
        w_temp = w_temp/sides*self.upscale
        h_temp = h_temp/sides*self.upscale
        # upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        # x = upsampling(x)
        # img = x[1, 1, :, :]

        if self.form == 1:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    #   print temp.size
                    temp.save('./temp/7.bmp', 'bmp', quality=100)
                    os.system('../openjpeg/build/bin/opj_compress2 -i ./temp/7.bmp -o ./temp/7.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                    os.system('../openjpeg/build/bin/opj_decompress2 -i ./temp/7.j2k -o ./temp/17.bmp' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/17.bmp')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()

                    x.data[i, j, :, :] = temp.cuda()


        if self.form == 0:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('./temp/7.png', 'PNG', quality=100)
                    # fz = os.path.getsize('./temp/7.png')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o ./temp/7.bpg ./temp/7.png' + ' > /dev/null 2>&1')
                    # fz = os.path.getsize('./temp/7.bpg')
                    # print fz
                    os.system('../libbpg-0.9.8/bpgdec -o ./temp/17.png ./temp/7.bpg' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/17.png')
                    temp = ToTensor()(temp)
                    temp = temp[0, :, :]
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()




        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        x = upsampling(x)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2

class Generator_STD_HEVC_DRRN(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_HEVC_DRRN, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.form = 2
        self.upscale = 32/(2 ** scale_factor)
        # for sampling
        self.sampling = nn.Conv2d(1, 1, blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.UpsamplingBilinear2d(size=(64, 64), scale_factor=None)
        # nn.UpsamplingBilinear2d(())
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
    
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
	self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
	self.output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
	self.relu = nn.ReLU(inplace=True)

		# weights initialization
	for m in self.modules():
		if isinstance(m, nn.Conv2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, sqrt(2. / n))



    def forward(self, x, sides):
        x = self.sampling(x)
        w_temp = x.shape[2]
        h_temp = x.shape[3]
        w_temp = w_temp/sides*self.upscale
        h_temp = h_temp/sides*self.upscale
        # upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        # x = upsampling(x)
        # img = x[1, 1, :, :]
        if self.form == 1:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    #   print temp.size
                    temp.save('./temp/7.png', 'PNG', quality=100)
                    os.system('../openjpeg/build/bin/opj_compress -i ./temp/7.png -o ./temp/7.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                    os.system('../openjpeg/build/bin/opj_decompress -i ./temp/7.j2k -o ./temp/7.png' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/7.png')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()

                    x.data[i, j, :, :] = temp.cuda()

        if self.form == 0:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('./temp/7.png', 'PNG', quality=100)
                    os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) +' -o ./temp/7.bpg ./temp/7.png' + ' > /dev/null 2>&1')
                    os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o ./temp/7.png ./temp/7.bpg' + ' > /dev/null 2>&1')
                    temp = Image.open('./temp/7.png')
                    temp = ToTensor()(temp)
                    temp = temp[0, :, :]
                    temp = temp.squeeze()
                    x.data[i, j, :, :] = temp.cuda()



        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        upsampling = nn.UpsamplingBilinear2d(size=(w_temp, h_temp), scale_factor=None)
        x = upsampling(x)

        residual = x
	inputs = self.input(self.relu(x))
	out = inputs
	for _ in range(25):
		out = self.conv2(self.relu(self.conv1(self.relu(out))))
		out = torch.add(out, inputs)

	out = self.output(self.relu(out))
	out = torch.add(out, residual) 
      
        return out



class Matric_Softmax(nn.Module):
    # no compression framework is integrated into
    def __init__(self, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))

        super(Matric_Softmax, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.subrate = subrate
        channels = 64

        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block5 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.SpatialSoftmax = nn.Softmax2d()

    def forward(self, x):

        # meas = int(np.round(self.blocksize * self.blocksize * self.subrate))
        # input_matric = torch.ones(meas, self.blocksize * self.blocksize, 1, 1)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block5 = block5.reshape(-1, self.blocksize*self.blocksize, 1, 1)
        block6 = self.SpatialSoftmax(block5)
        res = block6.reshape(-1, 1, self.blocksize, self.blocksize)

        return res


class Generator_STD_HEVC_Softmax(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_HEVC_Softmax, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        self.subrate = subrate
        upsample = [2,2]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize/2]
        else:
            upsample = [2, 2, 2]
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, upsample[_]) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)


    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        if False:
            batch_size = x.shape[0]
            channels = x.shape[1]
            # print channels
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('results1/' + str(j) + '.png', 'PNG', quality=100)
                w = x.shape[2]
                h = x.shape[3]
                os.system('ffmpeg -i results1/%d.png -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -y result.yuv' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/HM-16.18/bin/TAppEncoderStatic -c cfg/encoder_intra_main_rext.cfg -c cfg/test_400_8bit.cfg -wdt ' + str(h) + ' -hgt ' + str(w) + ' -q ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                os.system('ffmpeg -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -i rec.yuv -y results2/%d.png' + ' > /dev/null 2>&1')
                for j in range(0, channels):
                    temp = Image.open('results2/' + str(j+1) + '.png')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()
                    # print x.shape, temp.shape
                    x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x
        # meas = int(np.round(self.blocksize * self.blocksize * self.subrate))
        # input_matric = torch.ones(meas, self.blocksize*self.blocksize, 1, 1)
        block1 = self.block1(x)
        # block1_ = self.spatialSoftmax(block1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2



class Generator_STD_JPEG_Test(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        upsample = [2, 2]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize / 2]
        else:
            upsample = [2, 2, 2]
        super(Generator_STD_JPEG_Test, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, upsample[_]) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        file_size = 0
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp_j2k.png', 'PNG', quality=100)
                os.system('../openjpeg-master/build/bin/opj_compress -i ./temp_j2k.png -o ./temp_j2k.j2k -r ' + str(self.bitdepth) + ' > /dev/null 2>&1')
                os.system('../openjpeg-master/build/bin/opj_decompress -i ./temp_j2k.j2k -o ./temp_j2k.png' + ' > /dev/null 2>&1')
                file_size += os.path.getsize('temp_j2k.j2k')
                temp = Image.open('temp_j2k.png')
                temp = ToTensor()(temp)
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()

        w = x.shape[2]
        h = x.shape[3]
        bpp = file_size * 8.0 / w / h / self.blocksize / self.blocksize
        print bpp


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2, bpp

class Generator_STD_BPG_Test(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        upsample = [2, 2]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize / 2]
        else:
            upsample = [2, 2, 2]
        super(Generator_STD_BPG_Test, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, upsample[_]) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        file_size = 0
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp_bpg.png', 'PNG', quality=100)
                # os.system('../openjpeg-master/build/bin/opj_compress -i ./temp.png -o ./temp.j2k -r 1' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgenc -q '+ str(self.bitdepth) + ' -o temp_bpg.bpg temp_bpg.png')
                # os.system('../openjpeg-master/build/bin/opj_decompress -i ./temp.j2k -o ./temp.png' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/libbpg-0.9.8/bpgdec -o temp_bpg.png temp_bpg.bpg')
                file_size += os.path.getsize('temp_bpg.bpg')
                temp = Image.open('temp_bpg.png')
                temp = ToTensor()(temp)
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()[0,:,:]
        w = x.shape[2]
        h = x.shape[3]
        bpp = file_size * 8.0 / w / h / self.blocksize / self.blocksize
        print bpp


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2, bpp


class Generator_STD_HEVC_Test(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        upsample = [2, 2]
        if scale_factor == 1:
            upsample = [blocksize]
        elif scale_factor == 2:
            upsample = [2, blocksize/2]
        else:
            upsample = [2, 2, 2]
        super(Generator_STD_HEVC_Test, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, upsample[_]) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        # print channels
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                # temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('test_results1/' + str(j) + '.png', 'PNG', quality=100)
            w = x.shape[2]
            h = x.shape[3]
            os.system('ffmpeg -i test_results1/%d.png -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -y result_test.yuv' + ' > /dev/null 2>&1')
            os.system('/home/zyd/cuiwenxue/HM-16.18/bin/TAppEncoderStatic -c cfg/encoder_lowdelay_main_rext_test.cfg -c cfg/test_400_8bit_test.cfg -wdt ' + str(h) + ' -hgt ' + str(w) + ' -q ' + str(self.bitdepth) + ' > /dev/null 2>&1')
            os.system('ffmpeg -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -i rec_test.yuv -y test_results2/%d.png' + ' > /dev/null 2>&1')
            file_size = os.path.getsize('str_test.bin')
            bpp = file_size*8.0/w/h/self.blocksize/self.blocksize
            print bpp
            for j in range(0, channels):
                temp = Image.open('test_results2/' + str(j+1) + '.png')
                temp = ToTensor()(temp)
                temp = temp.squeeze()
                # print x.shape, temp.shape
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2, bpp



class Generator_STD_HEVC_Mapping(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_HEVC_Mapping, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)

        self.mapping_block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.mapping_block2 = ResidualBlock(64, has_BN=True)
        self.mapping_block3 = ResidualBlock(64, has_BN=True)
        self.mapping_block4 = ResidualBlock(64, has_BN=True)
        self.mapping_block5 = nn.Sequential(
            nn.Conv2d(64, int(np.round(blocksize*blocksize*subrate)), kernel_size=7, padding=3),
        )


        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 3) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        x_compress = Variable(torch.zeros(x.shape).cuda())
        x_temp = self.mapping_block1(x)
        x_temp = self.mapping_block2(x_temp)
        x_temp = self.mapping_block3(x_temp)
        x_temp = self.mapping_block4(x_temp)
        x_temp = self.mapping_block5(x_temp)
        x_temp = x_temp + x
        # img = x[1, 1, :, :]
        if True:
            batch_size = x.shape[0]
            channels = x.shape[1]
            for i in range(0, batch_size):
                for j in range(0, channels):
                    temp = x[i, j, :, :]
                    # temp_ = temp
                    temp = temp.unsqueeze(0)
                    temp = ToPILImage()(temp.data.cpu())
                    temp.save('hevc_mapping1/' + str(j) + '.png', 'PNG', quality=100)
                w = x.shape[2]
                h = x.shape[3]
                os.system('ffmpeg -i hevc_mapping1/%d.png -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -y result_hevc_mapping.yuv' + ' > /dev/null 2>&1')
                os.system('/home/zyd/cuiwenxue/HM-16.18/bin/TAppEncoderStatic -c cfg/encoder_lowdelay_main_rext_hevc_mapping.cfg -c cfg/test_400_8bit_hevc_mapping.cfg -wdt ' + str(h) + ' -hgt ' + str(w) + ' > /dev/null 2>&1')
                os.system('ffmpeg -s ' + str(h) + 'x' + str(w) + ' -pix_fmt gray8 -i rec_hevc_mapping.yuv -y hevc_mapping2/%d.png' + ' > /dev/null 2>&1')
                for j in range(0, channels):
                    temp = Image.open('hevc_mapping2/' + str(j+1) + '.png')
                    temp = ToTensor()(temp)
                    temp = temp.squeeze()
                    x_compress.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x_temp)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return x_compress, x_temp, (F.tanh(block8) + 1) / 2


class Generator_STD_JPEG_temp(nn.Module):
    # no compression framework is integrated into
    def __init__(self, scale_factor, blocksize, subrate, bitdepth):
        # upsample_block_num = int(math.log(scale_factor, 2))
        upsample_block_num = scale_factor

        super(Generator_STD_JPEG_temp, self).__init__()
        self.bitdepth = bitdepth
        self.blocksize = blocksize
        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 3) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.sampling(x)
        # img = x[1, 1, :, :]
        batch_size = x.shape[0]
        channels = x.shape[1]
        for i in range(0, batch_size):
            for j in range(0, channels):
                temp = x[i, j, :, :]
                # temp_ = temp
                temp = temp.unsqueeze(0)
                temp = ToPILImage()(temp.data.cpu())
                temp.save('temp2.png', 'PNG', quality=100)
                os.system('../openjpeg-master/build/bin/opj_compress -i ./temp2.png -o ./temp2.j2k -r 1' + ' > /dev/null 2>&1')
                os.system('../openjpeg-master/build/bin/opj_decompress -i ./temp2.j2k -o ./temp2.png' + ' > /dev/null 2>&1')
                temp = Image.open('temp2.png')
                temp = ToTensor()(temp)
                temp = temp.squeeze()
                x.data[i, j, :, :] = temp.cuda()


        # x = swish(self.upsampling(x))
        # x = My_LSMM_SQ_2(x, self.bitdepth)

        # x = self.upsampling(x)
        # x = My_Reshape_Adap(x, self.blocksize)
        # x = My_Reshape(x)
        # x = My_Reshape_Adap(x, self.bitdepth)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels, has_BN = False):
        super(ResidualBlock, self).__init__()
        self.has_BN = has_BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # if has_BN:
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # if has_BN:
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        if self.has_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if self.has_BN:
            residual = self.bn2(residual)

        return x + residual


class ResidualBlock2(nn.Module):
    def __init__(self, channels, has_BN = False):
        super(ResidualBlock2, self).__init__()
        self.has_BN = has_BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        # if has_BN:
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        # if has_BN:
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        if self.has_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if self.has_BN:
            residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


# this generator is for the single 3 channels.
class Generator_single(nn.Module):
    def __init__(self, scale_factor, subrate):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()

        # for sampling
        self.sampling1 = nn.Conv2d(1, int(np.round(32 * 32 * subrate)), 32, stride=32, padding=0)
        self.sampling2 = nn.Conv2d(1, int(np.round(32 * 32 * subrate)), 32, stride=32, padding=0)
        self.sampling3 = nn.Conv2d(1, int(np.round(32 * 32 * subrate)), 32, stride=32, padding=0)
        self.upsampling1 = nn.Conv2d(int(np.round(32 * 32 * subrate)), 1024, 1, stride=1, padding=0)
        self.upsampling2 = nn.Conv2d(int(np.round(32 * 32 * subrate)), 1024, 1, stride=1, padding=0)
        self.upsampling3 = nn.Conv2d(int(np.round(32 * 32 * subrate)), 1024, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

        # for test extra layers
        self.con_first = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.con = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.con_end = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):

        # get 3 channels
        x1 = x[:, 1, :, :]
        x2 = x[:, 2, :, :]
        x3 = x[:, 3, :, :]

        # for the 1th channel
        x1 = self.sampling1(x1)
        x1 = swish(self.upsampling1(x1))
        x1 = My_Reshape(x1)

        # for the 2th channel
        x2 = self.sampling1(x2)
        x2 = swish(self.upsampling1(x2))
        x2 = My_Reshape(x2)

        # for the 3th channel
        x3 = self.sampling1(x3)
        x3 = swish(self.upsampling1(x3))
        x3 = My_Reshape(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Generator_Original(nn.Module):
    def __init__(self):

        super(Generator_Original, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block8 = nn.Conv2d(64, 1, kernel_size=9, padding=4)

    def forward(self, x):

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Norm(nn.Module):
    def __init__(self, subrate):
        #upsample_block_num = int(math.log(scale_factor, 2))

        super(Norm, self).__init__()

        # for sampling
        self.blocksize = 32/scale_factor
        self.sampling = nn.Conv2d(3, int(np.round(32*32*subrate*3)), 32, stride=32, padding=0)
        self.upsampling = nn.Conv2d(int(np.round(32*32*subrate*3)), self.blocksize*self.blocksize*3, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block8 = (nn.Conv2d(64, 3, kernel_size=9, padding=4))

        # for test extra layers
        self.con_first = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.con = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.con_end = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sampling(x)
        x = swish(self.upsampling(x))
        x = My_Reshape_Adap(x, self.blocksize)

        # x = self.con_first(x)
        # x = self.con(x)
        # x = self.con_end(x)
        # return x

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2
