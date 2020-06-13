import torch
import numpy as np
import torch.nn as nn
import functools

# class PachGAN(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(PachGAN, self).__init__()
#         kw = 4
#         padw = 1
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         nf_mult = 1
#         nf_mult_prev = 1
#         self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
#         self.LR = nn.LeakyReLU(0.2, True)
#         self.conv2 = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
#         self.seq1 = nn.Sequential(*[norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)])
#         # for n in range(1, n_layers):  # gradually increase the number of filters
#         #     nf_mult_prev = nf_mult
#         #     nf_mult = min(2 ** n, 8)
#         #     sequence += [
#         #         nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#         #         norm_layer(ndf * nf_mult),
#         #         nn.LeakyReLU(0.2, True)
#         #     ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         self.conv3 = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)
#         self.seq2 = nn.Sequential(*[norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)])
#         # sequence += [
#         #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#         #     norm_layer(ndf * nf_mult),
#         #     nn.LeakyReLU(0.2, True)
#         # ]
#
#
#         self.conv4 = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
#
#         # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         # self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         """Standard forward."""
#         input = self.conv1(input)
#         input = self.LR(input)
#         input = self.conv2(input)
#         input = self.seq1(input)
#         input = self.conv3(input)
#         input = self.seq2(input)
#         input = self.conv4(input)
#
#         return input

        # return self.model(input)


# test_module = PachGAN(3, n_layers=2)
# ipt = torch.rand([1, 3, 2048, 2048])
# out = test_module(ipt)


# [filter size, stride, padding]
# Assume the two dimensions are the same
# Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
# Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

import math

def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (
    layer[0], layer[1], layer[2], layer[3]))


def weights_matrix(patch, img, n_f_h, n_f_w, jump, rf, start):
    # B, C, H, W = patch.shape
    wm = np.zeros(img.shape)
    for i in range(n_f_h):
        for j in range(n_f_w):
            val = patch[:, :, i, j]
            hf, ht = int(max(0, start + i*jump - rf//2)), int(start + i*jump + rf - rf//2)
            wf, wt = int(max(0, start + j*jump - rf//2)), int(start + j*jump + rf - rf//2)
            wm[:, :, hf:ht, wf:wt] += val

            # wm[:,:, max(0, start + i*jump - rf):start + i*jump + rf, max(0, start + j*jump - rf):start + j*jump + rf] += \
            # patch[:,:, max(0, start + i*jump - rf):start + i*jump + rf, max(0, start + j*jump - rf):start + j*jump + rf]
    return wm


def receptive_cal(imsize):
    convnet = [[4, 2, 1], [4, 2, 1], [4, 1, 1], [4, 1, 1]]
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
    currentLayer = [imsize, 1, 1, 0.5]
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        # printLayer(currentLayer, layer_names[i])
    return currentLayer

def getWeights(patch, img, currentLayer_h, currentLayer_w):
    n_f_h, jump, rf, start = currentLayer_h[0], currentLayer_h[1], currentLayer_h[2], currentLayer_h[3]
    n_f_w, jump, rf, start = currentLayer_w[0], currentLayer_w[1], currentLayer_w[2], currentLayer_w[3]
    s = weights_matrix(patch, img, n_f_h, n_f_w, jump, rf, start)
    count = weights_matrix(np.ones_like(patch), img, n_f_h, n_f_w, jump, rf, start)
    return s / count


layerInfos = []
if __name__ == '__main__':
    patch = np.ones([1,1,19,12])*2
    img = np.ones([1,1,86,56])*2
    currentLayer_h, currentLayer_w = receptive_cal(img.shape[2]), receptive_cal(img.shape[3])
    res1 = getWeights(patch, img, currentLayer_h, currentLayer_w)
    print(res1.shape)



    # imsize = 344
    # currentLayer = [imsize, 1, 1, 0.5]
    # # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    # print("-------Net summary------")
    # # currentLayer = [imsize, 1, 1, 0.5]
    # convnet = [[4, 2, 1], [4, 2, 1], [4, 1, 1], [4, 1, 1]]
    # layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
    # printLayer(currentLayer, "input image")
    # for i in range(len(convnet)):
    #     currentLayer = outFromIn(convnet[i], currentLayer)
    #     layerInfos.append(currentLayer)
    #     printLayer(currentLayer, layer_names[i])
    # print("------------------------")
    # n_f, jump, rf, start = currentLayer[0], currentLayer[1], currentLayer[2], currentLayer[3]
    # ipt = np.ones([1,1,10,10])
    # img = np.ones([1,1,48,48])
    # weights_matrix(ipt, img, n_f, jump, rf, start)



    # print(start, receptive)
    # layer_name = input("Layer name where the feature in: ")
    # layer_idx = layer_names.index(layer_name)
    # idx_x = int(input("index of the feature in x dimension (from 0)"))
    # idx_y = int(input("index of the feature in y dimension (from 0)"))
    #
    # n = layerInfos[layer_idx][0]
    # j = layerInfos[layer_idx][1]
    # r = layerInfos[layer_idx][2]
    # start = layerInfos[layer_idx][3]
    # assert (idx_x < n)
    # assert (idx_y < n)
    #
    # print("receptive field: (%s, %s)" % (r, r))
    # print("center: (%s, %s)" % (start + idx_x * j, start + idx_y * j))