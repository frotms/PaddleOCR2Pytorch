
import os, sys
import numpy as np
import paddle
# paddle.enable_static()
import paddle.fluid as fluid
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

import torch

SEED = 666
# INPUT_SIZE = 89
# KERNEL_SIZE = 1
# STRIDES = (1,1)
# PADDING = 0

def paddle_conv():
    np.random.seed(SEED)

    import cv2
    image = cv2.imread('images/Snipaste.jpg')
    image = cv2.resize(image, (320, 32))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = 0.5
    std = 0.5
    scale = 1. / 255
    norm_img = (image * scale - mean) / std
    transpose_img = norm_img.transpose(2, 0, 1)
    transpose_img = np.expand_dims(transpose_img, 0)
    inp = transpose_img.astype(np.float32)
    x = inp

    with fluid.dygraph.guard():
        simple_conv = nn.Conv2D(in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=ParamAttr(name='conv' + "_weights"),
            bias_attr=False)

        inp = fluid.dygraph.to_variable(x)
        ret = simple_conv(inp)
        np.save('fc_w.npy', list(simple_conv.state_dict()['weight'].numpy()))
        # np.save('fc_b.npy', list(simple_conv.state_dict().values())[1].numpy())

        # print(ret)
    return ret.numpy()

def torch_conv():
    # np.random.seed(SEED)
    # org = torch.Tensor(np.random.rand(1,200).astype(np.float32))
    # org = torch.Tensor(np.load('org.npy'))
    import cv2
    image = cv2.imread('images/Snipaste.jpg')
    image = cv2.resize(image, (320, 32))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = 0.5
    std = 0.5
    scale = 1. / 255
    norm_img = (image * scale - mean) / std
    transpose_img = norm_img.transpose(2, 0, 1)
    transpose_img = np.expand_dims(transpose_img, 0)
    inp = transpose_img.astype(np.float32)
    org = torch.Tensor(inp)

    tfc = torch.nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1,groups=1,bias=False)

    fc_w = np.load('fc_w.npy')
    # fc_b = np.load('fc_b.npy')
    tfc.state_dict()['weight'].copy_(torch.Tensor(fc_w))
    # tfc.state_dict()['bias'].copy_(torch.Tensor(fc_b))

    tres = tfc(org)
    # print(tres)
    return tres.data.numpy()


if __name__ == '__main__':
    a = paddle_conv()
    b = torch_conv()
    print('a: ', np.sum(a), np.mean(a), np.max(a), np.min(a))
    print(b.shape)
    print('b: ', np.sum(b), np.mean(b), np.max(b), np.min(b))
    print(np.sum(np.abs(a-b)), np.mean(np.abs(a-b)))
