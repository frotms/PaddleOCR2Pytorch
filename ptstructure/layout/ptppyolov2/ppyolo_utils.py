import math
import six
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# from ..core.workspace import register, serializable

def mish(x):
    return x * torch.tanh(F.softplus(x))


def batch_norm(ch,
               norm_type='bn',
               norm_decay=0.,
               freeze_norm=False,
               initializer=None,
               data_format='NCHW'):
    if norm_type == 'sync_bn':
        batch_norm = nn.SyncBatchNorm
    else:
        batch_norm = nn.BatchNorm2d

    norm_lr = 0. if freeze_norm else 1.
    # weight_attr = ParamAttr(
    #     initializer=initializer,
    #     learning_rate=norm_lr,
    #     regularizer=L2Decay(norm_decay),
    #     trainable=False if freeze_norm else True)
    # bias_attr = ParamAttr(
    #     learning_rate=norm_lr,
    #     regularizer=L2Decay(norm_decay),
    #     trainable=False if freeze_norm else True)

    norm_layer = batch_norm(
        num_features=ch,
        affine=True,
        track_running_stats=True,
        )

    # norm_params = norm_layer.parameters()
    # if freeze_norm:
    #     for param in norm_params:
    #         param.stop_gradient = True

    return norm_layer


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]

def torch_initialize_weights(modules):
    # weight initialization
    for m in modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

class DeformableConvV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 ):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size ** 2
        self.mask_channel = kernel_size ** 2

        self.conv_offset = torch.nn.Conv2d(
            in_channels,
            3 * kernel_size ** 2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True)

        self.conv_dcn = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        torch_initialize_weights(self.modules)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = torch.split(
            offset_mask,
            split_size_or_sections=[self.offset_channel, self.mask_channel],
            dim=1,
        )
        mask = torch.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

# out = max(0, min(1, slop*x+offset))
# paddle.fluid.layers.hard_sigmoid(x, slope=0.2, offset=0.5, name=None)
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # torch: F.relu6(x + 3., inplace=self.inplace) / 6.
        # paddle: F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.
        return F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.


class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()

        if act_type is None:
            self.act = nn.Identity()
        elif act_type.lower() == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type.lower() == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type.lower() == 'sigmoid':
            raise NotImplementedError
        elif act_type.lower() == 'hard_sigmoid':
            self.act = Hsigmoid(inplace)
        elif act_type.lower() == 'hard_swish':
            self.act = Hswish(inplace=inplace)
        elif act_type.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        else:
            self.act = nn.Identity()

    def forward(self, inputs):
        return self.act(inputs)


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob, name, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890
        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            # matrix = paddle.cast(paddle.rand(x.shape, x.dtype) < gamma, x.dtype)
            matrix = (torch.rand(x.size(), dtype=x.dtype) < gamma).to(x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
                data_format=self.data_format)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y

class NameAdapter(object):
    """Fix the backbones variable names for pretrained weight"""

    def __init__(self, model):
        super(NameAdapter, self).__init__()
        self.model = model

    @property
    def model_type(self):
        return getattr(self.model, '_model_type', '')

    @property
    def variant(self):
        return getattr(self.model, 'variant', '')

    def fix_conv_norm_name(self, name):
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        # the naming rule is same as pretrained weight
        if self.model_type == 'SEResNeXt':
            bn_name = name + "_bn"
        return bn_name

    def fix_shortcut_name(self, name):
        if self.model_type == 'SEResNeXt':
            name = 'conv' + name + '_prj'
        return name

    def fix_bottleneck_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            conv_name3 = 'conv' + name + '_x3'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            conv_name3 = name + "_branch2c"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, conv_name3, shortcut_name

    def fix_basicblock_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, shortcut_name

    def fix_layer_warp_name(self, stage_num, count, i):
        name = 'res' + str(stage_num)
        if count > 10 and stage_num == 4:
            if i == 0:
                conv_name = name + "a"
            else:
                conv_name = name + "b" + str(i)
        else:
            conv_name = name + chr(ord("a") + i)
        if self.model_type == 'SEResNeXt':
            conv_name = str(stage_num + 2) + '_' + str(i + 1)
        return conv_name

    def fix_c1_stage_name(self):
        return "res_conv1" if self.model_type == 'ResNeXt' else "conv1"
