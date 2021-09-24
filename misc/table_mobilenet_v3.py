import os, sys
from collections import OrderedDict
import numpy as np
import paddle
# paddle.enable_static()
import paddle.fluid as fluid
import torch
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F


SEED = 666
INPUT_SIZE = (1, 3, 488, 488)
IN_CHANNELS = INPUT_SIZE[1]
OUT_CHANNELS = 16
MODEL_NAME = 'large'
SCALE = 1.0
NAME = 'table_mobilenet_v3'
DISABLE_SE = True
tmp_save_name = '{}.npy'.format(NAME)

def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))
def compare_ret(pp_ret, pt_ret, info):
    print('============ {} ============='.format(info))
    print('pp: ', np.sum(pp_ret), np.mean(pp_ret), np.max(pp_ret), np.min(pp_ret))
    print('ms: ', np.sum(pt_ret), np.mean(pt_ret), np.max(pt_ret), np.min(pt_ret))
    print('sub: ', np.sum(np.abs(pp_ret-pt_ret)), np.mean(np.abs(pp_ret-pt_ret)))

def clean(filename):
    filename = os.path.abspath(os.path.expanduser(filename))
    if os.path.exists(filename):
        os.remove(filename)
        print('remove: {}'.format(filename))

def get_pp_static_dict(input_dict):
    sd = OrderedDict()
    for key, value in input_dict.items():
        v = value.numpy()
        sd[key] = v
        print('pp: {} ---- {}'.format(key, v.shape))
    return sd
def get_np_static_dict(npy_path):
    sd = np.load(npy_path, allow_pickle=True)
    sd = sd.tolist()
    return sd

import pp_table_mobilenet_v3
class PPNet(paddle.nn.Layer):
    def __init__(self,**kwargs):
        super(PPNet, self).__init__()

        self.backbone = pp_table_mobilenet_v3.MobileNetV3(
            in_channels=IN_CHANNELS,
            model_name=MODEL_NAME,
            scale=SCALE,
            disable_se=DISABLE_SE,
        )

        head_in_channels = self.backbone.out_channels

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        return x
def paddle_func():
    np.random.seed(SEED)
    x = np.random.rand(*INPUT_SIZE).astype(np.float32)
    del x; x = np.load('inp.npy')

    sd_ = get_np_static_dict('table_org.npy')
    sd = OrderedDict()
    for k, v in sd_.items():
        if k.startswith('backbone.'):
            sd[k] = v.copy()

    with fluid.dygraph.guard():
        layer = PPNet()

        layer.eval()
        layer.set_state_dict(sd)

        inp = fluid.dygraph.to_variable(x)
        ret = layer(inp)

        sd = get_pp_static_dict(layer.state_dict())
        np.save(tmp_save_name, sd, allow_pickle=True)

    return [e_ret.numpy() for e_ret in ret]
def paddle_func_():
    np.random.seed(SEED)
    x = np.random.rand(*INPUT_SIZE).astype(np.float32)

    with fluid.dygraph.guard():
        layer = PPNet()

        layer.eval()

        inp = fluid.dygraph.to_variable(x)
        ret = layer(inp)

        sd = get_pp_static_dict(layer.state_dict())
        np.save(tmp_save_name, sd, allow_pickle=True)

    return [e_ret.numpy() for e_ret in ret]

import pt_table_mobilenet_v3

class PTNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PTNet, self).__init__()

        self.backbone = pt_table_mobilenet_v3.MobileNetV3(
            in_channels=IN_CHANNELS,
            model_name=MODEL_NAME,
            scale=SCALE,
            disable_se=DISABLE_SE,
        )

        head_in_channels = self.backbone.out_channels

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        return x


def torch_func():
    np.random.seed(SEED)
    x = np.random.rand(*INPUT_SIZE).astype(np.float32)
    del x; x = np.load('inp.npy')
    layer = PTNet()

    sd = get_np_static_dict(tmp_save_name)

    for key, value in layer.state_dict().items():
        print('pytorch: {} ---- {}'.format(key, value.shape))
    #
    for k, v in layer.state_dict().items():
        ppname = k

        if k.endswith('num_batches_tracked'):
            continue

        ppname = ppname.replace('.running_mean', '._mean')
        ppname = ppname.replace('.running_var', '._variance')

        if k.startswith('backbone.conv.'):
            pass

        if k.startswith('backbone.stages.'):
            ppname = ppname.replace('backbone.stages.', 'backbone.stage')

        if k.startswith('head.'):
            pass

        try:
            if ppname.endswith('.weight') \
                    and len(sd[ppname].shape) == len(layer.state_dict()[k].shape) == 2 \
                    and sd[ppname].shape[0] == layer.state_dict()[k].shape[1] \
                    and sd[ppname].shape[1] == layer.state_dict()[k].shape[0]:  # for general fc
                layer.state_dict()[k].copy_(torch.Tensor(sd[ppname].T))

            else:
                layer.state_dict()[k].copy_(torch.Tensor(sd[ppname]))
        except Exception as e:
            print('pytorch: {}, {}'.format(k, v.size()))
            print('paddle: {}'.format(ppname))
            print('paddle: {}'.format(sd[ppname].shape))
            raise e


    layer.eval()
    with torch.no_grad():
        inp = torch.from_numpy(x)
        ret = layer(inp)

    return [e_ret.cpu().numpy() for e_ret in ret]


if __name__ == '__main__':
    clean(tmp_save_name)
    pp = paddle_func()

    print('==========++++=================')
    pt = torch_func()
    [compare_ret(e_pp, e_pt, NAME) for e_pp, e_pt in zip(pp, pt)]
    clean(tmp_save_name)
    [print(i.shape) for i in pp]
    print('done.')
