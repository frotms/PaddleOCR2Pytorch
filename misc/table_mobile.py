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
NAME = 'table_mobile'
DISABLE_SE = True
HIDDEN_SIZE = 256
LOC_TYPE = 2
MAX_ELEM_LENGTH = 800
IN_MAX_LEN = 488

tmp_save_name = '{}.npy'.format(NAME)

def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))
def compare_ret(pp_ret, pt_ret, info):
    print('============ {} ============='.format(info))
    print('shape:', pp_ret.shape)
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
import pp_table_att_head
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

        self.head = pp_table_att_head.TableAttentionHead(
            in_channels=head_in_channels,
            hidden_size=HIDDEN_SIZE,
            loc_type=LOC_TYPE,
            in_max_len=IN_MAX_LEN,
            max_elem_length=MAX_ELEM_LENGTH,
        )

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = self.head(x)
        return x
def paddle_func():
    np.random.seed(SEED)
    x = np.random.rand(*INPUT_SIZE).astype(np.float32)
    # del x; x = np.load('inp.npy')

    sd = get_np_static_dict('table_org.npy')
    with fluid.dygraph.guard():
        layer = PPNet()
        layer.set_state_dict(sd)
        layer.eval()



        inp = fluid.dygraph.to_variable(x)
        ret = layer(inp)

        sd = get_pp_static_dict(layer.state_dict())
        np.save(tmp_save_name, sd, allow_pickle=True)
    # {'structure_probs': structure_probs, 'loc_preds': loc_preds}

    return [e_ret.numpy() for kk, e_ret in ret.items()]
def paddle_func_():
    np.random.seed(SEED)
    x = np.random.rand(*INPUT_SIZE).astype(np.float32)
    # del x; x = np.load('inp.npy')
    with fluid.dygraph.guard():
        layer = PPNet()

        layer.eval()

        inp = fluid.dygraph.to_variable(x)
        ret = layer(inp)

        sd = get_pp_static_dict(layer.state_dict())
        np.save(tmp_save_name, sd, allow_pickle=True)
    # {'structure_probs': structure_probs, 'loc_preds': loc_preds}

    return [e_ret.numpy() for kk, e_ret in ret.items()]


import pt_table_mobilenet_v3
import pt_table_att_head

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

        self.head = pt_table_att_head.TableAttentionHead(
            in_channels=head_in_channels,
            hidden_size=HIDDEN_SIZE,
            loc_type=LOC_TYPE,
            in_max_len=IN_MAX_LEN,
            max_elem_length=MAX_ELEM_LENGTH,
        )

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = self.head(x)
        return x


def torch_func():
    np.random.seed(SEED)
    x = np.random.rand(*INPUT_SIZE).astype(np.float32)
    # del x;x = np.load('inp.npy')
    layer = PTNet()


    sd = get_np_static_dict(tmp_save_name)

    for key, value in layer.state_dict().items():
        print('pytorch: {} ---- {}'.format(key, value.shape))

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

    return [e_ret.cpu().numpy() for kk, e_ret in ret.items()]


if __name__ == '__main__':
    clean(tmp_save_name)
    pp = paddle_func()
    print('==========++++=================')
    pt = torch_func()
    [compare_ret(e_pp, e_pt, NAME) for e_pp, e_pt in zip(pp, pt)]
    clean(tmp_save_name)
    print('done.')
