import os, sys
import numpy as np
import paddle
# paddle.enable_static()
import paddle.fluid as fluid
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from collections import OrderedDict
import torch

SEED = 666
input_shape = (1, 256, 512)
d_model = 512

def paddle_func():
    np.random.seed(SEED)
    # x = np.load('input.npy', allow_pickle=True)
    x = np.random.rand(input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
    np.save('input.npy', x)
    print('pp input size: ', x.shape, np.sum(x), np.mean(x), np.max(x), np.min(x))

    # np.save('org.npy', x)
    with fluid.dygraph.guard():
        layer = paddle.nn.LayerNorm(normalized_shape=d_model,
                                    epsilon=1e-05,
                                    weight_attr=fluid.ParamAttr(
                                        initializer=fluid.initializer.Constant(1.)),
                                    bias_attr=fluid.ParamAttr(
                                        initializer=fluid.initializer.Constant(0.)),
                                    name=None)

        # para_state_dict, opti_state_dict = fluid.load_dygraph('rec_r50_vd_srn_train/best_accuracy')
        # layer.set_state_dict(para_state_dict)

        state_dict = layer.state_dict()
        sd = OrderedDict()
        for key, value in state_dict.items():
            v = value.numpy()
            print(key, value.shape, np.sum(v), np.mean(v), np.max(v), np.min(v))
            sd[key] = v

        np.save('layernorm.npy', sd)

        inputs = fluid.dygraph.to_variable(x)
        out = layer(inputs)
        # print(len(outputs))
    return out.numpy()
    # return alpha.numpy()


def torch_func():
    np.random.seed(SEED)
    x = np.load('input.npy', allow_pickle=True)
    # x = np.random.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
    print('pt input size: ', x.shape, np.sum(x), np.mean(x), np.max(x), np.min(x))
    inputs = torch.Tensor(x)
    layer = torch.nn.LayerNorm(normalized_shape=d_model,
                               elementwise_affine=True,
                               eps=1e-5)
    sd = np.load('layernorm.npy', allow_pickle=True)
    sd = sd.tolist()

    for key, value in layer.state_dict().items():
        name = key
        keyword = 'block_list.'
        if keyword in name:
            # replace: block_list.
            name = name.replace(keyword, '')
        else:
            name = name

        # for srn
        keyword = 'base_block.'
        if keyword in name:
            # replace: base_block.
            name = name.replace(keyword, '')
        keyword = 'base_block_2.0.'
        if keyword in name:
            # replace: base_block_2.0. -> base_block_2.
            name = name.replace(keyword, 'base_block_2.')


        if name.endswith('num_batches_tracked'):
            continue

        if name.endswith('running_mean'):
            ppname = name.replace('running_mean', '_mean')
        elif name.endswith('running_var'):
            ppname = name.replace('running_var', '_variance')
        elif name.endswith('bias') or name.endswith('weight'):
            ppname = name
        elif 'lstm' in name:
            ppname = name
        elif 'attention_cell' in name:
            ppname = name

        else:
            print('Redundance:')
            print(name)
            raise ValueError



        try:
            if key.endswith('.weight'):
                if len(sd[ppname].shape) == len(layer.state_dict()[key].shape) == 2 \
                    and sd[ppname].shape[0] == layer.state_dict()[key].shape[1] \
                    and sd[ppname].shape[1] == layer.state_dict()[key].shape[0]:

                    layer.state_dict()[key].copy_(torch.Tensor(sd[ppname].T))
                else:
                    layer.state_dict()[key].copy_(torch.Tensor(sd[ppname]))
            else:
                layer.state_dict()[key].copy_(torch.Tensor(sd[ppname]))
        except Exception as e:
            print('except: pt: ', key)
            print('except: pp: ', ppname)
            print('except: pt: ', layer.state_dict()[key].shape)
            print('except: pp: ', sd[ppname].shape)
            raise e

        v = layer.state_dict()[key].numpy()
        print(ppname, v.shape, np.sum(v), np.mean(v), np.max(v), np.min(v))


    out = layer(inputs)
    return out.data.numpy()
    # return alpha.data.numpy()


if __name__ == '__main__':
    print('==========paddle=================')
    a = paddle_func()
    print(a.shape)
    print('a: ', np.sum(a), np.mean(a), np.max(a), np.min(a))
    print('===========pytorch================')
    b = torch_func()
    print(b.shape)
    print('b: ', np.sum(b), np.mean(b), np.max(b), np.min(b))