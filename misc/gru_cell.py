
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

input_size_x = (1, 230)
input_size_y = (1, 96)

def get_para_bias_attr(l2_decay, k, name):
    import math
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_w_attr")
    bias_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_b_attr")
    return [weight_attr, bias_attr]

def paddle_grucell():
    np.random.seed(SEED)
    x = np.random.rand(1, 230).astype(np.float32)
    y = np.random.rand(1, 96).astype(np.float32)
    # np.save('org.npy', x)
    with fluid.dygraph.guard():
        layer = nn.GRUCell(
            input_size=230, hidden_size=96)

        # sd = np.load('lstm.npy', allow_pickle=True).tolist()
        # lstm.set_state_dict(sd)

        state_dict = layer.state_dict()
        sd = OrderedDict()
        for key, value in state_dict.items():
            v = value.numpy()
            print(key, value.shape, np.sum(v), np.mean(v), np.max(v), np.min(v))
            sd[key] = v

        np.save('gru_cell.npy', sd)

        inp = fluid.dygraph.to_variable(x)
        prev_hidden = fluid.dygraph.to_variable(y)
        ret, hidden = layer(inp, prev_hidden)
        print(ret-hidden)
    return ret.numpy()


def torch_grucell():
    np.random.seed(SEED)
    x = np.random.rand(1, 230).astype(np.float32)
    y = np.random.rand(1, 96).astype(np.float32)
    inp = torch.Tensor(x)
    prev_hidden = torch.Tensor(y)

    layer = torch.nn.GRUCell(
        input_size=230, hidden_size=96, bias=True)

    sd = np.load('gru_cell.npy', allow_pickle=True)
    sd = sd.tolist()

    for key, value in layer.state_dict().items():
        print(key, value.shape)
        layer.state_dict()[key].copy_(torch.Tensor(sd[key]))


    ret = layer(inp, prev_hidden)
    return ret.data.numpy()



if __name__ == '__main__':
    print('==========paddle=================')
    a = paddle_grucell()
    print(a.shape)
    print('a: ', np.sum(a), np.mean(a), np.max(a), np.min(a))
    print('===========pytorch================')
    b = torch_grucell()
    print(b.shape)
    print('b: ', np.sum(b), np.mean(b), np.max(b), np.min(b))
    # print('===========pytorch_m================')
    # torch_lstm_m()