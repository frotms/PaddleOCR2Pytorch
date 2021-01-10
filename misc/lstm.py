
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
INPUT_SIZE = 89
KERNEL_SIZE = 1
STRIDES = (1,1)
PADDING = 0

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

def paddle_lstm():
    np.random.seed(SEED)
    x = np.random.rand(1, 80, 512).astype(np.float32)
    # np.save('org.npy', x)
    with fluid.dygraph.guard():
        lstm = nn.LSTM(512, 256, num_layers=2, direction='bidirectional')

        # sd = np.load('lstm.npy', allow_pickle=True).tolist()
        # lstm.set_state_dict(sd)

        state_dict = lstm.state_dict()
        sd = OrderedDict()
        for key, value in state_dict.items():
            v = value.numpy()
            print(key, value.shape, np.sum(v), np.mean(v), np.max(v), np.min(v))
            sd[key] = v

        np.save('lstm.npy', sd)

        inp = fluid.dygraph.to_variable(x)
        ret, _ = lstm(inp)
        print(len(ret))
    return ret.numpy()


def torch_lstm():
    np.random.seed(SEED)
    org = torch.Tensor(np.random.rand(1, 80, 512).astype(np.float32))
    lstm = torch.nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)
    sd = np.load('lstm.npy', allow_pickle=True)
    sd = sd.tolist()

    for key, value in lstm.state_dict().items():
        lstm.state_dict()[key].copy_(torch.Tensor(sd[key]))


    ret, _ = lstm(org)
    return ret.data.numpy()


def torch_lstm_m():
    class EncoderWithRNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_size):
            super(EncoderWithRNN, self).__init__()
            self.out_channels = hidden_size * 2
            self.rnn1 = torch.nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)
            self.rnn2 = torch.nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)

        def forward(self, x):
            self.rnn1.flatten_parameters()
            self.rnn2.flatten_parameters()
            out1, h1 = self.rnn1(x)
            out2, h2 = self.rnn2(torch.flip(x, [1]))
            return torch.cat([out1, torch.flip(out2, [1])], 2)

    lstm = EncoderWithRNN(10, 3)
    for key, value in lstm.state_dict().items():
        print(key, value.shape)



if __name__ == '__main__':
    print('==========paddle=================')
    a = paddle_lstm()
    print(a.shape)
    print('a: ', np.sum(a), np.mean(a), np.max(a), np.min(a))
    print('===========pytorch================')
    b = torch_lstm()
    print(b.shape)
    print('b: ', np.sum(b), np.mean(b), np.max(b), np.min(b))
    # print('===========pytorch_m================')
    # torch_lstm_m()