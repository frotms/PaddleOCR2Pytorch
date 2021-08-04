
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
            print(key, value.shape)#, np.sum(v), np.mean(v), np.max(v), np.min(v))
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
        print(key, value.shape)
        lstm.state_dict()[key].copy_(torch.Tensor(sd[key]))


    ret, _ = lstm(org)
    return ret.data.numpy()


def torch_lstm_m():
    class EncoderWithRNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_size):
            super(EncoderWithRNN, self).__init__()
            self.out_channels = hidden_size * 2
            self.rnn1_fw = torch.nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=1)
            self.rnn1_bw = torch.nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=1)
            self.rnn2_fw = torch.nn.LSTM(self.out_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=1)
            self.rnn2_bw = torch.nn.LSTM(self.out_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=1)

        def bi_lstm(self, x, fw_fn, bw_fn):
            out1, h1 = fw_fn(x)
            out2, h2 = bw_fn(torch.flip(x, [1]))
            return torch.cat([out1, torch.flip(out2, [1])], 2)

        def forward(self, x):
            x = self.bi_lstm(x, self.rnn1_fw, self.rnn1_bw)
            x = self.bi_lstm(x, self.rnn2_fw, self.rnn2_bw)
            return x
            # self.rnn1.flatten_parameters()
            # self.rnn2.flatten_parameters()
            # out1, h1 = self.rnn1(x)
            # out2, h2 = self.rnn2(torch.flip(x, [1]))
            # return torch.cat([out1, torch.flip(out2, [1])], 2)

    map_list = [
                'weight_ih_l0',
                'weight_hh_l0',
                'bias_ih_l0',
                'bias_hh_l0',
                'weight_ih_l0_reverse',
                'weight_hh_l0_reverse',
                'bias_ih_l0_reverse',
                'bias_hh_l0_reverse',

                'weight_ih_l1',
                'weight_hh_l1',
                'bias_ih_l1',
                'bias_hh_l1',
                'weight_ih_l1_reverse',
                'weight_hh_l1_reverse',
                'bias_ih_l1_reverse',
                'bias_hh_l1_reverse',

                # 'weight_ih_l0',
                # 'weight_hh_l0',
                # 'bias_ih_l0',
                # 'bias_hh_l0',
                # 'weight_ih_l1',
                # 'weight_hh_l1',
                # 'bias_ih_l1',
                # 'bias_hh_l1',
                # 'weight_ih_l0_reverse',
                # 'weight_hh_l0_reverse',
                # 'bias_ih_l0_reverse',
                # 'bias_hh_l0_reverse',
                # 'weight_ih_l1_reverse',
                # 'weight_hh_l1_reverse',
                # 'bias_ih_l1_reverse',
                # 'bias_hh_l1_reverse',
                ]

    np.random.seed(SEED)
    org = torch.Tensor(np.random.rand(1, 80, 512).astype(np.float32))

    lstm = EncoderWithRNN(512, 256)
    sd = np.load('lstm.npy', allow_pickle=True)
    sd = sd.tolist()

    for idx, (key, value) in enumerate(lstm.state_dict().items()):
        print(key, value.shape)
        lstm.state_dict()[key].copy_(torch.Tensor(sd[map_list[idx]]))

    ret = lstm(org)
    return ret.data.numpy()


def torch_lstm_bi():
    class EncoderWithRNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_size):
            super(EncoderWithRNN, self).__init__()
            self.out_channels = hidden_size * 2
            # self.rnn = torch.nn.ModuleList()
            # self.rnn.append(torch.nn.LSTM(in_channels, hidden_size, bidirectional=True, batch_first=True, num_layers=1))
            # self.rnn.append(torch.nn.LSTM(in_channels, hidden_size, bidirectional=True, batch_first=True, num_layers=1))

            # self.rnn = torch.nn.ModuleList(
            #     torch.nn.LSTM(in_channels, hidden_size, bidirectional=True, batch_first=True, num_layers=1),
            #      torch.nn.LSTM(in_channels, hidden_size, bidirectional=True, batch_first=True, num_layers=1),
            #                                         )

            self.rnn = torch.nn.LSTM(in_channels, hidden_size, bidirectional=True, batch_first=True, num_layers=1)
            self.rnn2 = torch.nn.LSTM(in_channels, hidden_size, bidirectional=True, batch_first=True, num_layers=1)

        def forward(self, x):
            ret, _ = self.rnn(x)
            ret, _ = self.rnn2(ret)

            # for i in range(2):
            #     x, _ = self.rnn[i](x)
            # ret = x
            return ret

    map_list = [
                'weight_ih_l0',
                'weight_hh_l0',
                'bias_ih_l0',
                'bias_hh_l0',
                'weight_ih_l0_reverse',
                'weight_hh_l0_reverse',
                'bias_ih_l0_reverse',
                'bias_hh_l0_reverse',

                'weight_ih_l1',
                'weight_hh_l1',
                'bias_ih_l1',
                'bias_hh_l1',
                'weight_ih_l1_reverse',
                'weight_hh_l1_reverse',
                'bias_ih_l1_reverse',
                'bias_hh_l1_reverse',

                ]

    np.random.seed(SEED)
    org = torch.Tensor(np.random.rand(1, 80, 512).astype(np.float32))

    lstm = EncoderWithRNN(512, 256)
    sd = np.load('lstm.npy', allow_pickle=True)
    sd = sd.tolist()

    for idx, (key, value) in enumerate(lstm.state_dict().items()):
        print(key, value.shape)
        lstm.state_dict()[key].copy_(torch.Tensor(sd[map_list[idx]]))

    ret = lstm(org)
    return ret.data.numpy()


def torch_lstm_di():
    class EncoderWithRNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_size):
            super(EncoderWithRNN, self).__init__()
            self.out_channels = hidden_size * 2
            self.rnn1 = torch.nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)
            self.rnn2 = torch.nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)

        def forward(self, x):
            out1, h1 = self.rnn1(x)
            out2, h2 = self.rnn2(torch.flip(x, [1]))
            return torch.cat([out1, torch.flip(out2, [1])], 2)

    map_list = [
                'weight_ih_l0',
                'weight_hh_l0',
                'bias_ih_l0',
                'bias_hh_l0',
                'weight_ih_l0_reverse',
                'weight_hh_l0_reverse',
                'bias_ih_l0_reverse',
                'bias_hh_l0_reverse',

                'weight_ih_l1',
                'weight_hh_l1',
                'bias_ih_l1',
                'bias_hh_l1',
                'weight_ih_l1_reverse',
                'weight_hh_l1_reverse',
                'bias_ih_l1_reverse',
                'bias_hh_l1_reverse',

                # 'weight_ih_l0',
                # 'weight_hh_l0',
                # 'bias_ih_l0',
                # 'bias_hh_l0',
                # 'weight_ih_l1',
                # 'weight_hh_l1',
                # 'bias_ih_l1',
                # 'bias_hh_l1',
                # 'weight_ih_l0_reverse',
                # 'weight_hh_l0_reverse',
                # 'bias_ih_l0_reverse',
                # 'bias_hh_l0_reverse',
                # 'weight_ih_l1_reverse',
                # 'weight_hh_l1_reverse',
                # 'bias_ih_l1_reverse',
                # 'bias_hh_l1_reverse',
                ]

    np.random.seed(SEED)
    org = torch.Tensor(np.random.rand(1, 80, 512).astype(np.float32))

    lstm = EncoderWithRNN(512, 256)
    sd = np.load('lstm.npy', allow_pickle=True)
    sd = sd.tolist()

    for idx, (key, value) in enumerate(lstm.state_dict().items()):
        print(key, value.shape)
        # lstm.state_dict()[key].copy_(torch.Tensor(sd[map_list[idx]]))

    ret = lstm(org)
    return ret.data.numpy()

if __name__ == '__main__':
    print('==========paddle=================')
    a = paddle_lstm()
    print(a.shape)
    print('a: ', np.sum(a), np.mean(a), np.max(a), np.min(a))
    print('===========pytorch================')
    b = torch_lstm()
    print(b.shape)
    print('b: ', np.sum(b), np.mean(b), np.max(b), np.min(b))
    print('===========pytorch_m================')
    c = torch_lstm_m()
    print(c.shape)
    print('c: ', np.sum(c), np.mean(c), np.max(c), np.min(c))
    print('===========pytorch_bi================')
    d = torch_lstm_bi()
    print(d.shape)
    print('d: ', np.sum(d), np.mean(d), np.max(d), np.min(d))
    print('===========pytorch_di================')
    e = torch_lstm_di()
    print(e.shape)
    print('e: ', np.sum(e), np.mean(e), np.max(e), np.min(e))