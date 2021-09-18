import os, sys
from typing import List
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn

SEED = 666
INPUT_SIZE = (1, 80, 512)
NAME = 'lstm'
tmp_save_name = '{}.npy'.format(NAME)
def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))
def compare_ret(pp_ret, pt_ret, info):
    print('============ {} ============='.format(info))
    print('pt official: ', np.sum(pp_ret), np.mean(pp_ret), np.max(pp_ret), np.min(pp_ret))
    print('pt diy: ', np.sum(pt_ret), np.mean(pt_ret), np.max(pt_ret), np.min(pt_ret))
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

def lstm_official():
    np.random.seed(SEED)
    org = torch.from_numpy(np.random.rand(*INPUT_SIZE).astype(np.float32))

    lstm = torch.nn.LSTM(512, 256, num_layers=1, batch_first=True, bidirectional=False)

    sd = OrderedDict()
    for key, value in lstm.state_dict().items():
        # print(key, value.shape)
        # lstm.state_dict()[key].copy_(torch.Tensor(sd[key]))
        sd[key] = value.numpy()
    np.save(tmp_save_name, sd)

    with torch.no_grad():
        ret, _ = lstm(org)
    return ret.data.numpy()


def lstm_diy():
    np.random.seed(SEED)
    org = torch.from_numpy(np.random.rand(*INPUT_SIZE).astype(np.float32))

    lstm = UniDirectionalLSTM(512, 256)
    sd = get_np_static_dict(tmp_save_name)
    for key, value in lstm.state_dict().items():
        # print(key, value.shape)
        key_w = '{}'.format(key)
        lstm.state_dict()[key].copy_(torch.Tensor(sd[key_w]))

    with torch.no_grad():
        ret, _ = lstm(org)
    return ret.data.numpy()

# https://github.com/piEsposito/pytorch-lstm-by-hand/blob/master/LSTM.ipynb
class UniDirectionalLSTM(nn.Module):
    def __init__(self, input_size , hidden_size, bias=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # self.weight_ih_l0 = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        # self.weight_hh_l0 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.weight_ih_l0 = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if self.bias:
            self.bias_ih_l0 = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh_l0 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            # gates = x_t @ self.weight_ih_l0.T + h_t @ self.weight_hh_l0.T # + self.bias_ih_l0 + self.bias_hh_l0
            gates = torch.mm(x_t, self.weight_ih_l0.T) + torch.mm(h_t, self.weight_hh_l0.T) # + self.bias_ih_l0 + self.bias_hh_l0
            if self.bias:
                gates += self.bias_ih_l0 + self.bias_hh_l0
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(ingate),  # input
                torch.sigmoid(forgetgate),  # forget
                torch.tanh(cellgate),
                torch.sigmoid(outgate),  # output
            )
            # i_t, f_t, g_t, o_t = (
            #     torch.sigmoid(gates[:, :HS]),  # input
            #     torch.sigmoid(gates[:, HS:HS * 2]),  # forget
            #     torch.tanh(gates[:, HS * 2:HS * 3]),
            #     torch.sigmoid(gates[:, HS * 3:]),  # output
            # )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


########################################################################################################################
# https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py#L184
# https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py#L93
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L24
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L470
# https://github.com/piEsposito/pytorch-lstm-by-hand/blob/master/LSTM.ipynb
# https://github.com/diaomin/crnn-mxnet-chinese-text-recognition/blob/master/fit/lstm.py

class SimpleLSTMJit(torch.jit.ScriptModule):
    def __init__(self, input_size , hidden_size, bias=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

    @torch.jit.script_method
    def forward(self, x, ):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                    torch.zeros(bs, self.hidden_size).to(x.device))

        hidden_seq = torch.jit.annotate(List[torch.Tensor], [])
        for t in range(seq_sz):
            x_in = x[:,t,:]
            gates = (torch.mm(x_in, self.weight_ih.t()) + self.bias_ih +
                     torch.mm(h_t, self.weight_hh.t()) + self.bias_hh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            c_t = (forgetgate * c_t) + (ingate * cellgate)
            h_t = outgate * torch.tanh(c_t)

            hidden_seq += [h_t]

        outputs = torch.stack(hidden_seq).transpose(0, 1).contiguous()
        return outputs, (h_t, c_t)

class UniDirectionalLSTM_using_lstmcell_jit(torch.jit.ScriptModule):
    def __init__(self, input_size , hidden_size, bias=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.LSTMCell = LSTMCell_jit(input_size, hidden_size)

    @torch.jit.script_method
    def forward(self, x, ):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                    torch.zeros(bs, self.hidden_size).to(x.device))
        # if h_t is None and c_t is None:
        #     h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
        #                 torch.zeros(bs, self.hidden_size).to(x.device))
            # h_t, c_t = init_states

        hidden_seq = torch.jit.annotate(List[torch.Tensor], [])
        for t in range(seq_sz):
            out, (h_t, c_t) = self.LSTMCell(x[:,t,:], h_t, c_t)
            hidden_seq += [out]
        outputs = torch.stack(hidden_seq).transpose(0, 1).contiguous()
        return outputs, (h_t, c_t)

# https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py#L93
class LSTMCell_jit(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell_jit, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

    @torch.jit.script_method
    def forward(self, input, hx, cx):
        # hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_size , hidden_size, bias=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, x, ):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                    torch.zeros(bs, self.hidden_size).to(x.device))

        hidden_seq = []
        for t in range(seq_sz):
            x_in = x[:,t,:]
            gates = (torch.mm(x_in, self.weight_ih.t()) + self.bias_ih +
                     torch.mm(h_t, self.weight_hh.t()) + self.bias_hh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            c_t = (forgetgate * c_t) + (ingate * cellgate)
            h_t = outgate * torch.tanh(c_t)

            hidden_seq += [h_t]

        outputs = torch.stack(hidden_seq).transpose(0, 1).contiguous()
        return outputs, (h_t, c_t)

class UniDirectionalLSTM_using_lstmcell(nn.Module):
    def __init__(self, input_size , hidden_size, bias=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.LSTMCell = LSTMCell(input_size, hidden_size)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        state = (h_t, c_t)

        hidden_seq = []
        for t in range(seq_sz):
            out, state = self.LSTMCell(x[:,t,:], state)
            hidden_seq += [out]
        outputs = torch.stack(hidden_seq).transpose(0, 1).contiguous()
        return outputs, state

# https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py#L93
class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, x, state):
        hx, cx = state
        gates = (torch.mm(x, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


def lstm_diy_2():
    np.random.seed(SEED)
    org = torch.from_numpy(np.random.rand(*INPUT_SIZE).astype(np.float32))

    lstm = SimpleLSTM(512, 256)
    sd = get_np_static_dict(tmp_save_name)
    for key, value in lstm.state_dict().items():
        # print(key, value.shape)
        key_w = '{}_l0'.format(key)
        lstm.state_dict()[key].copy_(torch.Tensor(sd[key_w]))

    with torch.no_grad():
        ret, _ = lstm(org)
    return ret.data.numpy()

def lstm_diy_3():
    np.random.seed(SEED)
    org = torch.from_numpy(np.random.rand(*INPUT_SIZE).astype(np.float32))

    lstm = SimpleLSTMJit(512, 256)
    sd = get_np_static_dict(tmp_save_name)
    for key, value in lstm.state_dict().items():
        # print(key, value.shape)
        key_w = '{}_l0'.format(key)
        lstm.state_dict()[key].copy_(torch.Tensor(sd[key_w]))

    with torch.no_grad():
        ret, _ = lstm(org)
    return ret.data.numpy()

if __name__ == '__main__':
    clean(tmp_save_name)
    pp = lstm_official()
    print('==========++++=================')
    pt = lstm_diy()
    compare_ret(pp, pt, NAME)
    print('==========++++=================')
    pt_2 = lstm_diy_2()
    compare_ret(pp, pt_2, NAME)
    print('==========++++=================')
    pt_3 = lstm_diy_3()
    compare_ret(pp, pt_3, NAME)
    clean(tmp_save_name)
    print('done.')