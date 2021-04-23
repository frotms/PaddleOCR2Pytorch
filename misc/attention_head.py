
import os, sys
import numpy as np
import paddle
# paddle.enable_static()
import paddle.fluid as fluid
from paddle import ParamAttr
# import paddle.nn as nn
# import paddle.nn.functional as F
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 666
input_shape = (1, 25, 192)
in_channels = 192
hidden_size = 96
out_channels = 38


class PPAttentionHead(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(PPAttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = PPAttentionGRUCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = paddle.nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = paddle.nn.functional.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = paddle.shape(inputs)[0]
        num_steps = batch_max_length

        hidden = paddle.zeros((batch_size, self.hidden_size))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                output_hiddens.append(paddle.unsqueeze(outputs, axis=1))
            output = paddle.concat(output_hiddens, axis=1)
            probs = self.generator(output)

        else:
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None
            char_onehots = None
            outputs = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                probs_step = self.generator(outputs)
                if probs is None:
                    probs = paddle.unsqueeze(probs_step, axis=1)
                else:
                    probs = paddle.concat(
                        [probs, paddle.unsqueeze(
                            probs_step, axis=1)], axis=1)
                next_input = probs_step.argmax(axis=1)
                targets = next_input

        return probs


class PPAttentionGRUCell(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(PPAttentionGRUCell, self).__init__()
        self.i2h = paddle.nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = paddle.nn.Linear(hidden_size, hidden_size)
        self.score = paddle.nn.Linear(hidden_size, 1, bias_attr=False)

        self.rnn = paddle.nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):

        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden), axis=1)

        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)

        alpha = paddle.nn.functional.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


class PTAttentionHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(PTAttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = PTAttentionGRUCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char.type(torch.int64), onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.size()[0]
        num_steps = batch_max_length

        hidden = torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                output_hiddens.append(torch.unsqueeze(outputs, dim=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)

        else:
            targets = torch.zeros([batch_size],dtype=torch.int32)
            probs = None
            char_onehots = None
            outputs = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                probs_step = self.generator(outputs)
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.cat(
                        [probs, torch.unsqueeze(
                            probs_step, dim=1)], dim=1)
                next_input = probs_step.argmax(dim=1)
                targets = next_input

        return probs


class PTAttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(PTAttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size, bias=True)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)

        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots.float()], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return (cur_hidden, cur_hidden), alpha


def paddle_grucell():
    np.random.seed(SEED)
    x = np.random.rand(input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)

    # np.save('org.npy', x)
    with fluid.dygraph.guard():
        layer = PPAttentionHead(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size)

        # sd = np.load('lstm.npy', allow_pickle=True).tolist()
        # lstm.set_state_dict(sd)

        state_dict = layer.state_dict()
        sd = OrderedDict()
        for key, value in state_dict.items():
            v = value.numpy()
            print(key, value.shape, np.sum(v), np.mean(v), np.max(v), np.min(v))
            sd[key] = v

        np.save('att_head.npy', sd)

        inputs = fluid.dygraph.to_variable(x)
        outputs = layer(inputs, None, 25)
        # print(len(outputs))
    return outputs.numpy()


def torch_grucell():
    np.random.seed(SEED)
    x = np.random.rand(input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
    inputs = torch.Tensor(x)

    layer = PTAttentionHead(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size)

    sd = np.load('att_head.npy', allow_pickle=True)
    sd = sd.tolist()

    for key, value in layer.state_dict().items():
        print(key, value.shape)
        try:
            if key.endswith('.weight'):
                layer.state_dict()[key].copy_(torch.Tensor(sd[key].T))
            else:
                layer.state_dict()[key].copy_(torch.Tensor(sd[key]))
        except Exception as e:
            print('pp: ', key, sd[key].shape)
            print('pt: ', key, layer.state_dict()[key].shape)
            raise e


    outputs = layer(inputs, None, 25)
    return outputs.data.numpy()



if __name__ == '__main__':
    print('==========paddle=================')
    a = paddle_grucell()
    print(a.shape)
    print('a: ', np.sum(a), np.mean(a), np.max(a), np.min(a))
    print('===========pytorch================')
    b = torch_grucell()
    print(b.shape)
    print('b: ', np.sum(b), np.mean(b), np.max(b), np.min(b))