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

from pp_rec_srn_head import SRNHead as PPHead
from pt_rec_srn_head import SRNHead as PTHead

SEED = 666
input_shape = (1, 512, 8, 32)
in_channels = 512
out_channels = 38
max_text_length = 25
num_heads = 8
num_encoder_TUs = 2
num_decoder_TUs = 4
hidden_dims = 512

def paddle_func():
    np.random.seed(SEED)
    # x = np.load('input.npy', allow_pickle=True)
    x = np.random.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
    print('pp input size: {}'.format(x.shape))
    # x1 = np.random.rand(1, 256, 1).astype(np.float32)
    x1 = np.load('encoder_word_pos_list.npy', allow_pickle=True)
    # x2 = np.random.rand(1, 25, 1).astype(np.int)
    x2 = np.load('gsrm_word_pos_list.npy', allow_pickle=True)
    # x3 = np.random.rand(1, 8, 25, 25).astype(np.float32)
    x3 = np.load('gsrm_slf_attn_bias1_list.npy', allow_pickle=True)
    # x4 = np.random.rand(1, 8, 25, 25).astype(np.float32)
    x4 = np.load('gsrm_slf_attn_bias2_list.npy', allow_pickle=True)

    # np.save('org.npy', x)
    with fluid.dygraph.guard():
        layer = PPHead(in_channels=in_channels,
                       out_channels=out_channels,
                       max_text_length=max_text_length,
                       num_heads=num_heads,
                       num_encoder_TUs=num_encoder_TUs,
                       num_decoder_TUs=num_decoder_TUs,
                       hidden_dims=hidden_dims)

        # sd = np.load('lstm.npy', allow_pickle=True).tolist()
        # lstm.set_state_dict(sd)

        state_dict = layer.state_dict()
        sd = OrderedDict()
        for key, value in state_dict.items():
            v = value.numpy()
            # print(key, value.shape, np.sum(v), np.mean(v), np.max(v), np.min(v))
            sd[key] = v

        np.save('srnhead.npy', sd)

        inputs = fluid.dygraph.to_variable(x)
        x1 = fluid.dygraph.to_variable(x1)
        x2 = fluid.dygraph.to_variable(x2)
        x3 = fluid.dygraph.to_variable(x3)
        x4 = fluid.dygraph.to_variable(x4)
        out = layer(inputs, [x1,x2,x3,x4])

    predict = out['predict']
    pvam_feature = out['pvam_feature']
    decoded_out = out['decoded_out']
    word_out = out['word_out']
    gsrm_out = out['gsrm_out']
    return predict.numpy(), pvam_feature.numpy(), decoded_out.numpy(), word_out.numpy(), gsrm_out.numpy()
    # return alpha.numpy()


def torch_func():
    np.random.seed(SEED)
    # x = np.load('input.npy', allow_pickle=True)
    x = np.random.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
    print('pt input size: {}'.format(x.shape))
    # x1 = np.random.rand(1, 256, 1).astype(np.float32)
    x1 = np.load('encoder_word_pos_list.npy', allow_pickle=True)
    # x2 = np.random.rand(1, 25, 1).astype(np.int)
    x2 = np.load('gsrm_word_pos_list.npy', allow_pickle=True)
    # x3 = np.random.rand(1, 8, 25, 25).astype(np.float32)
    x3 = np.load('gsrm_slf_attn_bias1_list.npy', allow_pickle=True)
    # x4 = np.random.rand(1, 8, 25, 25).astype(np.float32)
    x4 = np.load('gsrm_slf_attn_bias2_list.npy', allow_pickle=True)

    inputs = torch.Tensor(x)
    x1 = torch.Tensor(x1)
    x2 = torch.Tensor(x2)
    x3 = torch.Tensor(x3)
    x4 = torch.Tensor(x4)
    layer = PTHead(in_channels=in_channels,
                       out_channels=out_channels,
                       max_text_length=max_text_length,
                       num_heads=num_heads,
                       num_encoder_TUs=num_encoder_TUs,
                       num_decoder_TUs=num_decoder_TUs,
                       hidden_dims=hidden_dims)

    sd = np.load('srnhead.npy', allow_pickle=True)
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
        # for srn head
        keyword = 'encoder_layers.'
        if keyword in name:
            # replace: encoder_layers.
            name = name.replace(keyword, '')
        keyword = 'functors.'
        if keyword in name:
            # replace: functors.
            name = name.replace(keyword, '')


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

    out = layer(inputs, [x1,x2,x3,x4])
    predict = out['predict']
    pvam_feature = out['pvam_feature']
    decoded_out = out['decoded_out']
    word_out = out['word_out']
    gsrm_out = out['gsrm_out']
    return predict.data.numpy(), pvam_feature.data.numpy(), decoded_out.data.numpy(), word_out.data.numpy(), gsrm_out.data.numpy()
    # return alpha.data.numpy()

def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))

if __name__ == '__main__':
    print('==========paddle=================')
    predict, pvam_feature, decoded_out, word_out, gsrm_out = paddle_func()
    print_cmp(predict, name='predict')
    print('===========pytorch================')
    predict, pvam_feature, decoded_out, word_out, gsrm_out = torch_func()
    print_cmp(predict, name='predict')