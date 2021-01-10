
import os, sys
import numpy as np
import paddle
# paddle.enable_static()
import paddle.fluid as fluid
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

import torch

SEED = 666
# INPUT_SIZE = 89
KERNEL_SIZE = 1
STRIDES = (1,1)
PADDING = 0

def paddle_fc():
    np.random.seed(SEED)
    x = np.random.randn(1, 200).astype(np.float32)
    with fluid.dygraph.guard():
        simple_conv = nn.Linear(200,
                                2,
                                weight_attr=ParamAttr(name='simple_conv' + "_weights"),
                                bias_attr=ParamAttr(name='simple_conv' + "_bias", initializer=nn.initializer.Uniform(-1, 1)))
        inp = fluid.dygraph.to_variable(x)
        ret = simple_conv(inp)

        np.save('fc_w.npy', list(simple_conv.state_dict().values())[0].numpy())
        np.save('fc_b.npy', list(simple_conv.state_dict().values())[1].numpy())

        print(ret)
    return ret.numpy()

def torch_fc():
    np.random.seed(SEED)
    org = torch.Tensor(np.random.rand(1,200).astype(np.float32))
    tfc = torch.nn.Linear(200, 2)

    fc_w = np.load('fc_w.npy')
    fc_b = np.load('fc_b.npy')
    tfc.state_dict()['weight'].copy_(torch.Tensor(fc_w.T))
    tfc.state_dict()['bias'].copy_(torch.Tensor(fc_b))

    tres = tfc(org)
    print(tres)



if __name__ == '__main__':
    paddle_fc()
    torch_fc()
