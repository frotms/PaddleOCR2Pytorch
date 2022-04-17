
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
INPUT_SIZE = (1,3,224,224)
KERNEL_SIZE = 1
STRIDES = (1,1)
PADDING = 0


def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))
def compare_ret(pp_ret, pt_ret, info):
    print('============ {} ============='.format(info))
    print('pp: ', np.sum(pp_ret), np.mean(pp_ret), np.max(pp_ret), np.min(pp_ret))
    print('ms: ', np.sum(pt_ret), np.mean(pt_ret), np.max(pt_ret), np.min(pt_ret))
    print('sub: ', np.sum(np.abs(pp_ret-pt_ret)), np.mean(np.abs(pp_ret-pt_ret)))


def hard_swish(x, inplace=True):
    return x * torch.nn.functional.relu6(x + 3., inplace=inplace) / 6.

def paddle_hs():
    np.random.seed(SEED)
    x = np.random.randn(*INPUT_SIZE).astype(np.float32)

    with fluid.dygraph.guard():

        inp = fluid.dygraph.to_variable(x)
        ret = F.hardswish(inp)
        # ret = F.activation.hardsigmoid(inp)

        # print(ret)
    return ret.numpy()

def torch_hs():
    np.random.seed(SEED)
    org = torch.from_numpy(np.random.randn(*INPUT_SIZE).astype(np.float32))
    # ret = Hsigmoid(inplace=True, slope=0.)(org)
    ret = hard_swish(org)
    return ret.numpy()



if __name__ == '__main__':
    a = paddle_hs()
    b = torch_hs()
    compare_ret(a,b,'hardsigmoid')
