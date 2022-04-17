
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


class Hsigmoid(torch.nn.Module):
    def __init__(self, inplace=True, slope=0.2, offset=0.5):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace
        self.slope = slope
        self.offset = offset
        self.bias = 6. * self.offset

    def forward(self, x):
        # torch: torch.nn.functional.relu6(x + 3., inplace=self.inplace) / 6.
        # paddle: torch.nn.functional.relu6(1.2 * x + 3., inplace=self.inplace) / 6.
        return torch.nn.functional.relu6((1.+self.slope) * x + self.bias, inplace=self.inplace) / 6.

def hard_sigmoid(x, slope=0.1666667, offset=0.5, inplace=True):
    # return torch.nn.functional.relu6((1.+slope) * x + offset*6, inplace=inplace) / 6.
    return torch.clamp(slope * x + offset, 0., 1.)

def paddle_hs():
    np.random.seed(SEED)
    x = np.random.randn(*INPUT_SIZE).astype(np.float32)

    with fluid.dygraph.guard():

        inp = fluid.dygraph.to_variable(x)
        # ret = F.hardsigmoid(inp, slope=0.2, offset=0.5)
        ret = F.hardsigmoid(inp, slope=0.2, offset=0.5)
        # ret = F.hardsigmoid(inp)

        # print(ret)
    return ret.numpy()

def torch_hs():
    np.random.seed(SEED)
    org = torch.from_numpy(np.random.randn(*INPUT_SIZE).astype(np.float32))
    # hs = Hsigmoid(inplace=True, slope=0.2)(org)
    # ret = hs
    ret = hard_sigmoid(org, slope=0.2, offset=0.5)
    return ret.numpy()



if __name__ == '__main__':
    a = paddle_hs()
    b = torch_hs()
    compare_ret(a,b,'hardsigmoid')
