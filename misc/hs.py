
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


class Hsigmoid(torch.nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # torch: torch.nn.functional.relu6(x + 3., inplace=self.inplace) / 6.
        # paddle: torch.nn.functional.relu6(1.2 * x + 3., inplace=self.inplace) / 6.
        return torch.nn.functional.relu6(1.2 * x + 3., inplace=self.inplace) / 6.

def paddle_hs():
    np.random.seed(SEED)
    x = np.random.randn(3, 3).astype(np.float32)

    with fluid.dygraph.guard():

        inp = fluid.dygraph.to_variable(x)
        ret = F.activation.hard_sigmoid(inp)

        # print(ret)
    return ret.numpy()

def torch_hs():
    np.random.seed(SEED)
    org = torch.Tensor(np.random.rand(3,3).astype(np.float32))
    tres = Hsigmoid(inplace=True)(org)
    # print(tres)
    return tres.numpy()



if __name__ == '__main__':
    a = paddle_hs()
    b = torch_hs()
    print(a)
    print(b)
    print(a-b)
