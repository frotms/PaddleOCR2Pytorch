import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch

from pytorchocr.modeling.architectures.base_model import BaseModel

class BaseOCRV20:
    def __init__(self, config):
        self.config = config
        self.build_net()
        self.net.eval()


    def build_net(self):
        self.net = BaseModel(self.config)


    def load_paddle_weights(self, weights_path):
        raise NotImplementedError('implemented in converter.')
        print('paddle weights loading...')
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)

        for k,v in self.net.state_dict().items():
            name = k

            if name.endswith('num_batches_tracked'):
                continue

            if name.endswith('running_mean'):
                ppname = name.replace('running_mean', '_mean')
            elif name.endswith('running_var'):
                ppname = name.replace('running_var', '_variance')
            elif name.endswith('bias') or name.endswith('weight'):
                ppname = name

            else:
                print('Redundance:')
                print(name)
                raise ValueError
            try:
                if ppname.endswith('fc.weight'):
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
                else:
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname, para_state_dict[ppname].shape))
                raise e

        print('model is loaded: {}'.format(weights_path))


    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path))
        print('model is loaded: {}'.format(weights_path))


    def save_pytorch_weights(self, weights_path):
        torch.save(self.net.state_dict(), weights_path)
        print('model is saved: {}'.format(weights_path))


    def print_pytorch_state_dict(self):
        print('pytorch:')
        for k,v in self.net.state_dict().items():
            print('{}----{}'.format(k,type(v)))


    def print_paddle_state_dict(self, weights_path):
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        print('paddle"')
        for k,v in para_state_dict.items():
            print('{}----{}'.format(k,type(v)))


    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer
