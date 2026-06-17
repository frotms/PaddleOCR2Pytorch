import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch

from pytorchocr.modeling.architectures.base_model import BaseModel

class BaseOCRV20:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()


    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)


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
            elif 'lstm' in name:
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

    def read_pytorch_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = torch.load(weights_path)
        return weights

    def get_out_channels(self, weights):
        if list(weights.keys())[-1].endswith('.weight') and len(list(weights.values())[-1].shape) == 2:
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        print('weights is loaded.')

    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path))
        print('model is loaded: {}'.format(weights_path))


    def save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.net.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.net.state_dict(), weights_path) # _use_new_zipfile_serialization=False for torch>=1.6.0
        print('model is saved: {}'.format(weights_path))


    def print_pytorch_state_dict(self):
        print('pytorch:')
        for k,v in self.net.state_dict().items():
            print('{}----{}'.format(k,type(v)))

    def read_paddle_weights(self, weights_path):
        try:
            import paddle.fluid as fluid
            with fluid.dygraph.guard():
                para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        except:
            import paddle
            import sys
            import numpy
            # Workaround for Paddle 3.0 pickle format requiring numpy 2.x APIs
            # Paddle 3.0+ stores tensors using numpy._core namespace (numpy >= 2.0)
            # but we have numpy < 2.0 with numpy.core namespace
            if 'numpy._core' not in sys.modules:
                # Create a proxy module that redirects numpy._core.* to numpy.core.*
                import types
                _core = types.ModuleType('numpy._core')
                # Redirect common numpy._core submodules to their numpy equivalents
                _core.multiarray = numpy.core.multiarray
                _core._multiarray_umath = numpy.core._multiarray_umath
                _core.numerictypes = numpy.core.numerictypes
                _core.umath = numpy.core.umath
                _core._methods = numpy.core._methods
                _core.fromnumeric = numpy.core.fromnumeric
                _core.shape_base = numpy.core.shape_base
                _core.records = numpy.core.records
                sys.modules['numpy._core'] = _core
                sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
                sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
                sys.modules['numpy._core.numerictypes'] = numpy.core.numerictypes
                sys.modules['numpy._core.umath'] = numpy.core.umath
                sys.modules['numpy._core._methods'] = numpy.core._methods
                sys.modules['numpy._core.fromnumeric'] = numpy.core.fromnumeric
            para_state_dict = paddle.load(weights_path)
            opti_state_dict = None
        return para_state_dict, opti_state_dict

    def print_paddle_state_dict(self, weights_path):
        try:
            import paddle.fluid as fluid
            with fluid.dygraph.guard():
                para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        except:
            import paddle
            para_state_dict = paddle.load(weights_path)
        print('paddle"')
        for k,v in para_state_dict.items():
            print('{}----{}'.format(k,type(v)))


    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer
