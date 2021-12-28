import os
import sys
import numpy as np
from collections import OrderedDict
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

SEED = 666
INPUT_SIZE = (1, 3, 640, 640)
NAME = 'tabel_det'
tmp_save_name = '{}.npy'.format(NAME)
weights_path = 'PaddleOCR/ppstructure_models/OCR_and_Table/en_ppocr_mobile_v2.0_table_det_train/best_accuracy'

def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))
def compare_ret(pp_ret, pt_ret, info):
    print('============ {} ============='.format(info))
    print('shape:', pp_ret.shape)
    print('pp: ', np.sum(pp_ret), np.mean(pp_ret), np.max(pp_ret), np.min(pp_ret))
    print('ms: ', np.sum(pt_ret), np.mean(pt_ret), np.max(pt_ret), np.min(pt_ret))
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

import paddle
from PaddleOCR.ppocr.modeling.backbones import det_mobilenet_v3
from PaddleOCR.ppocr.modeling.necks import db_fpn
from PaddleOCR.ppocr.modeling.heads import det_db_head
class PPNet(paddle.nn.Layer):
    def __init__(self,**kwargs):
        super(PPNet, self).__init__()

        self.backbone = det_mobilenet_v3.MobileNetV3(
            in_channels=3,
            model_name='large',
            scale=0.5,
            disable_se=False,
        )

        self.neck = db_fpn.DBFPN(
            in_channels=self.backbone.out_channels,
            out_channels=96,
        )

        self.head = det_db_head.DBHead(
            in_channels=96,
            k=50,
        )

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


def paddle_func():
    from paddle import fluid


    np.random.seed(SEED)
    x = np.random.rand(*INPUT_SIZE).astype(np.float32)

    with fluid.dygraph.guard():
        para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
    sd = para_state_dict
    with fluid.dygraph.guard():
        layer = PPNet()
        layer.set_state_dict(sd)
        layer.eval()

        inp = fluid.dygraph.to_variable(x)
        ret = layer(inp)

        # sd = get_pp_static_dict(layer.state_dict())
        # np.save(tmp_save_name, sd, allow_pickle=True)
    return ret['maps'].numpy()


import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20
class PTStructureTableDetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(PTStructureTableDetConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()

    def load_paddle_weights(self, weights_path):
        print('paddle weights loading...')
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)

        [print('paddle: {} ---- {}'.format(k, v.shape)) for k, v in para_state_dict.items()]
        [print('pytorch: {} ---- {}'.format(k, v.shape)) for k, v in self.net.state_dict().items()]

        for k,v in self.net.state_dict().items():
            keyword = 'stages.'
            if keyword in k:
                # replace: stages. -> stage
                name = k.replace(keyword, 'stage')
            else:
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

            self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
        print('model is loaded: {}'.format(weights_path))

def torch_func():
    cfg = {'model_type': 'det',
           'algorithm': 'DB',
           'Transform': None,
           'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': False},
           'Neck': {'name': 'DBFPN', 'out_channels': 96},
           'Head': {'name': 'DBHead', 'k': 50}}
    # paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    net = PTStructureTableDetConverter(cfg, weights_path).net
    net.eval()

    np.random.seed(SEED)
    x = np.random.rand(*INPUT_SIZE).astype(np.float32)
    inp = torch.from_numpy(x)

    with torch.no_grad():
        ret = net(inp)

    return ret['maps'].cpu().numpy()






if __name__ == '__main__':
    clean(tmp_save_name)
    pp = paddle_func()
    print('==========++++=================')
    pt = torch_func()
    [compare_ret(e_pp, e_pt, NAME) for e_pp, e_pt in zip(pp, pt)]
    clean(tmp_save_name)
    print('done.')

