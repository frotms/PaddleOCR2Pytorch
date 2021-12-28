import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20


class PPStructureTableDetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(PPStructureTableDetConverter, self).__init__(config, **kwargs)
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


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    cfg = {'model_type':'det',
           'algorithm':'DB',
           'Transform':None,
           'Backbone':{'name':'MobileNetV3', 'model_name':'large', 'scale':0.5, 'disable_se':False},
           'Neck':{'name':'DBFPN', 'out_channels':96},
           'Head':{'name':'DBHead', 'k':50}}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = PPStructureTableDetConverter(cfg, paddle_pretrained_model_path)

    print('todo')

    np.random.seed(666)
    inp = torch.from_numpy(np.random.randn(1, 3, 640, 640).astype(np.float32))
    with torch.no_grad():
        out = converter.net(inp)['maps'].cpu().numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    converter.save_pytorch_weights('en_ptocr_mobile_v2.0_table_det_infer.pth')
    print('done.')