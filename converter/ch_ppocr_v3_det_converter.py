import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20


class PPOCRv3DetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(PPOCRv3DetConverter, self).__init__(config, **kwargs)
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

            if k.endswith('num_batches_tracked'):
                continue

            ppname = k
            ppname = ppname.replace('.running_mean', '._mean')
            ppname = ppname.replace('.running_var', '._variance')

            if k.startswith('backbone.'):
                ppname = ppname.replace('backbone.', 'Student2.backbone.')
                ppname = ppname.replace('.stages.', '.stage')


            elif k.startswith('neck.'):
                ppname = ppname.replace('neck.', 'Student2.neck.')

            elif k.startswith('head.'):
                ppname = ppname.replace('head.', 'Student2.head.')

            else:
                print('Redundance:')
                print(k)
                raise ValueError

            try:
                self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname, para_state_dict[ppname].shape))
                raise e
        print('model is loaded: {}'.format(weights_path))


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    cfg = {'model_type':'det',
           'algorithm':'DB',
           'Transform':None,
           'Backbone':{'name':'MobileNetV3', 'model_name':'large', 'scale':0.5, 'disable_se':True},
           'Neck':{'name':'RSEFPN', 'out_channels':96, 'shortcut': True},
           'Head':{'name':'DBHead', 'k':50}}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = PPOCRv3DetConverter(cfg, paddle_pretrained_model_path)

    print('todo')

    np.random.seed(666)
    inputs = np.random.randn(1, 3, 640, 640).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    out = out['maps'].data.numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    save_basename = os.path.basename(os.path.abspath(args.src_model_path))

    if save_basename.lower()[:3] == 'en_':
        save_name = 'en_ptocr_v3_det_infer.pth'
    elif save_basename.lower()[:3] == 'ch_':
        save_name = 'ch_ptocr_v3_det_infer.pth'
    else:
        save_name = 'multilingual_ptocr_v3_det_infer.pth'
    converter.save_pytorch_weights(save_name)

    print('done.')