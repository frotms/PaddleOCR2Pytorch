import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20


class MobileV20DetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(MobileV20DetConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()

    def load_paddle_weights(self, weights_path):
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


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    cfg = {'model_type':'cls',
           'algorithm':'CLS',
           'Transform':None,
           'Backbone':{'name':'MobileNetV3', 'model_name':'small', 'scale':0.35},
           'Neck':None,
           'Head':{'name':'ClsHead', 'class_dim':2}}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')

    converter = MobileV20DetConverter(cfg, paddle_pretrained_model_path)
    print('todo')

    # image = cv2.imread('images/Snipaste.jpg')
    # image = cv2.resize(image, (192, 48))
    # mean = 0.5
    # std = 0.5
    # scale = 1. / 255
    # norm_img = (image * scale - mean) / std
    # transpose_img = norm_img.transpose(2, 0, 1)
    # transpose_img = np.expand_dims(transpose_img, 0)
    # inputs = transpose_img.astype(np.float32)

    # print(np.sum(inputs), np.mean(inputs), np.max(inputs), np.min(inputs))
    # print('done')

    # inp = torch.Tensor(inputs)

    # out = converter.net(inp)
    # print('out:')
    # print(out.data.numpy())

    # save
    converter.save_pytorch_weights('ch_ptocr_mobile_v2.0_cls_infer.pth')
    print('done.')