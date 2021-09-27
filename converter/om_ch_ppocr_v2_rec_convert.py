# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class PPOCRv2RecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        out_channels = list(para_state_dict.values())[-1].shape[0]
        print('out_channels: ', out_channels)
        print(type(kwargs), kwargs)
        kwargs['out_channels'] = out_channels
        super(PPOCRv2RecConverter, self).__init__(config, **kwargs)
        # self.load_paddle_weights(paddle_pretrained_model_path)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        [print('paddle: {} ---- {}'.format(k, v.shape)) for k, v in para_state_dict.items()]
        [print('pytorch: {} ---- {}'.format(k, v.shape)) for k, v in self.net.state_dict().items()]

        for k,v in self.net.state_dict().items():

            if k.endswith('num_batches_tracked'):
                continue

            ppname = k
            ppname = ppname.replace('.running_mean', '._mean')
            ppname = ppname.replace('.running_var', '._variance')

            if k.startswith('backbone.'):
                ppname = ppname.replace('backbone.', 'Student.backbone.')

            elif k.startswith('neck.'):
                ppname = ppname.replace('neck.', 'Student.neck.')

                if 'lstm' in ppname:
                    ppname = ppname.replace('lstm_0_', 'lstm.0.')
                    ppname = ppname.replace('lstm_1_', 'lstm.1.')
                    ppname = ppname.replace('_ih_l0', '_ih')
                    ppname = ppname.replace('_hh_l0', '_hh')


            elif k.startswith('head.'):
                ppname = ppname.replace('head.', 'Student.head.')

            else:
                print('Redundance:')
                print(k)
                raise ValueError

            try:
                if ppname.endswith('fc1.weight') or ppname.endswith('fc2.weight'):
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
                else:
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))

            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname, para_state_dict[ppname].shape))
                raise e

        print('model is loaded.')


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    cfg = {'model_type':'rec',
           'algorithm':'CRNN',
           'Transform':None,
           'Backbone':{'name':'MobileNetV1Enhance', 'scale':0.5},
           'Neck':{'name':'SequenceEncoder', 'hidden_size':64, 'encoder_type':'om'},
           'Head':{'name':'CTCHead', 'mid_channels': 96, 'fc_decay': 2e-05}}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = PPOCRv2RecConverter(cfg, paddle_pretrained_model_path)

    # image = cv2.imread('images/Snipaste.jpg')
    # image = cv2.resize(image, (320, 32))
    # mean = 0.5
    # std = 0.5
    # scale = 1. / 255
    # norm_img = (image * scale - mean) / std
    # transpose_img = norm_img.transpose(2, 0, 1)
    # transpose_img = np.expand_dims(transpose_img, 0).astype(np.float32)
    # inp = torch.Tensor(transpose_img)
    # print('inp:', np.sum(transpose_img), np.mean(transpose_img), np.max(transpose_img), np.min(transpose_img))

    np.random.seed(666)
    inputs = np.random.randn(1,3,32,320).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    out = out.data.numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))
    # print(out['maps'].data.numpy())

    # save
    # converter.save_pytorch_weights('ch_ptocr_server_v2.0_rec_infer.pth')
    print('done.')

    dummy_input = torch.autograd.Variable(torch.randn(1, 3, 32, 320))
    dynamic_axes = {'input.1': [0, 1, 2, 3],
                    '808': [0, 1],
                    }
    torch.onnx.export(converter.net, dummy_input, 'ch_ptocr_v2_rec_infer.onnx', opset_version=11,
                      do_constant_folding=False, verbose=False,
                      dynamic_axes=dynamic_axes,
                      )