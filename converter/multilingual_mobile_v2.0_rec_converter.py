# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class MultilingualV20RecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        out_channels = list(para_state_dict.values())[-1].shape[0]
        print('out_channels: ', out_channels)
        print(type(kwargs), kwargs)
        kwargs['out_channels'] = out_channels
        super(MultilingualV20RecConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        print(list(para_state_dict.values())[-1].shape)
        for k,v in self.net.state_dict().items():
            keyword = 'block_list.'
            if keyword in k:
                # replace: block_list.
                name = k.replace(keyword, '')
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


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    parser.add_argument("--dst_model_path", type=str, help='save model path in pytorch', default=None)
    args = parser.parse_args()

    cfg = {'model_type':'rec',
           'algorithm':'CRNN',
           'Transform':None,
           'Backbone':{'model_name':'small', 'name':'MobileNetV3', 'scale':0.5, 'small_stride':[1,2,2,2]},
           'Neck':{'name':'SequenceEncoder', 'hidden_size':48, 'encoder_type':'rnn'},
           'Head':{'name':'CTCHead', 'fc_decay': 4e-05},
           }
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = MultilingualV20RecConverter(cfg, paddle_pretrained_model_path)

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

    # out = converter.net(inp)
    # out = out.data.numpy()
    # print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    if args.dst_model_path is not None:
        save_name = args.dst_model_path
    else:
        save_name = '{}infer.pth'.format(os.path.basename(os.path.dirname(paddle_pretrained_model_path))[:-5])
    converter.save_pytorch_weights(save_name)
    print('done.')