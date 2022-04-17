# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class RecV20RecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        print('config: ', config)
        print(type(kwargs), kwargs)
        super(RecV20RecConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        # [print('paddle: {} ---- {}'.format(k, v.shape)) for k,v in para_state_dict.items()]
        # [print('torch: {} ---- {}'.format(k, v.shape)) for k,v in self.net.state_dict().items()]
        # exit()

        for k,v in para_state_dict.items():
            name = k
            if k.endswith('._mean'):
                name = k.replace('._mean', '.running_mean')
            if k.endswith('._variance'):
                name = k.replace('._variance', '.running_var')

            try:
                if k.endswith('.weight'):
                    if len(v.shape) == len(self.net.state_dict()[name].shape) == 2 \
                        and v.shape[0] == self.net.state_dict()[name].shape[1] \
                        and v.shape[1] == self.net.state_dict()[name].shape[0]:
                        self.net.state_dict()[name].copy_(torch.Tensor(v.T))
                    else:
                        self.net.state_dict()[name].copy_(torch.Tensor(v))
                else:
                    self.net.state_dict()[name].copy_(torch.Tensor(v))
            except Exception as e:
                print('pytorch: {}'.format(name))
                print('paddle: {}'.format(k))
                print('pytorch: {}'.format(self.net.state_dict()[name].shape))
                print('paddle: {}'.format(v.shape))
                print(e)
                raise e


        print('weights are loaded.')

        # for k,v in self.net.state_dict().items():
        #     keyword = 'block_list.'
        #     if keyword in k:
        #         # replace: block_list.
        #         name = k.replace(keyword, '')
        #     else:
        #         name = k
        #
        #     if name.endswith('num_batches_tracked'):
        #         continue
        #
        #     if name.endswith('running_mean'):
        #         ppname = name.replace('running_mean', '_mean')
        #     elif name.endswith('running_var'):
        #         ppname = name.replace('running_var', '_variance')
        #     elif name.endswith('bias') or name.endswith('weight'):
        #         ppname = name
        #     elif 'lstm' in name:
        #         ppname = name
        #     elif 'attention_cell' in name:
        #         ppname = name
        #
        #     else:
        #         print('Redundance:')
        #         print(name)
        #         raise ValueError
        #
        #     try:
        #         if ppname.endswith('fc.weight'):
        #             self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
        #         elif ppname.endswith('fc1.weight'): # for tps loc
        #             self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
        #         elif ppname.endswith('fc2.weight'): # for tps loc
        #             self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
        #         elif ppname.endswith('.weight') \
        #             and len(para_state_dict[ppname].shape) == len(self.net.state_dict()[k].shape) == 2 \
        #                 and para_state_dict[ppname].shape[0] == self.net.state_dict()[k].shape[1] \
        #                     and para_state_dict[ppname].shape[1] == self.net.state_dict()[k].shape[0]: # for general fc
        #             self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
        #
        #         else:
        #             self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
        #     except Exception as e:
        #         print('pytorch: {}, {}'.format(k, v.size()))
        #         print('paddle: {}'.format(ppname))
        #         print('paddle: {}'.format(para_state_dict[ppname].shape))
        #         raise e

def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))
    if res.get('Global') is None:
        raise ValueError('{} has no Global'.format(yaml_path))
    if res['Global']['use_space_char'] is None:
        raise ValueError('res["Global"] has no use_space_char')
    return res['Architecture'], res['Global']['use_space_char']


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help='Assign the yaml path of network configuration', default=None)
    parser.add_argument("--dict_path", type=str, help='Assign the dict.txt path of network configuration', default=None)
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    parser.add_argument("--dst_model_path", type=str, help='save model path in pytorch', default=None)
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        cfg, use_space_char = read_network_config_from_yaml(yaml_path)

    else:
        raise NotImplementedError
        # cfg = {'model_type':'rec',
        #        'algorithm':'CRNN',
        #        'Transform':None,
        #        'Backbone':{'model_name':'small', 'name':'MobileNetV3', 'scale':0.5, 'small_stride':[1,2,2,2]},
        #        'Neck':{'name':'SequenceEncoder', 'hidden_size':48, 'encoder_type':'rnn'},
        #        'Head':{'name':'CTCHead', 'fc_decay': 4e-05},
        #        }
    kwargs = {}

    dict_path = args.dict_path
    if dict_path is not None:
        if not os.path.exists(dict_path):
            raise FileNotFoundError(dict_path)
    else:
        ValueError('Assign the dict.txt path of network configuration')

    dict_character = ['blank', '<unk>', '<s>', '</s>']
    out_channels = len(dict_character)+1 # ['blank', '<unk>', '<s>', '</s>'] + use_space?
    with open(dict_path, 'r', encoding='utf-8') as f:
        num_chars = len(f.readlines())
    out_channels += num_chars
    kwargs['out_channels'] = out_channels

    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = RecV20RecConverter(cfg, paddle_pretrained_model_path, **kwargs)

    # image = cv2.imread('./doc/imgs_words_en/word_10.png')
    # image = cv2.resize(image, (100, 32))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = np.expand_dims(image, -1)
    # mean = 0.5
    # std = 0.5
    # scale = 1. / 255
    # norm_img = (image * scale - mean) / std
    # transpose_img = norm_img.transpose(2, 0, 1)
    # transpose_img = np.expand_dims(transpose_img, 0).astype(np.float32)
    # print('inp:', np.sum(transpose_img), np.mean(transpose_img), np.max(transpose_img), np.min(transpose_img))
    # with torch.no_grad():
    #     inp = torch.Tensor(transpose_img)
    #     out = converter.net(inp)
    # out = out.data.numpy()
    # print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    if args.dst_model_path is not None:
        save_name = args.dst_model_path
    else:
        save_name = '{}infer.pth'.format(os.path.basename(os.path.dirname(paddle_pretrained_model_path))[:-5])
    converter.save_pytorch_weights(save_name)
    print('done.')