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
        for k,v in self.net.state_dict().items():
            keyword = 'block_list.'
            if keyword in k:
                # replace: block_list.
                name = k.replace(keyword, '')
            else:
                name = k

            # for srn backbone
            keyword = 'base_block.'
            if keyword in name:
                # replace: base_block.
                name = name.replace(keyword, '')
            keyword = 'base_block_2.0.'
            if keyword in name:
                # replace: base_block_2.0. -> base_block_2.
                name = name.replace(keyword, 'base_block_2.')
            # for srn head
            keyword = 'encoder_layers.'
            if keyword in name:
                # replace: encoder_layers.
                name = name.replace(keyword, '')
            keyword = 'functors.'
            if keyword in name:
                # replace: functors.
                name = name.replace(keyword, '')


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
            elif 'attention_cell' in name:
                ppname = name

            else:
                print('Redundance:')
                print(name)
                raise ValueError

            try:
                if ppname.endswith('fc.weight'):
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
                elif ppname.endswith('fc1.weight'): # for tps loc
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
                elif ppname.endswith('fc2.weight'): # for tps loc
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
                elif ppname.endswith('.weight') \
                    and len(para_state_dict[ppname].shape) == len(self.net.state_dict()[k].shape) == 2 \
                        and para_state_dict[ppname].shape[0] == self.net.state_dict()[k].shape[1] \
                            and para_state_dict[ppname].shape[1] == self.net.state_dict()[k].shape[0]: # for general fc
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))

                else:
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}'.format(ppname))
                print('paddle: {}'.format(para_state_dict[ppname].shape))
                raise e

def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))
    return res['Architecture']


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help='Assign the yaml path of network configuration', default=None)
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    parser.add_argument("--dst_model_path", type=str, help='save model path in pytorch', default=None)
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        cfg = read_network_config_from_yaml(yaml_path)
    else:
        raise NotImplementedError

    kwargs = {}
    if 'att' in yaml_path:
        kwargs['out_channels'] = 37 + 1
    elif 'srn' in yaml_path:
        kwargs['out_channels'] = 37 + 1
    else:
        kwargs['out_channels'] = 37
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