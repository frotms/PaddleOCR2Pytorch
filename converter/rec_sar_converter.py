# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class RecSARConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        print('config: ', config)
        print(type(kwargs), kwargs)
        super(RecSARConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        # [print('paddle: {} ---- {}'.format(k, v.shape)) for k,v in para_state_dict.items()]
        # [print('torch: {} ---- {}'.format(k, v.shape)) for k,v in self.net.state_dict().items()]
        # exit()

        for k,v in self.net.state_dict().items():
            name = k
            if k.endswith('num_batches_tracked'):
                continue
            if k.endswith('.running_mean'):
                name = k.replace('.running_mean', '._mean')
            if k.endswith('.running_var'):
                name = k.replace('.running_var', '._variance')

            try:
                if k.endswith('.weight'):
                    if len(v.shape) == len(para_state_dict[name].shape) == 2 \
                        and v.shape[0] == para_state_dict[name].shape[1] \
                        and v.shape[1] == para_state_dict[name].shape[0]:
                        self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[name].T))
                    else:
                        self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[name]))
                else:
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[name]))
            except Exception as e:
                print('pytorch: {}'.format(k))
                print('paddle: {}'.format(name))
                print('pytorch: {}'.format(v.shape))
                print('paddle: {}'.format(para_state_dict[name].shape))
                print(e)
                raise e

        print('weights are loaded.')


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

    kwargs = {}

    dict_path = args.dict_path
    if dict_path is not None:
        if not os.path.exists(dict_path):
            raise FileNotFoundError(dict_path)
    else:
        ValueError('Assign the dict.txt path of network configuration')

    with open(dict_path, 'r', encoding='utf-8') as f:
        num_chars = len(f.readlines())
    beg_end_str = "<BOS/EOS>"
    unknown_str = "<UKN>"
    padding_str = "<PAD>"
    aux_str_list = [beg_end_str, unknown_str, padding_str]
    out_channels = num_chars + len(aux_str_list)
    kwargs['out_channels'] = out_channels

    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = RecSARConverter(cfg, paddle_pretrained_model_path, **kwargs)


    inp = torch.from_numpy(np.random.randn(1,3,48,160).astype(np.float32))
    with torch.no_grad():
        out = converter.net(inp)
        print('out: ', out.shape)

    # save
    if args.dst_model_path is not None:
        save_name = args.dst_model_path
    else:
        save_name = '{}infer.pth'.format(os.path.basename(os.path.dirname(paddle_pretrained_model_path))[:-5])
    converter.save_pytorch_weights(save_name)
    print('done.')