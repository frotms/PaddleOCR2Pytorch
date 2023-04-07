# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class SRConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        print('config: ', config)
        print(type(kwargs), kwargs)
        super(SRConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        # [print('paddle: {}---- {} ---- {}'.format(i, n, s.shape)) for i, (n,s) in enumerate(para_state_dict.items())]
        # [print('torch: {}---- {} ---- {}'.format(i, n, s.shape)) for i, (n, s) in enumerate(self.net.state_dict().items())]
        for k,v in para_state_dict.items():
            name = k

            name = name.replace('._weight', '.weight')

            if name.endswith('num_batches_tracked'):
                continue

            if name.endswith('.cell_fw.weight_ih') or name.endswith('.cell_fw.weight_hh') \
                    or name.endswith('.cell_fw.bias_ih') or name.endswith('.cell_fw.bias_hh') \
                    or name.endswith('.cell_bw.weight_ih') or name.endswith('.cell_bw.weight_hh') \
                    or name.endswith('.cell_bw.bias_ih') or name.endswith('.cell_bw.bias_hh'):
                continue
            ptname = name

            if name.endswith('._mean'):
                ptname = name.replace('._mean', '.running_mean')
            elif name.endswith('._variance'):
                ptname = name.replace('._variance', '.running_var')
            elif name.endswith('bias') or name.endswith('weight'):
                ptname = name
            elif 'lstm' in name:
                ptname = name
            elif 'attention_cell' in name:
                ptname = name

            # else:
            #     print('Redundance:')
            #     print(name)
            #     raise ValueError

            try:
                if ptname.endswith('fc.weight'):
                    self.net.state_dict()[ptname].copy_(torch.Tensor(para_state_dict[k].T))
                elif ptname.endswith('fc1.weight'): # for tps loc
                    self.net.state_dict()[ptname].copy_(torch.Tensor(para_state_dict[k].T))
                elif ptname.endswith('fc2.weight'): # for tps loc
                    self.net.state_dict()[ptname].copy_(torch.Tensor(para_state_dict[k].T))
                elif ptname.endswith('.weight') \
                    and len(para_state_dict[k].shape) == len(self.net.state_dict()[ptname].shape) == 2 \
                        and para_state_dict[k].shape[0] == self.net.state_dict()[ptname].shape[1] \
                            and para_state_dict[k].shape[1] == self.net.state_dict()[ptname].shape[0]: # for general fc
                    self.net.state_dict()[ptname].copy_(torch.Tensor(para_state_dict[k].T))

                else:
                    self.net.state_dict()[ptname].copy_(torch.Tensor(para_state_dict[k]))
            except Exception as e:
                print('pytorch: {}, {}'.format(ptname, self.net.state_dict()[ptname].size()))
                print('paddle: {}, {}'.format(k, para_state_dict[k].shape))
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

    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = SRConverter(cfg, paddle_pretrained_model_path)

    # save
    if args.dst_model_path is not None:
        save_name = args.dst_model_path
    else:
        save_name = '{}infer.pth'.format(os.path.basename(os.path.dirname(paddle_pretrained_model_path))[:-5])
    converter.save_pytorch_weights(save_name)
    print('done.')