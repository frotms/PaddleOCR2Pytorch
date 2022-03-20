# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import copy
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class PPStructureTableStructureConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        print('config: ', config)
        print(type(kwargs), kwargs)
        super(PPStructureTableStructureConverter, self).__init__(config, **kwargs)
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

            if k.startswith('backbone.conv'):
                pass

            elif k.startswith('backbone.stages.'):
                ppname = ppname.replace('backbone.stages.', 'backbone.stage')

            elif k.startswith('head.'):
                pass

            else:
                print('Redundance:')
                print(k, ' ---- ', ppname)
                raise ValueError

            try:
                if ppname.endswith('.weight') \
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
        # cfg = {'model_type':'rec',
        #        'algorithm':'CRNN',
        #        'Transform':None,
        #        'Backbone':{'model_name':'small', 'name':'MobileNetV3', 'scale':0.5, 'small_stride':[1,2,2,2]},
        #        'Neck':{'name':'SequenceEncoder', 'hidden_size':48, 'encoder_type':'rnn'},
        #        'Head':{'name':'CTCHead', 'fc_decay': 4e-05},
        #        }
    kwargs = {}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = PPStructureTableStructureConverter(cfg, paddle_pretrained_model_path, **kwargs)

    np.random.seed(666)
    inp = torch.from_numpy(np.random.randn(1, 3, 488, 488).astype(np.float32))
    with torch.no_grad():
        out = converter.net(inp)
    # print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    if args.dst_model_path is not None:
        save_name = args.dst_model_path
    else:
        save_name = '{}infer.pth'.format(os.path.basename(os.path.dirname(paddle_pretrained_model_path))[:-5])
    converter.save_pytorch_weights(save_name)
    print('done.')