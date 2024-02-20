# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class PPOCRv4RecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        para_state_dict = self.del_invalid_state_dict(para_state_dict)
        out_channels = list(para_state_dict.values())[-1].shape[0]
        print('out_channels: ', out_channels)
        print(type(kwargs), kwargs)
        kwargs['out_channels'] = out_channels
        super(PPOCRv4RecConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()


    def del_invalid_state_dict(self, para_state_dict):
        new_state_dict = OrderedDict()
        for i, (k,v) in enumerate(para_state_dict.items()):
            if k.startswith('head.gtc_head.'):
                continue

            elif k.startswith('head.before_gtc'):
                continue

            else:
                new_state_dict[k] = v
        return new_state_dict


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights

        for k,v in para_state_dict.items():
            ptname = k
            ptname = ptname.replace('._mean', '.running_mean')
            ptname = ptname.replace('._variance','.running_var')

            try:
                if k.endswith('fc1.weight') or k.endswith('fc2.weight') \
                        or k.endswith('fc.weight') or k.endswith('qkv.weight') \
                        or k.endswith('proj.weight'):
                    self.net.state_dict()[ptname].copy_(torch.Tensor(v.T))
                else:
                    self.net.state_dict()[ptname].copy_(torch.Tensor(v))

            except Exception as e:
                print('exception:')
                print('pytorch: {}, {}'.format(ptname, self.net.state_dict()[v].size()))
                print('paddle: {}, {}'.format(k, v.shape))
                raise e

        print('model is loaded.')

def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))
    if res['Architecture']['Head']['name'] == 'MultiHead':
        char_dict_path = os.path.abspath(res['Global']['character_dict_path'])
        if not os.path.exists(char_dict_path):
            raise FileNotFoundError('{} is not existed.'.format(char_dict_path))
        character_str = []
        with open(char_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                character_str.append(line)
        use_space_char = res['Global']['use_space_char']
        if use_space_char:
            character_str.append(" ")
        character_str = ['blank'] + character_str
        char_num = len(character_str)
        res['Architecture']['Head']['out_channels_list'] = {
            'CTCLabelDecode': char_num,
            'SARLabelDecode': char_num + 2,
            'NRTRLabelDecode': char_num + 3
        }
    return res['Architecture']

if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help='Assign the yaml path of network configuration', default=None)
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        cfg = read_network_config_from_yaml(yaml_path)

    else:
        raise NotImplementedError

    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    if not os.path.exists(paddle_pretrained_model_path + '.pdparams'):
        paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'student')
    converter = PPOCRv4RecConverter(cfg, paddle_pretrained_model_path)

    np.random.seed(666)
    inputs = np.random.randn(1,3,48,320).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    out = out.data.numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    save_name = 'ch_ptocr_v4_rec_server_infer.pth'
    converter.save_pytorch_weights(save_name)
    print('done.')
