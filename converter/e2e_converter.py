import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class E2EV20DetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(E2EV20DetConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()

    def load_paddle_weights(self, weights_path):
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)

        for k, v in para_state_dict.items():
            if 'stage' in k:
                stages_replace_stage_flag = True
                break

        for k,v in self.net.state_dict().items():
            keyword = 'stages.'
            if keyword in k:
                # replace: 'stages.{}.' -> ''
                start_id = k.find(keyword)
                end_id = start_id + len(keyword) + 1 + 1
                name = k.replace(k[start_id:end_id], '')
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
            else:
                print('Redundance:')
                print(name)
                raise ValueError


            self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))

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

    # cfg = {'model_type':'e2e',
    #        'algorithm':'PGNet',
    #        'Transform':None,
    #        'Backbone':{'name':'ResNet', 'layers':50},
    #        'Neck':{'name':'PGFPN'},
    #        'Head':{'name':'PGHead'}}

    kwargs = {}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = E2EV20DetConverter(cfg, paddle_pretrained_model_path, **kwargs)
    print('todo')

    # image = cv2.imread('doc/imgs_en/img_10.jpg')
    # image = cv2.resize(image, (320, 448))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # scale = 1. / 255
    # norm_img = (image * scale - mean) / std
    # transpose_img = norm_img.transpose(2, 0, 1)
    # transpose_img = np.expand_dims(transpose_img, 0).astype(np.float32)
    # with torch.no_grad():
    #     inp = torch.Tensor(transpose_img)
    #     out = converter.net(inp)
    # out = out['maps'].data.numpy()
    # print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    if args.dst_model_path is not None:
        save_name = args.dst_model_path
    else:
        save_name = '{}_infer.pth'.format(os.path.basename(os.path.dirname(paddle_pretrained_model_path)))
    converter.save_pytorch_weights(save_name)
    print('done.')