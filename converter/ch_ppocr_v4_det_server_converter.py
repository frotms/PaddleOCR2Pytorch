import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20


class PPOCRv4DetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(PPOCRv4DetConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()

    def load_paddle_weights(self, weights_path):
        print('paddle weights loading...')
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)

        for k,v in para_state_dict.items():

            if k.endswith('num_batches_tracked'):
                continue

            ptname = k
            ptname = ptname.replace('._mean', '.running_mean')
            ptname = ptname.replace('._variance', '.running_var')

            if 'backbone.last_conv.weight' in k:
                continue

            if 'backbone.fc.weight' in k:
                continue

            if 'backbone.fc.bias' in k:
                continue

            try:
                self.net.state_dict()[ptname].copy_(torch.Tensor(v))
            except Exception as e:
                print('pytorch: {}, {}'.format(ptname, self.net.state_dict()[ptname].size()))
                print('paddle: {}, {}'.format(k, v.shape))
                raise e
        print('model is loaded: {}'.format(weights_path))

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
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        cfg = read_network_config_from_yaml(yaml_path)
    else:
        raise NotImplementedError

    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    converter = PPOCRv4DetConverter(cfg, paddle_pretrained_model_path)

    print('todo')

    np.random.seed(666)
    inputs = np.random.randn(1, 3, 640, 640).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    out = out['maps'].data.numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    save_basename = os.path.basename(os.path.abspath(args.src_model_path))

    save_name = 'ch_ptocr_v4_det_server_infer.pth'
    converter.save_pytorch_weights(save_name)

    print('done.')