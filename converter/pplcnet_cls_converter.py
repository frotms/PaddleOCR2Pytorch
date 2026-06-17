#!/usr/bin/env python3
"""
Convert PaddleOCR PP-LCNet textline_ori/doc_ori classification models to PyTorch.

Usage:
    # textline_ori (2-class, scale=0.25)
    python pplcnet_cls_converter.py \
        --yaml_path=configs/cls/textline_ori/PP-LCNet_x0_25_textline_ori.yml \
        --src_model_path=pretrained/PP-LCNet_x0_25_textline_ori_pretrained.pdparams

    # doc_ori (4-class, scale=1.0)
    python pplcnet_cls_converter.py \
        --yaml_path=configs/cls/doc_ori/PP-LCNet_x1_0_doc_ori.yml \
        --src_model_path=pretrained/PP-LCNet_x1_0_doc_ori_pretrained.pdparams
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20


class PPLCNetClsConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super().__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()

    def load_paddle_weights(self, weights_path):
        print('Loading PaddlePaddle weights from: {}'.format(weights_path))
        para_state_dict, _ = self.read_paddle_weights(weights_path)

        # Normalize Paddle keys: _mean → running_mean, _variance → running_var
        pp_dict = {}
        for k, v in para_state_dict.items():
            k_norm = k.replace('._mean', '.running_mean').replace('._variance', '.running_var')
            pp_dict[k_norm] = v

        py_state_dict = self.net.state_dict()
        loaded = 0
        missing = []

        for py_key, py_tensor in py_state_dict.items():
            if py_key.endswith('num_batches_tracked'):
                continue

            # Determine the flat Paddle key from the PyTorch key
            # PyTorch keys: backbone.xxx, backbone.blocksN.M.yyy, head.xxx
            # Paddle keys:  conv1, blocksN.M, fc, last_conv (no prefix)
            pp_match = self._find_paddle_key(py_key, pp_dict)

            if pp_match is not None:
                pp_key, pp_val = pp_match
                pp_np = pp_val.numpy() if hasattr(pp_val, 'numpy') else pp_val

                # Handle fc.weight transpose
                if pp_key.endswith('fc.weight') and py_key.endswith('head.fc.weight'):
                    # Paddle fc.weight: [1280, 2], PyTorch: [2, 1280]
                    py_state_dict[py_key].copy_(torch.Tensor(pp_np.T))
                    loaded += 1
                elif pp_np.shape == tuple(py_tensor.shape):
                    py_state_dict[py_key].copy_(torch.Tensor(pp_np))
                    loaded += 1
                else:
                    print(f'  shape mismatch: {py_key} {tuple(py_tensor.shape)} vs pp={pp_np.shape}')
            else:
                missing.append(py_key)

        print(f'Loaded {loaded}/{len(py_state_dict)} params')
        if missing:
            print(f'Missing {len(missing)} params:')
            for m in missing:
                print(f'  {m}')

    def _find_paddle_key(self, py_key, pp_dict):
        """Find matching Paddle key for a PyTorch key."""
        if py_key.startswith('backbone.'):
            flat = py_key[len('backbone.'):]
        elif py_key.startswith('head.'):
            flat = py_key[len('head.'):]
        else:
            flat = py_key

        # Stem: conv1
        if flat == 'conv1.weight':
            match_key = 'conv1.conv.weight'
        elif flat == 'conv1_bn.weight':
            match_key = 'conv1.bn.weight'
        elif flat == 'conv1_bn.bias':
            match_key = 'conv1.bn.bias'
        elif flat == 'conv1_bn.running_mean':
            match_key = 'conv1.bn.running_mean'
        elif flat == 'conv1_bn.running_var':
            match_key = 'conv1.bn.running_var'

        # Blocks: Paddle pattern = blocksN.M.dw_conv.{conv,bn}, blocksN.M.pw_conv.{conv,bn}
        # PyTorch pattern = blocksN.M.{dw_conv,dw_bn,dw_act},{pw_conv,pw_bn,pw_act},{se}
        elif '.dw_conv.weight' in flat:
            # PyTorch: blocks5.0.dw_conv.weight ← Paddle: blocks5.0.dw_conv.conv.weight
            match_key = flat.replace('.dw_conv.weight', '.dw_conv.conv.weight')
        elif '.dw_bn.weight' in flat:
            match_key = flat.replace('.dw_bn.weight', '.dw_conv.bn.weight')
        elif '.dw_bn.bias' in flat:
            match_key = flat.replace('.dw_bn.bias', '.dw_conv.bn.bias')
        elif '.dw_bn.running_mean' in flat:
            match_key = flat.replace('.dw_bn.running_mean', '.dw_conv.bn.running_mean')
        elif '.dw_bn.running_var' in flat:
            match_key = flat.replace('.dw_bn.running_var', '.dw_conv.bn.running_var')

        elif '.pw_conv.weight' in flat:
            match_key = flat.replace('.pw_conv.weight', '.pw_conv.conv.weight')
        elif '.pw_bn.weight' in flat:
            match_key = flat.replace('.pw_bn.weight', '.pw_conv.bn.weight')
        elif '.pw_bn.bias' in flat:
            match_key = flat.replace('.pw_bn.bias', '.pw_conv.bn.bias')
        elif '.pw_bn.running_mean' in flat:
            match_key = flat.replace('.pw_bn.running_mean', '.pw_conv.bn.running_mean')
        elif '.pw_bn.running_var' in flat:
            match_key = flat.replace('.pw_bn.running_var', '.pw_conv.bn.running_var')

        # SE block (no rename)
        elif '.se.conv1' in flat or '.se.conv2' in flat:
            match_key = flat

        # last_conv
        elif flat == 'last_conv.weight':
            match_key = 'last_conv.weight'

        # Head
        elif flat == 'fc.weight':
            match_key = 'fc.weight'
        elif flat == 'fc.bias':
            match_key = 'fc.bias'
        else:
            match_key = flat

        if match_key in pp_dict:
            return (match_key, pp_dict[match_key])
        return None


if __name__ == '__main__':
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--src_model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    arch_cfg = cfg['Architecture']
    paddle_path = args.src_model_path

    converter = PPLCNetClsConverter(arch_cfg, paddle_path)

    if args.output_path:
        output_path = args.output_path
    else:
        model_name = cfg.get('Global', {}).get('model_name', 'pplcnet_cls')
        output_path = f'{model_name}_infer.pth'

    converter.save_pytorch_weights(output_path)
    print(f'Model saved to: {output_path}')
    print('Done.')
