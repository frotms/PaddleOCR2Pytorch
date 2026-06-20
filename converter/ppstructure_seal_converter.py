#!/usr/bin/env python
"""
Seal Detection Weight Converter: Paddle → PyTorch

Converts PP-OCRv4 Seal Detection model weights from PaddlePaddle (.pdparams)
to PyTorch (.pth) format.

Usage:
    python converter/ppstructure_seal_converter.py \
        --src_model_path=PP-OCRv4_mobile_seal_det_pretrained.pdparams \
        --dst_model_path=ptocr_seal_det.pth
"""

import os, sys, argparse, logging
from collections import OrderedDict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        res = res
    config = res.get('Architecture', res)
    return config


class SealDetConverter:
    def __init__(self, config, **kwargs):
        from pytorchocr.modeling.architectures.base_model import BaseModel
        self.net = BaseModel(config, **kwargs)

    def load_paddle_weights(self, weights_path):
        logger.info('Loading Paddle weights from %s', weights_path)
        try:
            import paddle.fluid as fluid
            with fluid.dygraph.guard():
                para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        except Exception:
            import paddle
            import types
            # Workaround for Paddle 3.0 pickle format
            if 'numpy._core' not in sys.modules:
                _core = types.ModuleType('numpy._core')
                _core.multiarray = np.core.multiarray
                _core._multiarray_umath = np.core._multiarray_umath
                _core.numerictypes = np.core.numerictypes
                _core.umath = np.core.umath
                _core._methods = np.core._methods
                _core.fromnumeric = np.core.fromnumeric
                sys.modules['numpy._core'] = _core
                sys.modules['numpy._core.multiarray'] = np.core.multiarray
                sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
                sys.modules['numpy._core.numerictypes'] = np.core.numerictypes
                sys.modules['numpy._core.umath'] = np.core.umath
                sys.modules['numpy._core._methods'] = np.core._methods
                sys.modules['numpy._core.fromnumeric'] = np.core.fromnumeric
            para_state_dict = paddle.load(weights_path)

        para_state_dict = self._del_invalid_state_dict(para_state_dict)

        loaded = 0
        for k, v in para_state_dict.items():
            if k.endswith('num_batches_tracked'):
                continue
            ptname = k
            ptname = ptname.replace('._mean', '.running_mean')
            ptname = ptname.replace('._variance', '.running_var')
            try:
                target = self.net.state_dict().get(ptname)
                if target is None:
                    logger.debug('  Skip (not in model): %s', k)
                    continue
                self.net.state_dict()[ptname].copy_(torch.from_numpy(v))
                loaded += 1
            except Exception as e:
                logger.debug('  Convert error: %s | torch=%s paddle=%s',
                           k, target.shape if target is not None else 'None', v.shape)
        logger.info('  Loaded %d parameters', loaded)

    @staticmethod
    def _del_invalid_state_dict(para_state_dict):
        new_state_dict = OrderedDict()
        for k, v in para_state_dict.items():
            if 'aux_binarize_' in k or 'aux_thresh_' in k:
                continue
            new_state_dict[k] = v
        return new_state_dict

    def save(self, path):
        torch.save(self.net.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(description='Convert Seal Detection weights Paddle→PyTorch')
    parser.add_argument('--src_model_path', type=str, required=True,
                        help='Path to Paddle .pdparams file.')
    parser.add_argument('--dst_model_path', type=str, required=True,
                        help='Path to output PyTorch .pth file.')
    parser.add_argument('--yaml_path', type=str,
                        default='configs/det/PP-OCRv4/PP-OCRv4_mobile_seal_det.yml',
                        help='Path to seal detection YAML config.')
    args = parser.parse_args()

    logger.info('Converting PP-OCRv4 Seal Detection...')
    logger.info('  Source: %s', args.src_model_path)
    logger.info('  Config: %s', args.yaml_path)
    logger.info('  Target: %s', args.dst_model_path)

    config = read_network_config_from_yaml(args.yaml_path)
    converter = SealDetConverter(config)
    converter.load_paddle_weights(args.src_model_path)
    converter.save(args.dst_model_path)
    logger.info('  Saved to: %s', args.dst_model_path)
    logger.info('Done!')


if __name__ == '__main__':
    main()
