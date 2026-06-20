#!/usr/bin/env python
"""
PP-DocLayout Layout Detection Model Converter (PaddlePaddle -> PyTorch)

Converts PaddleOCR PP-DocLayout series models to PyTorch format.

Paddle key naming -> PyTorch key naming mapping:
  - BN._mean / ._variance  ->  BN.running_mean / .running_var
  - head.distribution_project.project  ->  proj
  - Most other keys are identical

Usage:
    python converter/ppstructure_layout_converter.py \
        --src_model_path=./models/structurev3/PP-DocLayout-S_pretrained.pdparams \
        --dst_model_path=./models/structurev3/ptocr_ppdoclayout_s.pth \
        --variant=S
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptstructure.layout.picodet import PPDocLayout, _VARIANTS


def _paddle_to_torch(paddle_val):
    """Convert paddle.Tensor or numpy array to torch.Tensor."""
    if hasattr(paddle_val, 'numpy'):
        return torch.from_numpy(paddle_val.numpy())
    elif isinstance(paddle_val, np.ndarray):
        return torch.from_numpy(paddle_val)
    else:
        return paddle_val


def _map_key_pp2pt(pp_key):
    """Map a PaddlePaddle state dict key to PyTorch equivalent."""
    pt_key = pp_key

    # BN: _mean -> running_mean, _variance -> running_var
    pt_key = pt_key.replace('._mean', '.running_mean')
    pt_key = pt_key.replace('._variance', '.running_var')

    # Distribution project
    pt_key = pt_key.replace('head.distribution_project.project', 'proj')

    return pt_key


class PPDocLayoutConverter:
    """Convert PaddlePaddle PP-DocLayout weights to PyTorch PPDocLayout."""

    def __init__(self, variant='S'):
        self.variant = variant
        self._build_model()

    def _build_model(self):
        self.model = PPDocLayout(variant=self.variant)
        self.model.eval()

    def convert(self, paddle_model_path):
        """Load PaddlePaddle weights and map to PyTorch model.

        Uses a per-key mapping approach: iterate over PyTorch keys and find
        corresponding PaddlePaddle keys.
        """
        import paddle

        if not os.path.exists(paddle_model_path):
            raise FileNotFoundError(f"Paddle model not found: {paddle_model_path}")

        pp_state = paddle.load(paddle_model_path)
        pt_state = self.model.state_dict()

        print(f'Paddle keys: {len(pp_state)}, PyTorch keys: {len(pt_state)}')

        # Build reverse mapping: pt_key -> pp_key
        loaded = 0
        not_found = []
        shape_mismatch = []

        # Keys to skip (use PyTorch defaults instead of Paddle values):
        # - proj: DFL uses exact integers [0, 1, ..., reg_max], not trained values
        # - scale_reg: dummy modules not used in forward
        _SKIP_KEYS = {'proj', 'head.p3_feat.scale_reg', 'head.p4_feat.scale_reg',
                      'head.p5_feat.scale_reg', 'head.p6_feat.scale_reg'}

        for pt_key, pt_tensor in pt_state.items():
            if pt_key.endswith('num_batches_tracked'):
                loaded += 1
                continue

            if pt_key in _SKIP_KEYS:
                loaded += 1  # Keep PyTorch default (exact integers)
                continue

            # Try direct mapping: pt_key -> pp_key via BN renaming
            pp_key = _map_pt2pp_direct(pt_key)

            if pp_key in pp_state:
                pp_val = _paddle_to_torch(pp_state[pp_key])
                if tuple(pt_tensor.shape) == tuple(pp_val.shape):
                    pt_state[pt_key].copy_(pp_val)
                    loaded += 1
                else:
                    shape_mismatch.append((pt_key, list(pt_tensor.shape), list(pp_val.shape)))
            else:
                not_found.append(pt_key)

        print(f'Loaded: {loaded}, Shape mismatch: {len(shape_mismatch)}, Not found: {len(not_found)}')

        if shape_mismatch:
            print('\nShape mismatches:')
            for pt_key, pt_shape, pp_shape in shape_mismatch[:20]:
                print(f'  {pt_key}: PT={pt_shape} PP={pp_shape}')

        if not_found:
            print('\nNot found keys (sample):')
            for k in not_found[:15]:
                print(f'  PT: {k}')

        self._load_stats = {'loaded': loaded, 'shape_mismatch': len(shape_mismatch),
                            'not_found': len(not_found)}
        return self._load_stats

    def save(self, output_path):
        """Save PyTorch model weights (clean, without num_batches_tracked)."""
        clean_state = {k: v for k, v in self.model.state_dict().items()
                       if 'num_batches_tracked' not in k}
        torch.save(clean_state, output_path)
        print(f'Model saved to {output_path}')

    def verify(self, input_size=640):
        """Verify model output with random input."""
        np.random.seed(666)
        inp = torch.from_numpy(np.random.randn(1, 3, input_size, input_size).astype(np.float32))
        with torch.no_grad():
            cls_all, boxes_all, strides = self.model(inp)
        print(f'\nVerification ({input_size}x{input_size} random input):')
        print(f'  cls_all: {cls_all.shape}, sum={cls_all.sum().item():.6f}, '
              f'mean={cls_all.mean().item():.6f}')
        print(f'  boxes_all: {boxes_all.shape}, sum={boxes_all.sum().item():.6f}')
        print(f'  strides: {strides}')
        return cls_all, boxes_all


def _map_pt2pp_direct(pt_key):
    """Map PyTorch key to PaddlePaddle key by reversing BN renaming rules."""
    pp_key = pt_key

    # BN: running_mean -> _mean, running_var -> _variance
    pp_key = pp_key.replace('.running_mean', '._mean')
    pp_key = pp_key.replace('.running_var', '._variance')

    # proj buffer -> distribution_project.project
    if pp_key == 'proj':
        pp_key = 'head.distribution_project.project'

    return pp_key


def main():
    parser = argparse.ArgumentParser(description='Convert PP-DocLayout model to PyTorch')
    parser.add_argument('--src_model_path', type=str, required=True,
                        help='Path to PaddlePaddle .pdparams weights.')
    parser.add_argument('--dst_model_path', type=str, default=None,
                        help='Output path for PyTorch .pth weights.')
    parser.add_argument('--variant', type=str, default='S', choices=['S', 'M', 'L'],
                        help='Model variant (S/M/L). Default: S')
    args = parser.parse_args()

    converter = PPDocLayoutConverter(variant=args.variant)
    converter.convert(args.src_model_path)
    converter.verify()

    if args.dst_model_path:
        converter.save(args.dst_model_path)
    else:
        base = os.path.splitext(os.path.basename(args.src_model_path))[0]
        save_name = f'ptocr_{base}.pth'
        converter.save(save_name)

    print('\nConversion completed.')


if __name__ == '__main__':
    main()
