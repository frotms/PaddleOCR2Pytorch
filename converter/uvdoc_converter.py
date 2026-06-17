#!/usr/bin/env python3
"""
Convert PaddleOCR UVDoc document unwarping model to PyTorch.

Usage:
    python uvdoc_converter.py \
        --src_model_path=pretrained/UVDoc_pretrained.pdparams \
        --output_path=pretrained/UVDoc_infer.pth
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import torch
import types

# Workaround for Paddle 3.0 + numpy 1.x compatibility
if 'numpy._core' not in sys.modules:
    _core = types.ModuleType('numpy._core')
    _core.multiarray = np.core.multiarray
    _core._multiarray_umath = np.core._multiarray_umath
    _core.numerictypes = np.core.numerictypes
    _core.umath = np.core.umath
    _core._methods = np.core._methods
    _core.fromnumeric = np.core.fromnumeric
    _core.shape_base = np.core.shape_base
    _core.records = np.core.records
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    sys.modules['numpy._core.numerictypes'] = np.core.numerictypes
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._methods'] = np.core._methods
    sys.modules['numpy._core.fromnumeric'] = np.core.fromnumeric

from pytorchocr.modeling.architectures.uvdoc_model import UVDocModel


def normalize_paddle_key(k):
    """Normalize PaddlePaddle BN keys to PyTorch format."""
    k = k.replace('._mean', '.running_mean')
    k = k.replace('._variance', '.running_var')
    return k


def key_match_score(paddle_key, pytorch_key):
    """Compute a similarity score between Paddle and PyTorch keys."""
    pp_parts = paddle_key.split('.')
    py_parts = pytorch_key.split('.')

    if pp_parts[-1] != py_parts[-1]:
        return 0

    # Count matching suffix parts
    matches = 0
    i, j = len(pp_parts) - 1, len(py_parts) - 1
    while i >= 0 and j >= 0 and pp_parts[i] == py_parts[j]:
        matches += 1
        i -= 1
        j -= 1

    # Bonus for same total length (penalize partial matches)
    if len(pp_parts) == len(py_parts):
        matches += 100  # Exact length match bonus

    return matches


def convert_weights(paddle_state, pytorch_model):
    """
    Convert PaddlePaddle UVDoc weights to PyTorch using best-match strategy.
    """
    # Build normalized Paddle weight dict: norm_key -> (original_key, value, shape)
    pp_norm = OrderedDict()
    for k, v in paddle_state.items():
        nk = normalize_paddle_key(k)
        pp_norm[nk] = (k, v, tuple(v.shape))

    py_state = pytorch_model.state_dict()
    py_keys = list(py_state.keys())
    pp_used = set()

    loaded = 0
    missing = []

    for py_key in py_keys:
        if py_key.endswith('num_batches_tracked'):
            continue

        py_shape = tuple(py_state[py_key].shape)
        best_match = None
        best_score = -1

        # Find best matching Paddle key by score and shape
        for pp_nk, (pp_orig, pp_val, pp_shape) in pp_norm.items():
            if pp_nk in pp_used:
                continue
            if pp_shape != py_shape:
                continue
            score = key_match_score(pp_nk, py_key)
            if score > best_score:
                best_score = score
                best_match = (pp_nk, pp_val)

        if best_match and best_score > 0:
            pp_nk, pp_val = best_match
            pp_np = pp_val.numpy()
            py_state[py_key].copy_(torch.Tensor(pp_np))
            pp_used.add(pp_nk)
            loaded += 1
        else:
            # Fallback: try exact key lookup (for hard-to-match keys)
            if py_key in pp_norm:
                _pp_orig, pp_val, pp_shape = pp_norm[py_key]
                pp_np = pp_val.numpy()
                py_state[py_key].copy_(torch.Tensor(pp_np))
                pp_used.add(py_key)
                loaded += 1
            else:
                missing.append((py_key, py_shape))

    print(f'Loaded {loaded} params')
    if missing:
        print(f'Missing {len(missing)} PyTorch params')
        for m, s in missing[:10]:
            print(f'  {m}: {s}')
        if len(missing) > 10:
            print(f'  ... and {len(missing)-10} more')

    # Report Paddle params that weren't used
    unused = []
    for pp_nk, (pp_orig, pp_val, pp_shape) in pp_norm.items():
        if pp_nk not in pp_used:
            unused.append((pp_nk, pp_shape))
    if unused:
        print(f'\nUnused Paddle params ({len(unused)}):')
        for u, s in unused[:10]:
            print(f'  {u}: {s}')
        if len(unused) > 10:
            print(f'  ... and {len(unused)-10} more')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default='pretrained/UVDoc_infer.pth')
    args = parser.parse_args()

    import paddle
    print('Loading PaddlePaddle weights...')
    pp_state = paddle.load(args.src_model_path)
    print(f'PaddlePaddle weights: {len(pp_state)} params')

    print('Building PyTorch UVDoc model...')
    model = UVDocModel()
    model.eval()

    print('Converting weights...')
    convert_weights(pp_state, model)

    print(f'Saving to: {args.output_path}')
    torch.save(model.state_dict(), args.output_path, _use_new_zipfile_serialization=False)
    print('Done.')
