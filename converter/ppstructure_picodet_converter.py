#!/usr/bin/env python
"""
PP-DocLayout Converter: Paddle .pdparams → PyTorch .pth
Converts PP-DocLayout-M pretrained weights.

Usage:
    python converter/ppstructure_picodet_converter.py \
        --src_model_path=PP-DocLayout-M_pretrained.pdparams \
        --dst_model_path=ptocr_ppdoclayout_m.pth
"""

import os, sys, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from ptstructure.layout.picodet import PPDocLayout


def convert(paddle_path, output_path):
    """Convert Paddle .pdparams to PyTorch .pth."""
    import paddle
    pp = paddle.load(paddle_path)

    model = PPDocLayout()
    pt_dict = model.state_dict()

    skip_patterns = [
        'head.p3_feat.scale_reg', 'head.p4_feat.scale_reg',
        'head.p5_feat.scale_reg', 'head.p6_feat.scale_reg',
        'head.distribution_project.project',
    ]
    head_direct = re.compile(r'head\.head_cls\d+\b')
    head_reg_direct = re.compile(r'head\.head_reg\d+\b')

    loaded = 0
    skipped = 0
    for pp_key, pp_val in pp.items():
        pt_key = pp_key.replace('._mean', '.running_mean').replace('._variance', '.running_var')

        should_skip = False
        for sp in skip_patterns:
            if pt_key.startswith(sp):
                should_skip = True
                break
        if head_direct.match(pt_key) or head_reg_direct.match(pt_key):
            should_skip = True
        if should_skip:
            skipped += 1
            continue

        if pt_key not in pt_dict:
            skipped += 1
            continue

        pp_arr = pp_val.numpy() if hasattr(pp_val, 'numpy') else np.array(pp_val)
        if list(pp_arr.shape) == list(pt_dict[pt_key].shape):
            pt_dict[pt_key].copy_(torch.from_numpy(pp_arr.copy()))
            loaded += 1

    n_pt = len([k for k in pt_dict if 'num_batches_tracked' not in k])
    print(f'Converted: {loaded}/{n_pt} weights ({skipped} skipped, {len(pp)} total)')

    model.eval()
    torch.save(model.state_dict(), output_path)
    print(f'Saved: {output_path}')

    # Verify forward
    np.random.seed(666)
    x = torch.from_numpy(np.random.randn(1, 3, 640, 640).astype(np.float32))
    with torch.no_grad():
        cls_all, boxes_all, strides = model(x)
    print(f'Forward OK: cls={list(cls_all.shape)}, boxes={list(boxes_all.shape)}, strides={strides}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PP-DocLayout-M: Paddle → PyTorch')
    parser.add_argument('--src_model_path', type=str, required=True,
                        help='Path to PP-DocLayout-M_pretrained.pdparams')
    parser.add_argument('--dst_model_path', type=str, required=True,
                        help='Output .pth path')
    args = parser.parse_args()
    convert(args.src_model_path, args.dst_model_path)
