#!/usr/bin/env python
"""
PP-FormulaNet Weight Converter: Paddle → PyTorch

Converts PP-FormulaNet model weights from PaddlePaddle (.pdparams) format
to PyTorch (.pth) format.

Usage:
    # Convert PP-FormulaNet-S
    python converter/ppstructure_formula_converter.py \\
        --src_model_path=PP-FormulaNet-S_pretrained.pdparams \\
        --dst_model_path=ptocr_formulanet_s.pth --variant=S

    # Convert PP-FormulaNet_plus-M
    python converter/ppstructure_formula_converter.py \\
        --src_model_path=PP-FormulaNet_plus-M_pretrained.pdparams \\
        --dst_model_path=ptocr_formulanet_m.pth --variant=M

Architecture:
    Backbone: PPHGNetV2_B4_Formula (S) or PPHGNetV2_B6_Formula (M)
    Head: PPFormulaNet_Head (MBart-style decoder)
"""

import os, sys, argparse, logging
import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_paddle_weights(path):
    """Load PaddlePaddle .pdparams weights file."""
    import paddle
    return paddle.load(path)


def build_torch_model(variant='M'):
    """Build PyTorch PP-FormulaNet model.

    Args:
        variant: 'S' or 'M'.

    Returns:
        PyTorch nn.Module.
    """
    from pytorchocr.modeling.backbones.rec_pphgnetv2 import (
        PPHGNetV2_B4_Formula,
        PPHGNetV2_B6_Formula,
    )
    from pytorchocr.modeling.heads.rec_ppformulanet_head import PPFormulaNet_Head

    if variant == 'S':
        backbone = PPHGNetV2_B4_Formula()
        head = PPFormulaNet_Head(
            max_new_tokens=1024,
            decoder_start_token_id=0,
            encoder_hidden_size=2048,
            decoder_hidden_size=384,
            decoder_ffn_dim=1536,
            decoder_layers=2,
        )
    else:  # M (plus-M)
        backbone = PPHGNetV2_B6_Formula()
        head = PPFormulaNet_Head(
            max_new_tokens=2560,
            decoder_start_token_id=0,
            encoder_hidden_size=2048,
            decoder_hidden_size=512,
            decoder_ffn_dim=2048,
            decoder_layers=6,
        )

    from ptstructure.formula.ppformulanet import PPFormulaNet
    model = PPFormulaNet(backbone, head)
    return model


def map_weights(paddle_weights, torch_model, variant='M'):
    """Map Paddle weights to PyTorch model.

    Args:
        paddle_weights: Dict of Paddle parameter names → values.
        torch_model: PyTorch model.
        variant: 'S' or 'M'.

    Returns:
        State dict for PyTorch model.
    """
    torch_sd = torch_model.state_dict()
    converted = {}
    unmatched_paddle = list(paddle_weights.keys())
    stats = {'mapped': 0, 'skipped': 0}

    # Direct mapping: same name structure
    # Paddle prefix: none (bare keys like "backbone.xxx" or "head.xxx")
    # Torch prefix: "backbone." and "head."

    for torch_key in torch_sd.keys():
        # Try to find matching Paddle key
        paddle_key = None

        # Backbone keys
        if torch_key.startswith('backbone.'):
            # Strip "backbone." prefix for matching
            subkey = torch_key[len('backbone.'):]
            if subkey in paddle_weights:
                paddle_key = subkey

        # Head keys
        elif torch_key.startswith('head.'):
            subkey = torch_key[len('head.'):]
            if subkey in paddle_weights:
                paddle_key = subkey

        if paddle_key is not None:
            paddle_val = paddle_weights[paddle_key]
            torch_val = torch_sd[torch_key]

            # Convert numpy to tensor
            paddle_tensor = torch.from_numpy(paddle_val)

            # Handle transpositions for Conv2D and Linear layers
            # Paddle Conv2D: [out_ch, in_ch, h, w] → same in PyTorch
            # Paddle Linear: [out_features, in_features] → same in PyTorch
            # But Paddle stores Linear weights transposed compared to PyTorch
            is_linear_weight = (
                paddle_tensor.dim() == 2
                and torch_val.dim() == 2
                and paddle_tensor.shape == torch_val.shape[::-1]
            )
            if is_linear_weight:
                paddle_tensor = paddle_tensor.t()

            # Handle shape mismatch
            if paddle_tensor.shape != torch_val.shape:
                logger.debug(f'  Shape mismatch: {torch_key} {torch_val.shape} vs {paddle_tensor.shape}')
                if paddle_tensor.numel() == torch_val.numel():
                    paddle_tensor = paddle_tensor.reshape(torch_val.shape)
                else:
                    stats['skipped'] += 1
                    continue

            converted[torch_key] = paddle_tensor
            unmatched_paddle.remove(paddle_key)
            stats['mapped'] += 1

    # Fill remaining keys with original values
    for key in torch_sd.keys():
        if key not in converted:
            converted[key] = torch_sd[key]  # Keep initialized value

    logger.info(f'Weight mapping: {stats["mapped"]} mapped, {stats["skipped"]} skipped')
    if unmatched_paddle:
        logger.info(f'Unmatched Paddle keys: {len(unmatched_paddle)}')
        for k in unmatched_paddle[:10]:
            logger.debug(f'  {k}')

    return converted


def main():
    parser = argparse.ArgumentParser(description='Convert PP-FormulaNet weights Paddle→PyTorch')
    parser.add_argument('--src_model_path', type=str, required=True,
                        help='Path to Paddle .pdparams file.')
    parser.add_argument('--dst_model_path', type=str, required=True,
                        help='Path to output PyTorch .pth file.')
    parser.add_argument('--variant', type=str, default='M', choices=['S', 'M'],
                        help='Model variant (S or M).')
    args = parser.parse_args()

    logger.info(f'Converting PP-FormulaNet ({args.variant})...')
    logger.info(f'  Source: {args.src_model_path}')
    logger.info(f'  Target: {args.dst_model_path}')

    # Load Paddle weights
    paddle_weights = load_paddle_weights(args.src_model_path)
    logger.info(f'  Paddle params: {len(paddle_weights)} keys')

    # Build PyTorch model
    torch_model = build_torch_model(args.variant)
    torch_model.eval()

    # Map weights
    state_dict = map_weights(paddle_weights, torch_model, args.variant)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.dst_model_path)), exist_ok=True)
    torch.save(state_dict, args.dst_model_path)
    logger.info(f'  Saved to: {args.dst_model_path}')
    logger.info('Done!')


if __name__ == '__main__':
    main()
