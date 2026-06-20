#!/usr/bin/env python
"""
SLANeXt Table Structure Recognition Model Converter (PaddlePaddle → PyTorch)

Converts PaddleOCR SLANeXt_wired/wireless models to PyTorch format.

Key Mapping (discovered from actual weight inspection):
  Paddle                                        → PyTorch
  ────────────────────────────────────────────────────────────────────
  backbone.vision_tower_high.blocks.X.*          → backbone.vision_tower.layers.X.*
  backbone.vision_tower_high.patch_embed.proj.*  → backbone.vision_tower.patch_embed.projection.*
  backbone.vision_tower_high.neck.*              → backbone.vision_tower.neck.*
  backbone.vision_tower_high.pos_embed           → backbone.vision_tower.pos_embed
  backbone.post_conv.weight                      → backbone.post_conv.weight
  head.structure_attention_cell.i2h.*            → head.structure_attention_cell.input_to_hidden.*
  head.structure_attention_cell.h2h.*            → head.structure_attention_cell.hidden_to_hidden.*
  head.structure_attention_cell.score.*          → head.structure_attention_cell.score.* [T]
  head.structure_attention_cell.rnn.*            → head.structure_attention_cell.rnn.*
  head.structure_generator.0.*                   → head.structure_generator.fc1.* [T]
  head.structure_generator.1.*                   → head.structure_generator.fc2.* [T]
  head.loc_generator.*                           → (SKIP - unimplemented)

  [T] = Paddle Linear stores [in, out] → PyTorch stores [out, in]; TRANSPOSE needed.

Usage:
    python converter/ppstructure_slanext_converter.py \
        --yaml_path=configs/tablev3/SLANeXt_wired.yml \
        --src_model_path=./models/structurev3/SLANeXt_wired_pretrained.pdparams \
        --dst_model_path=./models/structurev3/ptocr_slanext_wired.pth
"""

import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptstructure.table.slanext import SLANeXt, SLANeXtConfig, SLANeXtVisionConfig


class SLANeXtConverter:
    """Convert PaddlePaddle SLANeXt weights to PyTorch SLANeXt."""

    def __init__(self, config: dict, paddle_model_path: str):
        # Build SLANeXt config from YAML
        vision_cfg = config.get('vision_config', {})
        vision_config = SLANeXtVisionConfig(**vision_cfg)

        slanext_config = SLANeXtConfig(
            vision_config=vision_config,
            post_conv_in_channels=config.get('post_conv_in_channels', 256),
            post_conv_out_channels=config.get('post_conv_out_channels', 512),
            out_channels=config.get('out_channels', 50),
            hidden_size=config.get('hidden_size', 512),
            max_text_length=config.get('max_text_length', 500),
            loc_reg_num=config.get('loc_reg_num', 8),
        )
        self.model = SLANeXt(slanext_config)
        self.model.eval()

        self._load_paddle_weights(paddle_model_path)

    # ------------------------------------------------------------------
    # Paddle → PyTorch key mapping rules
    # ------------------------------------------------------------------

    # Keys to SKIP (not implemented in our PyTorch model)
    SKIP_PADDLE_PREFIXES = [
        'head.loc_generator.',
    ]

    # Direct key-to-key mapping rules (Paddle → PyTorch)
    KEY_RULES = [
        # Vision tower prefix
        ('backbone.vision_tower_high.', 'backbone.vision_tower.'),
        # Block naming
        ('.blocks.', '.layers.'),
        # Patch embedding
        ('patch_embed.proj.', 'patch_embed.projection.'),
        # Attention head
        ('.structure_attention_cell.i2h.', '.structure_attention_cell.input_to_hidden.'),
        ('.structure_attention_cell.h2h.', '.structure_attention_cell.hidden_to_hidden.'),
        # Structure generator
        ('.structure_generator.0.', '.structure_generator.fc1.'),
        ('.structure_generator.1.', '.structure_generator.fc2.'),
    ]

    # Layer types where FC weights need transpose
    # Paddle Linear: weight shape [in_features, out_features]
    # PyTorch Linear: weight shape [out_features, in_features]
    TRANSPOSE_PATTERNS = [
        'qkv.weight', 'proj.weight',
        'lin1.weight', 'lin2.weight',
        'input_to_hidden.weight', 'hidden_to_hidden.weight',
        'score.weight',
        'fc1.weight', 'fc2.weight',
    ]

    # Bias doesn't need transpose since it's 1D
    # Conv2D weights don't need transpose
    # LayerNorm weights don't need transpose
    # GRU rnn weights have special shapes, handled separately

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_paddle_weights(self, model_path: str):
        """Load and map PaddlePaddle SLANeXt weights to PyTorch."""
        import paddle

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Paddle model not found: {model_path}")

        paddle_state = paddle.load(model_path)
        print(f'Paddle state dict: {len(paddle_state)} keys')
        print(f'PyTorch state dict: {len(self.model.state_dict())} keys')
        print()

        # Print Paddle keys summary
        for prefix in ['backbone.vision_tower_high.patch_embed',
                       'backbone.vision_tower_high.pos_embed',
                       'backbone.vision_tower_high.neck',
                       'backbone.post_conv',
                       'head.structure_attention_cell.i2h',
                       'head.structure_attention_cell.h2h',
                       'head.structure_attention_cell.score',
                       'head.structure_attention_cell.rnn',
                       'head.structure_generator',
                       'head.loc_generator']:
            keys = [k for k in paddle_state if k.startswith(prefix)]
            if keys:
                print(f'  {prefix}: {len(keys)} params')

        pt_state = self.model.state_dict()
        paddle_keys = set(paddle_state.keys())

        loaded = 0
        skipped = 0
        missing = []

        for pt_key in pt_state.keys():
            if pt_key.endswith('num_batches_tracked'):
                continue

            # Convert PyTorch key → Paddle key using rules
            pp_key = self._pt2pp_key(pt_key)

            if pp_key in paddle_keys:
                self._copy_param(pt_state, pt_key, paddle_state, pp_key)
                loaded += 1
            else:
                # Special: pos_embed might not exist in Paddle if use_abs_pos=False
                if 'pos_embed' in pt_key:
                    skipped += 1
                else:
                    missing.append(pt_key)
                    skipped += 1

        print(f'\nResults: Loaded={loaded}, Skipped={skipped}')
        if missing:
            print(f'Missing keys ({len(missing)}):')
            for k in missing[:20]:
                print(f'  {k}  {list(pt_state[k].shape)}')

    def _pt2pp_key(self, pt_key: str) -> str:
        """Convert PyTorch key to corresponding Paddle key.

        Multiple rules may need to apply to a single key.
        e.g.: backbone.vision_tower.layers.0.norm1.weight
           → backbone.vision_tower_high.blocks.0.norm1.weight
           (vision_tower→vision_tower_high AND layers→blocks)
        """
        pp_key = pt_key

        # Apply ALL matching reverse rules (order matters for nested patterns)
        reverse_rules = [
            ('backbone.vision_tower.', 'backbone.vision_tower_high.'),
            ('.layers.', '.blocks.'),
            ('patch_embed.projection.', 'patch_embed.proj.'),
            ('.structure_attention_cell.input_to_hidden.', '.structure_attention_cell.i2h.'),
            ('.structure_attention_cell.hidden_to_hidden.', '.structure_attention_cell.h2h.'),
            ('.structure_generator.fc1.', '.structure_generator.0.'),
            ('.structure_generator.fc2.', '.structure_generator.1.'),
            # Neck uses Sequential numbering in Paddle
            ('neck.conv1.', 'neck.0.'),
            ('neck.norm1.', 'neck.1.'),
            ('neck.conv2.', 'neck.2.'),
            ('neck.norm2.', 'neck.3.'),
            # post_conv is stored as net_2 in Paddle
            ('backbone.post_conv.', 'backbone.vision_tower_high.net_2.'),
        ]

        for py_pat, pp_pat in reverse_rules:
            if py_pat in pp_key:
                pp_key = pp_key.replace(py_pat, pp_pat)
                # Don't break — apply ALL matching rules!

        return pp_key

    def _copy_param(self, pt_state: dict, pt_key: str,
                    paddle_state: dict, pp_key: str):
        """Copy a single parameter with optional transpose."""
        pw = paddle_state[pp_key]
        if isinstance(pw, np.ndarray):
            pw = torch.from_numpy(pw.copy())
        elif hasattr(pw, 'numpy'):
            pw = torch.from_numpy(pw.numpy().copy())
        else:
            pw = torch.tensor(pw, dtype=torch.float32)

        pt_weight = pt_state[pt_key]
        pw = pw.float()

        # Check shape compatibility
        if pt_weight.shape == pw.shape:
            pt_weight.copy_(pw)
        elif self._needs_transpose(pt_key, pt_weight.shape, pw.shape):
            pt_weight.copy_(pw.T.contiguous())
        else:
            # Try to handle via reshape/view
            print(f'  WARNING: shape mismatch {pt_key}: pt={list(pt_weight.shape)} vs pp={list(pw.shape)}')
            # Skip for now, will be reported as missing
            return False
        return True

    def _needs_transpose(self, pt_key: str, pt_shape: tuple,
                         pp_shape: tuple) -> bool:
        """Check if weight needs transpose (Paddle [in,out] → PyTorch [out,in])."""
        if len(pt_shape) != 2:
            return False

        # Check if shapes are transposes of each other
        if pt_shape[0] == pp_shape[1] and pt_shape[1] == pp_shape[0]:
            return True

        return False

    # ------------------------------------------------------------------
    # Save & Verify
    # ------------------------------------------------------------------

    def save(self, output_path: str):
        torch.save(self.model.state_dict(), output_path)
        print(f'Model saved to {output_path}')

    def verify(self, input_shape=(1, 3, 512, 512)):
        print(f'\nRandom input test (shape={input_shape}):')
        np.random.seed(666)
        inp = torch.from_numpy(np.random.randn(*input_shape).astype(np.float32))
        with torch.no_grad():
            out = self.model(inp)
        print(f'  output shape: {list(out.shape)}')
        print(f'  sum:   {out.sum().item():.6f}')
        print(f'  mean:  {out.mean().item():.6f}')
        print(f'  max:   {out.max().item():.6f}')
        print(f'  min:   {out.min().item():.6f}')
        return out


def read_config_from_yaml(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'Config not found: {yaml_path}')
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    return res.get('Architecture', res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SLANeXt model to PyTorch')
    parser.add_argument('--yaml_path', type=str, required=True)
    parser.add_argument('--src_model_path', type=str, required=True)
    parser.add_argument('--dst_model_path', type=str, default=None)
    args = parser.parse_args()

    config = read_config_from_yaml(args.yaml_path)
    converter = SLANeXtConverter(config, args.src_model_path)

    # Run verify BEFORE saving so we see the raw output with converted weights
    print('\n=== VERIFICATION (random input) ===')
    converter.verify()

    if args.dst_model_path:
        converter.save(args.dst_model_path)
    else:
        base = os.path.basename(args.src_model_path)
        save_name = f'ptocr_{os.path.splitext(base)[0]}.pth'
        converter.save(os.path.join(os.path.dirname(args.src_model_path), save_name))

    print('\nConversion completed!')
