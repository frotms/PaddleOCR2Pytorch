#!/usr/bin/env python
"""
PP-FormulaNet: Formula (LaTeX) Recognition Model for PP-StructureV3.

Architecture:
    Backbone: PPHGNetV2_B4/B6_Formula (already ported in pytorchocr)
    Head: PPFormulaNet_Head (MBart-style decoder with cross-attention)
    Tokenizer: UniMERNet tokenizer (HuggingFace tokenizers format)

Usage:
    from ptstructure.formula.ppformulanet import FormulaRecognizer

    recognizer = FormulaRecognizer(variant='M')
    recognizer.load_weights('models/structurev3/ptocr_formulanet_m.pth')
    latex = recognizer.recognize(cropped_formula_image)  # BGR image
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PPFormulaNet(nn.Module):
    """PP-FormulaNet model: PPHGNetV2 backbone + PPFormulaNet_Head decoder."""

    def __init__(
        self,
        backbone,
        head,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        """Inference forward: backbone extracts features, head decodes to token IDs."""
        encoder_outputs = self.backbone(x)
        # encoder_outputs should be a dict with 'last_hidden_state' key
        if isinstance(encoder_outputs, torch.Tensor):
            # If backbone returns a flat tensor, wrap it
            encoder_outputs = {'last_hidden_state': encoder_outputs.unsqueeze(1)}
        elif not isinstance(encoder_outputs, dict):
            encoder_outputs = {'last_hidden_state': encoder_outputs}
        word_pred = self.head(encoder_outputs)
        return word_pred


class FormulaRecognizer:
    """Formula (LaTeX) recognition wrapper for PP-StructureV3.

    Handles model loading, preprocessing, inference, and token decoding.

    Args:
        variant: Model variant ('S' or 'M'). S uses PPHGNetV2_B4 backbone, M uses PPHGNetV2_B6.
        device: Torch device string.

    Variant specs:
        S:  PPHGNetV2_B4_Formula + 2 decoder layers, 384 hidden dim
        M:  PPHGNetV2_B6_Formula + 6 decoder layers, 512 hidden dim
    """

    # Model variant configurations
    VARIANTS = {
        'S': {
            'backbone': 'PPHGNetV2_B4_Formula',
            'input_size': [384, 384],
            'decoder_layers': 2,
            'decoder_hidden_size': 384,
            'decoder_ffn_dim': 1536,
            'encoder_hidden_size': 2048,  # backbone output dim
            'max_new_tokens': 1024,
        },
        'M': {
            'backbone': 'PPHGNetV2_B6_Formula',
            'input_size': [384, 384],
            'decoder_layers': 6,
            'decoder_hidden_size': 512,
            'decoder_ffn_dim': 2048,
            'encoder_hidden_size': 2048,  # backbone output dim
            'max_new_tokens': 2560,
        },
    }

    def __init__(
        self,
        variant='M',
        device='cpu',
    ):
        self.variant = variant.upper()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(self.VARIANTS.keys())}")

        self.cfg = self.VARIANTS[self.variant]
        self.device = device
        self.model = None
        self.tokenizer = None

        # ImageNet normalization (same as training)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._build_model()

    def _build_model(self):
        """Build the PP-FormulaNet PyTorch model."""
        from pytorchocr.modeling.backbones.rec_pphgnetv2 import (
            PPHGNetV2_B4_Formula,
            PPHGNetV2_B6_Formula,
        )
        from .ppformulanet_head import PPFormulaNet_Head

        # Build backbone
        backbone_fn = {
            'PPHGNetV2_B4_Formula': PPHGNetV2_B4_Formula,
            'PPHGNetV2_B6_Formula': PPHGNetV2_B6_Formula,
        }[self.cfg['backbone']]
        backbone = backbone_fn()

        # Build head
        head = PPFormulaNet_Head(
            max_new_tokens=self.cfg['max_new_tokens'],
            decoder_start_token_id=0,
            temperature=0.2,
            do_sample=False,
            top_p=0.95,
            encoder_hidden_size=self.cfg['encoder_hidden_size'],
            decoder_hidden_size=self.cfg['decoder_hidden_size'],
            decoder_ffn_dim=self.cfg['decoder_ffn_dim'],
            decoder_layers=self.cfg['decoder_layers'],
        )

        self.model = PPFormulaNet(backbone, head)
        self.model.eval()

    def _load_tokenizer(self, tokenizer_path=None):
        """Load UniMERNet tokenizer using our minimal BPE tokenizer."""
        if tokenizer_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            tokenizer_path = os.path.join(
                repo_root, 'pytorchocr', 'utils', 'dict', 'unimernet_tokenizer', 'tokenizer.json'
            )
        from .tokenizer import FormulaTokenizer
        self.tokenizer = FormulaTokenizer(tokenizer_path)

    def load_weights(self, model_path, tokenizer_path=None):
        """Load pretrained weights.

        Args:
            model_path: Path to .pth model weights file.
            tokenizer_path: Path to tokenizer.json (optional).
        """
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self._load_tokenizer(tokenizer_path)
        logger.info(f'Formula recognizer ({self.variant}) loaded from {model_path}')

    def preprocess(self, img):
        """Preprocess formula image for inference.

        Args:
            img: BGR image as numpy array (H, W, 3).

        Returns:
            torch.Tensor of shape (1, 3, H, W) normalized.
        """
        ih, iw = self.cfg['input_size']
        # Resize to fixed size
        img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Add batch dim
        img = torch.from_numpy(img).float().unsqueeze(0)
        return img

    def decode_tokens(self, token_ids):
        """Decode token IDs to LaTeX string.

        Args:
            token_ids: numpy array or torch tensor of token IDs.

        Returns:
            LaTeX string.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Convert to list
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        token_ids = token_ids.flatten().tolist()

        # Cut at EOS token (2)
        for i, tid in enumerate(token_ids):
            if tid in (2,):  # EOS
                token_ids = token_ids[:i + 1]
                break

        # Use the minimal tokenizer to decode
        latex = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Post-process LaTeX
        from .postprocess import post_process_formula
        latex = post_process_formula(latex)
        return latex

    def recognize(self, img):
        """Recognize formula from input image.

        Args:
            img: BGR image as numpy array.

        Returns:
            LaTeX formula string.
        """
        if self.model is None:
            raise RuntimeError('Model not loaded. Call load_weights() first.')

        # Preprocess
        inp = self.preprocess(img)
        inp = inp.to(self.device)

        # Inference
        with torch.no_grad():
            token_ids = self.model(inp)

        # Decode
        latex = self.decode_tokens(token_ids)
        return latex


def load_formula_recognizer(model_path=None, variant='M', tokenizer_path=None, device='cpu'):
    """Factory function to load formula recognizer.

    Args:
        model_path: Path to .pth model weights.
        variant: Model variant ('S' or 'M').
        tokenizer_path: Path to tokenizer.json.
        device: Torch device.

    Returns:
        FormulaRecognizer instance.
    """
    recognizer = FormulaRecognizer(variant=variant, device=device)
    if model_path is not None:
        recognizer.load_weights(model_path, tokenizer_path)
    return recognizer
