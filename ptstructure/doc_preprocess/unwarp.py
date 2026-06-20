#!/usr/bin/env python
"""
Document Image Unwarping (UVDoc CGU-Net).

Corrects perspective distortion and page curl in document images using
a neural grid-based approach.

Model: UVDoc CGU-Net (~30MB)
Input: 488x712 (standard UVDoc size)

Usage:
    from ptstructure.doc_preprocess.unwarp import UVDocUnwarper

    unwarper = UVDocUnwarper()
    unwarper.load_weights('models/structurev3/ptocr_uvdoc.pth')
    corrected_img = unwarper.unwarp(img)
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class UVDocUnwarper:
    """Document image unwarping using UVDoc CGU-Net.

    Args:
        model_path: Path to .pth model weights.
        input_size: UVDoc input size as (H, W). Default is (488, 712).
        device: Torch device.
    """

    def __init__(
        self,
        model_path=None,
        input_size=(488, 712),
        device='cpu',
    ):
        self.input_size = input_size
        self.device = device
        self.model = None

        if model_path is not None:
            self.load_weights(model_path)

    def load_weights(self, model_path):
        """Load UVDoc model weights.

        Args:
            model_path: Path to .pth model weights.
        """
        from pytorchocr.modeling.architectures.uvdoc_model import UVDocModel

        self.model = UVDocModel()
        self.model.eval()

        sd = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info('UVDocUnwarper loaded from %s', model_path)

    def _preprocess(self, img):
        """Preprocess image for UVDoc: resize to fixed input size and normalize.

        Args:
            img: BGR image as numpy array (H, W, 3).

        Returns:
            torch.Tensor of shape (1, 3, H, W).
        """
        ih, iw = self.input_size
        # Resize
        img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float().unsqueeze(0)
        return img

    def unwarp(self, img, output_size=None):
        """Unwarp a document image.

        Args:
            img: BGR image as numpy array (H, W, 3).
            output_size: Output (H, W) tuple. If None, uses input image size.

        Returns:
            Corrected BGR image as numpy array.
        """
        if self.model is None:
            raise RuntimeError('Model not loaded. Call load_weights() first.')

        ori_h, ori_w = img.shape[:2]
        if output_size is None:
            output_size = (ori_h, ori_w)

        # Preprocess
        inp = self._preprocess(img)
        inp = inp.to(self.device)

        # Run UVDoc
        with torch.no_grad():
            result = self.model(inp)

        grid = result['unwarp_grid']  # [1, 2, gh, gw]

        # Upsample grid to output size
        grid_full = F.interpolate(grid, size=output_size,
                                  mode='bilinear', align_corners=True)

        # Convert to grid_sample format [1, H, W, 2] normalized to [-1, 1]
        # Grid is predicted at the UVDoc input resolution
        ih, iw = self.input_size
        grid_norm = grid_full.permute(0, 2, 3, 1)
        grid_norm[..., 0] = grid_norm[..., 0] / max(iw - 1, 1) * 2.0 - 1.0
        grid_norm[..., 1] = grid_norm[..., 1] / max(ih - 1, 1) * 2.0 - 1.0

        # Resize input image tensor to match grid sample
        inp_resized = F.interpolate(inp, size=output_size,
                                     mode='bilinear', align_corners=True)

        # Apply grid sampling
        unwarped = F.grid_sample(inp_resized, grid_norm, mode='bilinear',
                                 padding_mode='zeros', align_corners=True)

        # Convert back to numpy BGR
        unwarped = unwarped.squeeze(0).cpu().numpy()  # [3, H, W]
        unwarped = unwarped.transpose(1, 2, 0)  # HWC
        unwarped = (unwarped * 255.0).clip(0, 255).astype(np.uint8)
        unwarped = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)

        logger.info('  UVDoc unwarp: %dx%d → %dx%d',
                    ori_w, ori_h, output_size[1], output_size[0])
        return unwarped

    def __call__(self, img):
        """Unwarp in one call."""
        return self.unwarp(img)


def load_doc_unwarper(model_path=None, device='cpu'):
    """Factory function."""
    return UVDocUnwarper(model_path=model_path, device=device)
