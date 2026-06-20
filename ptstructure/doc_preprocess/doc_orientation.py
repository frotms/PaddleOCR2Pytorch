#!/usr/bin/env python
"""
Document Image Orientation Classification.

Classifies the whole document image into one of 4 orientations:
    0°, 90°, 180°, 270°

Uses PP-LCNet backbone + ClsHead (same architecture as textline_ori,
but with 4 classes instead of 2).

Model: PP-LCNet_x1_0_doc_ori (7MB)
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
import yaml

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DocOrientationClassifier:
    """Document orientation classifier.

    Detects and corrects the orientation of a document image.

    Args:
        model_path: Path to .pth model weights.
        yaml_path: Path to YAML config.
        cls_image_shape: Input shape as "C,H,W" string.
        cls_thresh: Confidence threshold.
        device: Torch device.
    """

    def __init__(
        self,
        model_path=None,
        yaml_path=None,
        cls_image_shape='3,48,192',
        cls_thresh=0.9,
        device='cpu',
    ):
        self.cls_image_shape = [int(v) for v in cls_image_shape.split(",")]
        self.cls_thresh = cls_thresh
        self.device = device
        self.model = None
        self.label_list = ['0', '180']  # default; overridden by YAML or config

        if model_path is not None:
            self.load_weights(model_path, yaml_path)

    def load_weights(self, model_path, yaml_path=None):
        """Load classification model weights.

        Args:
            model_path: Path to .pth model weights.
            yaml_path: Path to YAML config. Defaults to doc_ori config.
        """
        if yaml_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            yaml_path = os.path.join(
                repo_root, 'configs', 'cls', 'doc_ori', 'PP-LCNet_x1_0_doc_ori.yml'
            )

        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        config = cfg['Architecture']
        # Determine class count from config
        class_dim = config.get('Head', {}).get('class_dim', 4)
        self.label_list = [str(i * 90) for i in range(class_dim)] if class_dim == 4 else ['0', '180']

        from pytorchocr.modeling.architectures.base_model import BaseModel
        self.model = BaseModel(config)
        self.model.eval()

        sd = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info('DocOrientationClassifier loaded from %s', model_path)

    def _resize_norm(self, img):
        """Resize and normalize image for classification."""
        imgC, imgH, imgW = self.cls_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))
        resized_w = max(min(resized_w, 512), 16)  # limit width
        resized = cv2.resize(img, (resized_w, imgH))
        resized = resized.astype('float32')
        if imgC == 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
            resized = resized / 255.0
            resized = resized[np.newaxis, :]
        else:
            resized = resized.transpose((2, 0, 1)) / 255.0
        resized -= 0.5
        resized /= 0.5
        # Pad to target width
        padded = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        return padded

    def classify(self, img):
        """Classify the orientation of a document image.

        Args:
            img: BGR image as numpy array (H, W, 3).

        Returns:
            (label, score) — e.g. ('90', 0.95)
        """
        import math
        if self.model is None:
            raise RuntimeError('Model not loaded. Call load_weights() first.')

        inp = self._resize_norm(img)
        inp = torch.from_numpy(inp).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inp)
        out = out.cpu().numpy()

        idx = int(np.argmax(out, axis=1)[0])
        score = float(np.max(F.softmax(torch.from_numpy(out), dim=1).numpy()))
        label = self.label_list[idx] if idx < len(self.label_list) else str(idx * 90)

        return label, score

    def correct_orientation(self, img, label=None):
        """Correct image orientation by rotating if needed.

        Args:
            img: BGR image as numpy array.
            label: Orientation label ('0', '90', '180', '270'). If None, auto-detects.

        Returns:
            Corrected image (same shape if 0°, otherwise rotated).
        """
        if label is None:
            label, _ = self.classify(img)

        rotate_map = {
            '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
            '180': cv2.ROTATE_180,
            '270': cv2.ROTATE_90_CLOCKWISE,
        }
        if label in rotate_map:
            img = cv2.rotate(img, rotate_map[label])
            logger.info('  Doc orientation corrected: %s°', label)
        return img

    def __call__(self, img):
        """Classify and correct in one call."""
        return self.correct_orientation(img)


def load_doc_orientation_classifier(model_path=None, yaml_path=None, device='cpu'):
    """Factory function."""
    return DocOrientationClassifier(
        model_path=model_path,
        yaml_path=yaml_path,
        device=device,
    )


# Need these at module level
import math
import torch.nn.functional as F
