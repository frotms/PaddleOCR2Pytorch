#!/usr/bin/env python
"""
Seal Text Detection for PP-StructureV3.

The seal detection model uses the same DB (Differentiable Binarization) architecture
as regular text detection, but trained specifically on seal/stamp images.
Reuses the existing PyTorch DB detector infrastructure.

Architecture:
    Backbone: PPLCNetV3 (scale=0.75)
    Neck: RSEFPN (out_channels=96)
    Head: DBHead (k=50)

Usage:
    from ptstructure.seal.seal_det import SealDetector

    detector = SealDetector()
    detector.load_weights('models/structurev3/ptocr_seal_det.pth')
    boxes = detector.detect(seal_crop_image)  # BGR image
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class SealDetector:
    """Seal text detector for PP-StructureV3.

    Detects text regions inside seal/stamp images using DB text detection.
    The detected text can then be fed to OCR for recognition.

    Args:
        det_limit_side_len: Maximum side length for detection.
        det_db_thresh: DB threshold for binarization.
        det_db_box_thresh: Box threshold for filtering.
        det_db_unclip_ratio: Unclip ratio for text region expansion.
        device: Torch device string.
    """

    def __init__(
        self,
        det_limit_side_len=960,
        det_db_thresh=0.1,
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=0.3,
        device='cpu',
    ):
        # Seal images often have faint/low-contrast text
        # Use lower defaults than regular text detection
        self.det_limit_side_len = det_limit_side_len
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.device = device
        self.model = None
        self.post_process = None

    def load_weights(self, model_path, yaml_path=None):
        """Load seal detection model weights.

        Args:
            model_path: Path to .pth model weights.
            yaml_path: Optional path to YAML config.
        """
        if yaml_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            yaml_path = os.path.join(
                repo_root, 'configs', 'det', 'PP-OCRv4', 'PP-OCRv4_mobile_seal_det.yml'
            )

        import yaml
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        config = cfg['Architecture']

        from pytorchocr.modeling.architectures.base_model import BaseModel
        self.model = BaseModel(config)
        self.model.eval()

        # Load weights
        weights = self.read_pytorch_weights(model_path)
        self.model.load_state_dict(weights, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Set up post-process
        from pytorchocr.postprocess import build_post_process
        postprocess_params = {
            'name': 'DBPostProcess',
            'thresh': self.det_db_thresh,
            'box_thresh': self.det_db_box_thresh,
            'max_candidates': 1000,
            'unclip_ratio': self.det_db_unclip_ratio,
            'box_type': 'poly',
        }
        self.post_process = build_post_process(postprocess_params)

        logger.info(f'Seal detector loaded from {model_path}')

    def read_pytorch_weights(self, model_path):
        """Read PyTorch weights."""
        return torch.load(model_path, map_location='cpu', weights_only=True)

    def preprocess(self, img):
        """Preprocess image for seal text detection.

        Args:
            img: BGR image as numpy array (H, W, 3).

        Returns:
            (image_tensor, shape_info)
        """
        h, w = img.shape[:2]

        # Resize to ensure dimensions are compatible with the RSEFPN neck
        # The model was trained with specific input sizes; 640x640 is recommended
        # but we resize to a compatible size based on the limit
        target_size = self.det_limit_side_len
        if max(h, w) > target_size:
            ratio = target_size / max(h, w)
        else:
            ratio = 1.0

        new_h, new_w = int(h * ratio), int(w * ratio)
        # Ensure dimensions are multiples of 32 (FPN compatibility)
        new_h = max(32, (new_h // 32) * 32)
        new_w = max(32, (new_w // 32) * 32)
        img = cv2.resize(img, (new_w, new_h))

        # Normalize
        img = img.astype(np.float32)
        img = img[..., ::-1]  # BGR to RGB
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img / 255.0
        img = (img - mean) / std

        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # Shape info for post-processing: (original_h, original_w, ratio_h, ratio_w)
        shape_info = {
            'shape': [(h, w, new_h / h, new_w / w)],
        }

        return img, shape_info

    def detect(self, img):
        """Detect text regions in seal image.

        Args:
            img: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected text boxes (each as [x1, y1, x2, y2]).
            List of scores for each box.
        """
        if self.model is None:
            raise RuntimeError('Model not loaded. Call load_weights() first.')

        ori_h, ori_w = img.shape[:2]

        # Preprocess
        inp, shape_info = self.preprocess(img)
        inp = torch.from_numpy(inp).float().to(self.device)

        # Inference
        with torch.no_grad():
            preds = self.model(inp)

        # Ensure preds is a dict with 'maps' key
        if isinstance(preds, torch.Tensor):
            preds = {'maps': preds}
        elif isinstance(preds, (list, tuple)):
            preds = {'maps': preds[0]}

        # Post-process
        shape_list = shape_info.get('shape', [(ori_h, ori_w)])
        if self.post_process is not None:
            post_result = self.post_process(preds, shape_list)
            boxes = post_result[0].get('points', [])
            scores = post_result[0].get('scores', [])
        else:
            boxes = []
            scores = []

        # Filter boxes and convert to list of [x1,y1,x2,y2]
        filtered_boxes = []
        filtered_scores = []
        for box, score in zip(boxes, scores):
            if score < 0.3:  # minimum score threshold
                continue
            # box is polygon points, convert to bbox
            box = np.array(box)
            x1, y1 = box.min(axis=0)
            x2, y2 = box.max(axis=0)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(ori_w, int(x2)), min(ori_h, int(y2))
            if x2 > x1 and y2 > y1:
                filtered_boxes.append([x1, y1, x2, y2])
                filtered_scores.append(float(score))

        return filtered_boxes, filtered_scores


def load_seal_detector(model_path=None, yaml_path=None, device='cpu'):
    """Factory function to load seal detector.

    Args:
        model_path: Path to .pth model weights.
        yaml_path: Path to YAML config.
        device: Torch device.

    Returns:
        SealDetector instance.
    """
    detector = SealDetector(device=device)
    if model_path is not None:
        detector.load_weights(model_path, yaml_path)
    return detector
