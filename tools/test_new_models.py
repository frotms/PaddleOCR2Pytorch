#!/usr/bin/env python3
"""
Test script for the newly ported models: textline_ori, doc_ori, and UVDoc.

Usage:
    python tools/test_new_models.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
import yaml

from pytorchocr.modeling.architectures.base_model import BaseModel
from pytorchocr.modeling.architectures.uvdoc_model import UVDocModel


def load_image(path, target_size=None):
    """Load and preprocess an image."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size:
        img = cv2.resize(img, target_size)
    return img


def preprocess_cls(img):
    """Preprocess for PP-LCNet classification (normalize to [0,1], then normalize with mean/std)."""
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, 0)  # Add batch dim
    return torch.from_numpy(img)


def test_textline_ori(model_path, yaml_path):
    """Test textline_ori model."""
    print("=" * 60)
    print("Testing textline_ori (PP-LCNet_x0_25_textline_ori)")
    print("=" * 60)

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    arch = cfg['Architecture']

    model = BaseModel(arch)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    # Create test images: normal and rotated 180
    test_dir = 'test_output'
    os.makedirs(test_dir, exist_ok=True)

    # Generate a synthetic test image: text "Hello" on white background
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'Hello', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    img_rot180 = cv2.rotate(img, cv2.ROTATE_180)

    cv2.imwrite(f'{test_dir}/textline_test.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{test_dir}/textline_test_rot180.jpg', cv2.cvtColor(img_rot180, cv2.COLOR_RGB2BGR))

    for name, img_test in [('normal', img), ('rotated_180', img_rot180)]:
        inp = preprocess_cls(cv2.resize(img_test, (224, 224)))
        with torch.no_grad():
            out = model(inp)
        probs = torch.softmax(out, dim=1).numpy()[0]
        pred = np.argmax(probs)
        labels = ['0_degree', '180_degree']
        print(f'  {name}: pred={labels[pred]}, probs={probs}')

    print(f'  Test images saved to {test_dir}/')
    print()


def test_doc_ori(model_path, yaml_path):
    """Test doc_ori model."""
    print("=" * 60)
    print("Testing doc_ori (PP-LCNet_x1_0_doc_ori)")
    print("=" * 60)

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    arch = cfg['Architecture']

    model = BaseModel(arch)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    # Create test image with text at different orientations
    test_dir = 'test_output'
    os.makedirs(test_dir, exist_ok=True)

    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'Document', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    rotations = {'0_degree': 0, '90_degree': cv2.ROTATE_90_CLOCKWISE,
                 '180_degree': cv2.ROTATE_180, '270_degree': cv2.ROTATE_90_COUNTERCLOCKWISE}

    for label, rot in rotations.items():
        if rot == 0:
            img_rot = img.copy()
        else:
            img_rot = cv2.rotate(img, rot)
        cv2.imwrite(f'{test_dir}/doc_ori_{label}.jpg', cv2.cvtColor(img_rot, cv2.COLOR_RGB2BGR))

        inp = preprocess_cls(cv2.resize(img_rot, (224, 224)))
        with torch.no_grad():
            out = model(inp)
        probs = torch.softmax(out, dim=1).numpy()[0]
        pred = np.argmax(probs)
        labels = ['0_degree', '90_degree', '180_degree', '270_degree']
        status = 'OK' if labels[pred] == label else 'MISMATCH'
        print(f'  true={label}: pred={labels[pred]} [{status}], probs={probs.round(4)}')

    print(f'  Test images saved to {test_dir}/')
    print()


def test_uvdoc(model_path):
    """Test UVDoc document unwarping model."""
    print("=" * 60)
    print("Testing UVDoc (document unwarping)")
    print("=" * 60)

    model = UVDocModel()
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    test_dir = 'test_output'
    os.makedirs(test_dir, exist_ok=True)

    # Create a synthetic warped document image
    h, w = 488, 712
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i in range(5):
        y = h // 6 * (i + 1)
        cv2.putText(img, 'Sample document text for testing UVDoc unwarping.',
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(f'{test_dir}/uvdoc_input.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Run UVDoc
    x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = model(x)

    grid_2d = out['unwarp_grid']
    mesh_3d = out['mesh_3d']

    # Unwarp
    unwarped, _ = model.unwarp(x)
    unwarped_img = unwarped[0].permute(1, 2, 0).numpy()
    unwarped_img = np.clip(unwarped_img, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{test_dir}/uvdoc_output.jpg', cv2.cvtColor(unwarped_img, cv2.COLOR_RGB2BGR))

    print(f'  Input: {img.shape}')
    print(f'  2D Unwarp grid: {grid_2d.shape}')
    print(f'  3D Mesh: {mesh_3d.shape}')
    print(f'  Unwarped image: {unwarped_img.shape}')
    print(f'  Test images saved to {test_dir}/')
    print()


if __name__ == '__main__':
    # textline_ori (2-class, scale=0.25)
    test_textline_ori(
        model_path='pretrained/PP-LCNet_x0_25_textline_ori_infer.pth',
        yaml_path='configs/cls/textline_ori/PP-LCNet_x0_25_textline_ori.yml'
    )

    # doc_ori (4-class, scale=1.0)
    test_doc_ori(
        model_path='pretrained/PP-LCNet_x1_0_doc_ori_infer.pth',
        yaml_path='configs/cls/doc_ori/PP-LCNet_x1_0_doc_ori.yml'
    )

    # UVDoc
    test_uvdoc(
        model_path='pretrained/UVDoc_infer.pth'
    )

    print("All tests completed!")
