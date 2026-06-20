"""
Document preprocessing for PP-StructureV3.

Provides optional preprocessing steps that run BEFORE layout detection:
    - DocOrientationClassifier: 4-class document image orientation (0°/90°/180°/270°)
    - UVDocUnwarper: document image unwarping/rectification (CGU-Net)
    - Text line orientation is handled by the OCR TextSystem (use_angle_cls)

Usage:
    from ptstructure.doc_preprocess import (
        DocOrientationClassifier,
        UVDocUnwarper,
    )
"""

from .doc_orientation import DocOrientationClassifier, load_doc_orientation_classifier
from .unwarp import UVDocUnwarper, load_doc_unwarper
