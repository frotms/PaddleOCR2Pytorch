"""
Seal text detection module for PP-StructureV3.
Reuses DB text detector with seal-specific weights.
"""

from .seal_det import SealDetector, load_seal_detector
