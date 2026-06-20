# Copyright (c) 2025 PaddleOCR2Pytorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PP-StructureV3: Document Structure Analysis Pipeline

A PyTorch implementation of PaddleOCR's PP-StructureV3 system for
document layout parsing, table recognition, formula recognition,
seal detection, and structured output generation.

Pipeline:
    Input Image → Layout Detection → [OCR + Table + Formula + Seal] → Markdown/JSON

Features:
    - Layout detection: PPDocLayout (PicoDet), 23 classes
    - OCR: PP-OCRv6 (DB + SVTR)
    - Table structure: SLANeXt (ViT + GRU)
    - Formula recognition: PP-FormulaNet (PPHGNetV2 + MBart)
    - Seal detection: DB-based seal text detector

Usage:
    python ptstructure/predict_structure.py --image_dir=./data/ --output_dir=./output/
"""

__all__ = []
