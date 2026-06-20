"""
PP-FormulaNet formula recognition module for PP-StructureV3.

Self-contained under ptstructure/formula/ — no external model imports needed
except the PPHGNetV2 backbone (shared from pytorchocr.modeling.backbones).

Files:
    unimernet_head.py    — UniMERNet/MBart decoder (PyTorch port)
    ppformulanet_head.py — PPFormulaNet Head
    ppformulanet.py      — FormulaRecognizer (main entry point)
    tokenizer.py         — Minimal BPE tokenizer (no tokenizers library needed)
    postprocess.py       — LaTeX string post-processing
"""

from .ppformulanet import FormulaRecognizer, load_formula_recognizer
