"""
Minimal BPE tokenizer for UniMERNet/PP-FormulaNet models.

Does NOT require the `tokenizers` library. Reads the tokenizer.json file
directly and provides basic decode functionality.

Reference: tokenizer.json is in HuggingFace tokenizers format (ByteLevel BPE).
"""

import json
import os
import re
from typing import List, Union


class FormulaTokenizer:
    """Minimal ByteLevel BPE tokenizer for formula/LaTeX decoding.

    Reads a tokenizer.json file (HuggingFace tokenizers format) and provides
    decode() functionality to convert token IDs to LaTeX strings.

    Usage:
        tokenizer = FormulaTokenizer('pytorchocr/utils/dict/unimernet_tokenizer/tokenizer.json')
        latex = tokenizer.decode([123, 456, 789])
    """

    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self._vocab = None
        self._id_to_token = None
        self._special_tokens = set()
        self._load()

    def _load(self):
        """Load tokenizer.json and build vocab mapping."""
        with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get vocab from BPE model
        model = data.get('model', {})
        self._vocab = model.get('vocab', {})

        # Build reverse mapping (id -> token)
        self._id_to_token = {}
        for token, idx in self._vocab.items():
            self._id_to_token[idx] = token

        # Get special tokens from added_tokens
        for tok_info in data.get('added_tokens', []):
            self._special_tokens.add(tok_info['content'])

        # Special tokens from model config
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        self.bos_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def id_to_token(self, idx: int) -> str:
        """Convert a token ID to its string representation."""
        return self._id_to_token.get(idx, self.unk_token)

    def decode(
        self,
        ids: Union[List[int], 'torch.Tensor', 'numpy.ndarray'],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to a LaTeX string.

        Args:
            ids: List/array of token IDs.
            skip_special_tokens: If True, skip special tokens like BOS/EOS/PAD.

        Returns:
            Decoded LaTeX string.
        """
        # Convert to Python list
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if isinstance(ids, (list, tuple)):
            ids = list(ids)
        else:
            ids = [ids]

        # Get tokens
        tokens = []
        for idx in ids:
            idx = int(idx)
            if idx not in self._id_to_token:
                continue
            token = self._id_to_token[idx]

            if skip_special_tokens and token in self._special_tokens:
                continue

            tokens.append(token)

        # ByteLevel decoding
        # In ByteLevel BPE, tokens are byte sequences encoded as strings
        # The 'Ġ' character (U+0120) represents a space at word start
        text_bytes = bytearray()
        for token in tokens:
            for ch in token:
                text_bytes.append(ord(ch))

        # Decode bytes to UTF-8
        try:
            text = text_bytes.decode('utf-8', errors='replace')
        except Exception:
            text = ''.join(tokens)

        # Post-process: replace Ġ with space (ByteLevel decoder)
        text = text.replace('Ġ', ' ')

        return text.strip()

    def token2str(self, token_ids: List[List[int]]) -> List[str]:
        """Convert batch of token ID sequences to LaTeX strings.

        Args:
            token_ids: List of token ID sequences (batch).

        Returns:
            List of decoded LaTeX strings.
        """
        from .postprocess import post_process_formula

        results = []
        for tok_ids in token_ids:
            # Truncate at EOS
            toks = list(tok_ids) if not isinstance(tok_ids, list) else tok_ids
            for i, tid in enumerate(toks):
                if tid == self.eos_token_id:
                    toks = toks[:i + 1]
                    break
            text = self.decode(toks, skip_special_tokens=True)
            text = post_process_formula(text)
            results.append(text)
        return results


# Global instance cache
_tokenizer_cache = {}


def get_formula_tokenizer(tokenizer_path: str = None) -> FormulaTokenizer:
    """Get or create a FormulaTokenizer instance (cached).

    Args:
        tokenizer_path: Path to tokenizer.json. If None, uses default path.

    Returns:
        FormulaTokenizer instance.
    """
    if tokenizer_path is None:
        # Default path relative to this file
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'dict', 'unimernet_tokenizer', 'tokenizer.json'
        )
        tokenizer_path = default_path

    if tokenizer_path not in _tokenizer_cache:
        _tokenizer_cache[tokenizer_path] = FormulaTokenizer(tokenizer_path)

    return _tokenizer_cache[tokenizer_path]
