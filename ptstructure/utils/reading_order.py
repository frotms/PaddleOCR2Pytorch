"""
Reading Order Recovery using XY-Cut algorithm.

Implementation based on PaddleOCR's layout parsing utilities.
"""

import numpy as np
from typing import List, Dict, Tuple


def _xy_cut(blocks: List[Dict], direction: str = 'horizontal', depth: int = 0) -> List[Dict]:
    """Recursively sort blocks by reading order using XY-Cut algorithm.

    Args:
        blocks: List of blocks, each with 'bbox' [x1, y1, x2, y2].
        direction: Splitting direction ('horizontal' or 'vertical').
        depth: Current recursion depth (prevent infinite loops).

    Returns:
        Sorted list of blocks in reading order.
    """
    if len(blocks) <= 1:
        return blocks

    # Prevent infinite recursion on overlapping boxes
    if depth > 50:
        return sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))

    if direction == 'horizontal':
        blocks = sorted(blocks, key=lambda b: b['bbox'][1])  # sort by y1
        gaps = []
        for i in range(len(blocks) - 1):
            y_gap = blocks[i + 1]['bbox'][1] - blocks[i]['bbox'][3]
            if y_gap > 0:
                gaps.append((y_gap, i))
        if gaps:
            _, split_idx = max(gaps, key=lambda g: g[0])
            top = blocks[:split_idx + 1]
            bottom = blocks[split_idx + 1:]
            if len(top) == len(blocks) or len(bottom) == len(blocks):
                return blocks  # no progress
            return _xy_cut(top, 'vertical', depth + 1) + _xy_cut(bottom, 'vertical', depth + 1)
        else:
            return _xy_cut(blocks, 'vertical', depth + 1)
    else:
        blocks = sorted(blocks, key=lambda b: b['bbox'][0])  # sort by x1
        gaps = []
        for i in range(len(blocks) - 1):
            x_gap = blocks[i + 1]['bbox'][0] - blocks[i]['bbox'][2]
            if x_gap > 0:
                gaps.append((x_gap, i))
        if gaps:
            _, split_idx = max(gaps, key=lambda g: g[0])
            left = blocks[:split_idx + 1]
            right = blocks[split_idx + 1:]
            if len(left) == len(blocks) or len(right) == len(blocks):
                return blocks  # no progress
            return _xy_cut(left, 'horizontal', depth + 1) + _xy_cut(right, 'horizontal', depth + 1)
        else:
            return _xy_cut(blocks, 'horizontal', depth + 1)


def xy_cut(blocks: List[Dict], direction: str = 'horizontal') -> List[int]:
    """Sort blocks by reading order and return sorted indices.

    Args:
        blocks: List of block dicts with 'bbox' [x1, y1, x2, y2].
        direction: Initial split direction.

    Returns:
        List of indices in reading order.
    """
    if not blocks:
        return []

    indexed_blocks = [(i, b) for i, b in enumerate(blocks)]
    sorted_pairs = _xy_cut(
        [{'bbox': b.get('bbox', b.get('block_bbox', [0,0,0,0])), 'idx': i}
         for i, b in indexed_blocks], direction
    )
    return [b['idx'] for b in sorted_pairs]


def recover_reading_order(blocks: List[Dict], direction: str = 'horizontal') -> List[Dict]:
    """Sort blocks by reading order and assign block_order.

    Args:
        blocks: List of block dicts, each with 'bbox' [x1, y1, x2, y2]
                and 'block_label', 'block_content', etc.
        direction: Initial split direction for XY-Cut.

    Returns:
        Blocks sorted in reading order with 'block_order' assigned.
    """
    if not blocks:
        return blocks

    indices = xy_cut(blocks, direction)
    result = []
    for order, idx in enumerate(indices):
        block = dict(blocks[idx])
        block['block_order'] = order
        result.append(block)
    return result
