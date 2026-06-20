"""
Markdown and JSON output generation from PP-StructureV3 parsing results.
"""

import json
from typing import List, Dict, Optional


def blocks_to_markdown(
    blocks: List[Dict],
    ignore_labels: Optional[List[str]] = None,
    image_base64: bool = False,
) -> str:
    """Convert parsed blocks to Markdown format.

    Args:
        blocks: List of parsed blocks, each with:
            - block_label: str (doc_title, paragraph_title, text, table, figure, etc.)
            - block_content: str
            - block_bbox: [x1, y1, x2, y2]
            - block_order: int
        ignore_labels: Labels to skip in Markdown output.
        image_base64: Whether to include image base64 data (not supported yet).

    Returns:
        Markdown string.
    """
    if ignore_labels is None:
        ignore_labels = ['header', 'footer', 'page_number', 'footnote']

    # Sort by block_order
    sorted_blocks = sorted(blocks, key=lambda b: b.get('block_order', 0))

    lines = []
    for block in sorted_blocks:
        label = block.get('block_label', 'text')
        content = block.get('block_content', '')

        if label in ignore_labels:
            continue

        if not content or not content.strip():
            continue

        content = content.strip()

        if label in ('doc_title',):
            lines.append(f'# {content}')
            lines.append('')
        elif label in ('paragraph_title',):
            lines.append(f'## {content}')
            lines.append('')
        elif label in ('sub_title', 'figure_title', 'table_title'):
            lines.append(f'### {content}')
            lines.append('')
        elif label in ('formula',):
            lines.append(f'$${content}$$')
            lines.append('')
        elif label == 'table':
            # content is already HTML table or can be converted
            lines.append(content)
            lines.append('')
        elif label in ('figure', 'chart', 'seal', 'image'):
            lines.append(f'*[{label.upper()}: {content}]*')
            lines.append('')
        elif label == 'reference':
            lines.append(f'> {content}')
            lines.append('')
        elif label == 'code':
            lines.append(f'```\n{content}\n```')
            lines.append('')
        else:
            # text, paragraph, etc.
            lines.append(content)
            lines.append('')

    return '\n'.join(lines)


def blocks_to_json(blocks: List[Dict], input_path: str = '') -> str:
    """Convert parsed blocks to JSON format.

    Args:
        blocks: List of parsed blocks.
        input_path: Original input image path.

    Returns:
        JSON string.
    """
    result = {
        'input_path': input_path,
        'blocks': [],
    }

    for block in sorted(blocks, key=lambda b: b.get('block_order', 0)):
        result['blocks'].append({
            'block_id': block.get('block_id', 0),
            'block_label': block.get('block_label', 'text'),
            'block_content': block.get('block_content', ''),
            'block_bbox': block.get('block_bbox', []),
            'block_order': block.get('block_order', 0),
            'confidence': block.get('confidence', 1.0),
        })

    return json.dumps(result, ensure_ascii=False, indent=2)
