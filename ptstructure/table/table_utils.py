"""
Table utilities: HTML generation from structure tokens and OCR results.
"""

from typing import List, Dict, Optional, Tuple


# Official SLANeXt character dictionary (from HuggingFace model config.json)
# Exactly 48 characters + sos + eos = 50 tokens (matches model out_channels=50)
# PaddleOCR loads this from PostProcess.character_dict in the inference config.
DEFAULT_TABLE_CHAR_DICT = [
    '<thead>', '</thead>', '<tbody>', '</tbody>',
    '<tr>', '</tr>', '<td>', '<td', '>', '</td>',
    ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"',
    ' colspan="6"', ' colspan="7"', ' colspan="8"', ' colspan="9"',
    ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"',
    ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"',
    ' colspan="18"', ' colspan="19"', ' colspan="20"',
    ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"',
    ' rowspan="6"', ' rowspan="7"', ' rowspan="8"', ' rowspan="9"',
    ' rowspan="10"', ' rowspan="11"', ' rowspan="12"', ' rowspan="13"',
    ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"',
    ' rowspan="18"', ' rowspan="19"', ' rowspan="20"',
]


def structure_to_html(
    structure_tokens: List[int],
    char_dict: Optional[List[str]] = None,
    cell_boxes: Optional[List[List[float]]] = None,
    cell_contents: Optional[List[str]] = None,
) -> str:
    """Convert SLANeXt structure tokens to HTML table.

    Args:
        structure_tokens: List of token indices from SLANeXt decoder.
        char_dict: Character dictionary mapping token idx → str.
        cell_boxes: Optional cell bounding boxes for coordinate mapping.
        cell_contents: Optional OCR text for each cell.

    Returns:
        HTML table string.
    """
    if char_dict is None:
        char_dict = DEFAULT_TABLE_CHAR_DICT

    # Decode tokens to character sequence
    chars = []
    prev_token = None
    for token in structure_tokens:
        if token == 0:
            break  # EOS or blank
        if token < len(char_dict):
            char = char_dict[token]
        else:
            char = ''
        chars.append(char)
        prev_token = token

    html = ''.join(chars)

    # Wrap in table tags if needed
    html = html.strip()
    if not html.startswith('<table'):
        html = f'<table>{html}</table>'
    if not html.endswith('</table>'):
        html = f'{html}</table>'

    # Insert OCR cell contents if provided
    if cell_contents and cell_boxes:
        html = _insert_cell_contents(html, cell_contents)

    return html


def _insert_cell_contents(html: str, cell_contents: List[str]) -> str:
    """Insert OCR text content into HTML <td> tags.

    Simple approach: replace each <td></td> pair with <td>content</td>.
    """
    import re
    # Find all <td...>...</td> patterns
    td_pattern = re.compile(r'(<td[^>]*>)(</td>)')
    matches = list(td_pattern.finditer(html))

    if len(matches) <= len(cell_contents):
        # Replace from end to start to preserve positions
        result = html
        for i, match in enumerate(reversed(matches)):
            content_idx = len(matches) - 1 - i
            if content_idx < len(cell_contents):
                new_td = f'{match.group(1)}{cell_contents[content_idx]}{match.group(2)}'
                result = result[:match.start()] + new_td + result[match.end():]
        return result

    return html


def cells_to_html(
    cells: List[List[Dict]],
    headers: Optional[List[str]] = None,
) -> str:
    """Convert structured cell data to HTML table.

    Args:
        cells: 2D list of cell dicts, each with:
            - 'text': cell text content
            - 'colspan': column span (default 1)
            - 'rowspan': row span (default 1)
            - 'bbox': [x1, y1, x2, y2]
        headers: Optional list of header texts for the first row.

    Returns:
        HTML table string.
    """
    if not cells:
        return '<table></table>'

    rows_html = []
    for r_idx, row in enumerate(cells):
        cells_html = []
        for c_idx, cell in enumerate(row):
            tag = 'th' if (headers and r_idx == 0) else 'td'
            attrs = ''
            colspan = cell.get('colspan', 1)
            rowspan = cell.get('rowspan', 1)
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'

            text = cell.get('text', '')
            cells_html.append(f'<{tag}{attrs}>{text}</{tag}>')
        rows_html.append(f'<tr>{"".join(cells_html)}</tr>')

    return f'<table>{"".join(rows_html)}</table>'


def parse_table_bbox_from_html(html: str, cell_bboxes: List[List[float]]) -> List[Dict]:
    """Extract cell information from HTML with bounding boxes.

    Returns:
        List of cell dicts with bbox and text.
    """
    import re
    cells = []
    td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)

    for i, match in enumerate(td_pattern.finditer(html)):
        cell_info = {
            'text': match.group(1).strip(),
            'bbox': cell_bboxes[i] if i < len(cell_bboxes) else [0, 0, 0, 0],
        }
        cells.append(cell_info)

    return cells
