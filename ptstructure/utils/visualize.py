"""
Visualization utilities for PP-StructureV3 results.
Uses PIL for text rendering to support CJK and other non-Latin characters.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont


# Color palette for different layout categories
LABEL_COLORS = {
    'doc_title': (255, 0, 0),           # Red
    'paragraph_title': (0, 128, 255),   # Orange-blue
    'text': (0, 255, 0),                # Green
    'table': (255, 128, 0),             # Orange
    'figure': (128, 0, 255),            # Purple
    'formula': (255, 255, 0),           # Yellow
    'header': (0, 255, 255),            # Cyan
    'footer': (255, 0, 255),            # Magenta
    'seal': (0, 0, 255),                # Blue
    'chart': (128, 128, 0),             # Olive
    'reference': (128, 128, 128),       # Gray
    'code': (0, 128, 128),              # Teal
    'number': (128, 0, 128),            # Maroon
    'page_number': (128, 0, 128),       # Maroon
    'content': (0, 100, 0),             # Dark Green
    'abstract': (100, 100, 255),        # Light Blue
    'image': (0, 165, 255),             # Orange-yellow
    'table_title': (200, 50, 50),       # Dark red
    'figure_title': (200, 100, 50),     # Brown
    'aside_text': (100, 200, 100),      # Light green
    'footnote': (150, 150, 200),        # Light purple
    'formula_number': (100, 200, 200),  # Teal
    'chart_title': (50, 150, 150),      # Dark teal
    'algorithm': (200, 150, 50),        # Bronze
    'header_image': (150, 200, 150),    # Sage
    'footer_image': (150, 150, 200),    # Lavender
}
DEFAULT_COLOR = (255, 255, 255)


def _get_color(label: str) -> tuple:
    return LABEL_COLORS.get(label, DEFAULT_COLOR)


def _get_cjk_font(font_size: int = 14) -> ImageFont.FreeTypeFont:
    """Get a PIL font that supports CJK characters.

    Searches doc/fonts/ for a suitable CJK font file.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    font_dir = os.path.join(repo_root, 'doc', 'fonts')

    # Try Chinese fonts first (widest character coverage)
    candidates = ['chinese_cht.ttf', 'simfang.ttf', 'japan.ttc', 'korean.ttf',
                  'latin.ttf', 'french.ttf', 'german.ttf']
    for name in candidates:
        path = os.path.join(font_dir, name)
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    # Fallback to PIL default
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    except Exception:
        return ImageFont.load_default()


def _put_text_pil(img_bgr: np.ndarray, text: str, position: tuple,
                  font_size: int = 14, color: tuple = (255, 255, 255),
                  bg_color: Optional[tuple] = None) -> None:
    """Draw text on a BGR OpenCV image using PIL (supports CJK characters).

    Args:
        img_bgr: OpenCV BGR image (modified in-place).
        text: Text to draw (supports Unicode/CJK).
        position: (x, y) top-left position.
        font_size: Font size in pixels.
        color: Text color in BGR order (like OpenCV).
        bg_color: Optional background fill color (BGR). If None, no background.
    """
    h, w = img_bgr.shape[:2]

    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = _get_cjk_font(font_size)

    # PIL color is RGB
    rgb_color = (color[2], color[1], color[0])

    # Get text bounding box
    bbox = draw.textbbox(position, text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Draw background if requested
    if bg_color is not None:
        rgb_bg = (bg_color[2], bg_color[1], bg_color[0])
        pad = 2
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                       fill=rgb_bg)

    # Draw text
    draw.text(position, text, font=font, fill=rgb_color)

    # Convert back to BGR and copy into original array
    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[:] = result_bgr


def _get_text_size_pil(text: str, font_size: int = 14) -> tuple:
    """Get text width and height using PIL."""
    font = _get_cjk_font(font_size)
    # Create a temporary image to measure text
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def draw_layout_boxes(
    image: np.ndarray,
    layout_boxes: List[Dict],
    thickness: int = 2,
    font_size: int = 14,
) -> np.ndarray:
    """Draw layout detection bounding boxes on image.

    Args:
        image: Input image (BGR numpy array).
        layout_boxes: List of layout detection results,
                      each with 'bbox' [x1,y1,x2,y2], 'label', 'score'.
        thickness: Box line thickness.
        font_size: Font size for labels (PIL pixels).

    Returns:
        Image with drawn bounding boxes.
    """
    img = image.copy()
    if img.shape[0] > 2000 or img.shape[1] > 2000:
        scale = min(2000 / img.shape[0], 2000 / img.shape[1])
        img = cv2.resize(img, None, fx=scale, fy=scale)
        for box in layout_boxes:
            bbox = [int(c * scale) for c in box['bbox']]
            box['_scaled_bbox'] = bbox
    else:
        for box in layout_boxes:
            box['_scaled_bbox'] = box['bbox']

    for box in layout_boxes:
        x1, y1, x2, y2 = box.get('_scaled_bbox', box['bbox'])
        label = box.get('label', 'unknown')
        score = box.get('score', 0.0)
        color = _get_color(label)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        text = f'{label}: {score:.2f}'
        tw, th = _get_text_size_pil(text, font_size)
        # Draw filled background for label
        _put_text_pil(img, text, (x1, y1 - th - 4), font_size,
                     color=(255, 255, 255), bg_color=color)

    return img


def draw_structure_result(
    image: np.ndarray,
    blocks: List[Dict],
    output_path: Optional[str] = None,
    show_text: bool = True,
) -> np.ndarray:
    """Draw PP-StructureV3 parsing result on image.

    Args:
        image: Input image (BGR numpy array).
        blocks: List of parsed blocks with 'block_bbox', 'block_label', 'block_content'.
        output_path: Path to save the result image.
        show_text: Whether to show recognized text on the image.

    Returns:
        Annotated image.
    """
    img = image.copy()

    for block in blocks:
        bbox = block.get('block_bbox', [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        label = block.get('block_label', 'text')
        content = block.get('block_content', '')
        color = _get_color(label)

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label with PIL (supports Unicode)
        _put_text_pil(img, label, (x1, y1 - 18), font_size=14,
                     color=(255, 255, 255), bg_color=color)

        # Draw text content if requested
        if show_text and content:
            display_text = content[:100] + '...' if len(content) > 100 else content
            # Use a semi-transparent overlay for text readability
            y_offset = y1 + 2
            for line in display_text.split('\n')[:5]:
                if not line.strip():
                    y_offset += 16
                    continue
                _put_text_pil(img, line[:80], (x1 + 2, y_offset), font_size=12,
                             color=(0, 0, 0), bg_color=(255, 255, 255))
                y_offset += 16

    if output_path:
        cv2.imwrite(output_path, img)

    return img
