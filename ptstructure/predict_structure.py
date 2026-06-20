#!/usr/bin/env python
"""
PP-StructureV3: Document Structure Analysis Pipeline
All models run in PyTorch (no PaddleX dependency).

Supported features:
    - Layout detection (PPDocLayout/PicoDet)
    - OCR text detection + recognition (PP-OCRv6)
    - Table structure recognition (SLANeXt)
    - Formula recognition (PP-FormulaNet)
    - Seal text detection (DB-based)

Usage:
    # Full pipeline with all features
    python ptstructure/predict_structure.py \
        --image_dir=doc/imgs/ \
        --output_dir=output/ppstructurev3/ \
        --use_formula --use_seal

    # Basic pipeline (layout + OCR + table)
    python ptstructure/predict_structure.py \
        --image_dir=doc/imgs/ \
        --output_dir=output/ppstructurev3/
"""

import os, sys, argparse, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools/infer'))

import cv2, numpy as np, torch, json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_layout_detector(layout_model_path=None, variant='M'):
    """Layout detection via PyTorch PPDocLayout.

    Args:
        layout_model_path: Path to .pth model file. Defaults to ptocr_ppdoclayout_m.pth.
        variant: Model variant ('S', 'M', or 'L'). Must match the weights file.

    Returns:
        PPDocLayout model in eval mode.
    """
    from ptstructure.layout.picodet import PPDocLayout
    if layout_model_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        layout_model_path = os.path.join(repo_root, 'models/structurev3/ptocr_ppdoclayout_m.pth')
    model = PPDocLayout(variant=variant)
    model.eval()
    model.load_state_dict(torch.load(layout_model_path, map_location='cpu', weights_only=True), strict=False)
    return model


def detect_layout(layout_model, img, score_thresh=0.3, nms_thresh=0.5):
    """Run layout detection using PyTorch PPDocLayout model.

    Args:
        layout_model: PPDocLayout instance.
        img: BGR image as numpy array (H, W, 3).
        score_thresh: minimum score threshold.
        nms_thresh: IoU threshold for NMS.

    Returns:
        List of dicts: [{label, score, bbox:[x1,y1,x2,y2]}].
    """
    return layout_model.detect(img, score_thresh=score_thresh, nms_thresh=nms_thresh)


def load_ocr_system(det_model_path, det_yaml_path, rec_model_path, rec_yaml_path,
                     rec_char_dict_path, use_angle_cls=False, cls_model_path=None,
                     cls_yaml_path=None, cls_image_shape='3,48,192',
                     cls_thresh=0.9, cls_batch_num=6, label_list=None):
    """OCR text detection + recognition via PP-OCRv6 (TextSystem)."""
    from tools.infer.predict_system import TextSystem
    args = argparse.Namespace(
        # Det params
        det_algorithm='DB', use_gpu=False,
        det_model_path=det_model_path, det_yaml_path=det_yaml_path,
        det_limit_side_len=960, det_limit_type='max',
        det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5,
        use_dilation=False, det_db_score_mode='fast', det_box_type='quad',
        det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2,
        det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_sast_polygon=False,
        det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16,
        det_pse_box_type='quad', det_pse_scale=1,
        scales=[8,16,32], alpha=1.0, beta=1.0, fourier_degree=5, det_fce_box_type='poly',
        # Rec params
        rec_algorithm='CRNN',
        rec_model_path=rec_model_path, rec_yaml_path=rec_yaml_path,
        rec_image_shape='3,48,320', rec_char_type='ch', rec_batch_num=6,
        max_text_length=25, rec_char_dict_path=rec_char_dict_path,
        use_space_char=True, limited_max_width=1280, limited_min_width=16,
        rec_image_inverse=False,
        # Cls params
        use_angle_cls=use_angle_cls,
        cls_model_path=cls_model_path, cls_yaml_path=cls_yaml_path,
        cls_image_shape=cls_image_shape, cls_thresh=cls_thresh,
        cls_batch_num=cls_batch_num, label_list=label_list or ['0', '180'],
        # System params
        drop_score=0.5, save_crop_res=False, crop_res_save_dir='',
    )
    return TextSystem(args)


def load_table_recognizer(table_model_path):
    """Table structure recognition via PyTorch SLANeXt (verified 100% precision)."""
    from ptstructure.table.slanext import SLANeXt, SLANeXtConfig, SLANeXtVisionConfig
    model = SLANeXt(SLANeXtConfig(vision_config=SLANeXtVisionConfig()))
    model.eval()
    model.load_state_dict(torch.load(table_model_path, map_location='cpu', weights_only=True))
    return model


# ============================================================
# Global OCR approach (aligned with PaddleX PP-StructureV3)
# Step 1: Run OCR once on the full image → all text boxes + texts
# Step 2: Filter OCR results by layout region bbox intersection
# ============================================================

def global_ocr(ocr_system, img):
    """Run OCR on the full image, return list of {text, confidence, text_region}."""
    filter_boxes, filter_rec_res, _ = ocr_system(img)
    if filter_rec_res is None:
        return []
    results = []
    for box, rec_res in zip(filter_boxes, filter_rec_res):
        text, conf = rec_res[0], rec_res[1]
        if not text or conf < 0.5:
            continue
        # box is a quad polygon: (4,2) → get bounding rect
        x1, y1 = box.min(axis=0)
        x2, y2 = box.max(axis=0)
        results.append({
            'text': text,
            'confidence': float(conf),
            'text_region': [float(x1), float(y1), float(x2), float(y2)],
        })
    return results


def _has_intersection(rect1, rect2):
    """Check if two rectangles intersect."""
    x_min1, y_min1, x_max1, y_max1 = rect1
    x_min2, y_min2, x_max2, y_max2 = rect2
    if x_min1 > x_max2 or x_max1 < x_min2:
        return False
    if y_min1 > y_max2 or y_max1 < y_min2:
        return False
    return True


def filter_text_by_bbox(text_res, bbox):
    """Filter global OCR results to those intersecting with a layout region bbox.

    Args:
        text_res: list of {text, confidence, text_region}
        bbox: [x1, y1, x2, y2]

    Returns:
        Filtered list of {text, confidence, text_region}
    """
    return [r for r in text_res if _has_intersection(bbox, r['text_region'])]


def format_text_res(text_res_list):
    """Format filtered text results into a single string."""
    texts = [r['text'] for r in text_res_list if r['text']]
    return ' '.join(texts)


# Legacy per-crop OCR — only used for seal text detection internally
# (seal detector finds text boxes within seal crop, then OCRs each box)
def _ocr_crop(ocr_system, crop):
    """Run OCR on a single crop, return text string."""
    if crop.size == 0:
        return ''
    _, rec_res, _ = ocr_system(crop)
    if rec_res is None:
        return ''
    texts = [text for text, score in rec_res if text and score > 0.5]
    return ' '.join(texts)


def recognize_table(table_model, crop):
    """Run SLANeXt table structure recognition."""
    if crop.size == 0:
        return ''
    h, w = crop.shape[:2]
    s = max(512 / max(h, w), 0.01)
    nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(crop, (nw, nh))
    canvas = np.zeros((512, 512, 3), dtype=np.float32)
    canvas[:nh, :nw] = resized
    canvas = (canvas / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    t = torch.from_numpy(canvas.transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        output = table_model(t)
    tokens = output[0].argmax(dim=-1).cpu().tolist()
    # Character dict matching SLANeXt training vocabulary
    tdict = ['sos', '<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>',
             '<td>', '<td', '>', '</td>'] + \
            [' colspan="{}"'.format(i) for i in range(2, 21)] + \
            [' rowspan="{}"'.format(i) for i in range(2, 21)] + ['eos']
    chars = []
    for tk in tokens:
        if tk == 0: continue
        if tk >= len(tdict) - 1: break
        chars.append(tdict[tk])
    return '<table>{}</table>'.format(''.join(chars))


def load_formula_recognizer(formula_model_path=None, variant='M', formula_tokenizer_path=None):
    """Load PP-FormulaNet formula recognition model.

    Args:
        formula_model_path: Path to .pth model file.
        variant: Model variant ('S' or 'M').
        formula_tokenizer_path: Path to tokenizer.json.

    Returns:
        FormulaRecognizer instance.
    """
    from ptstructure.formula.ppformulanet import FormulaRecognizer

    if formula_model_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        formula_model_path = os.path.join(repo_root, 'models/structurev3/ptocr_formulanet_m.pth')

    recognizer = FormulaRecognizer(variant=variant)
    recognizer.load_weights(formula_model_path, formula_tokenizer_path)
    return recognizer


def recognize_formula(formula_recognizer, crop):
    """Run PP-FormulaNet recognition on a formula region.

    Args:
        formula_recognizer: FormulaRecognizer instance.
        crop: BGR image crop of formula region.

    Returns:
        LaTeX formula string.
    """
    if crop.size == 0:
        return ''
    return formula_recognizer.recognize(crop)


def load_seal_detector(seal_model_path=None, seal_yaml_path=None):
    """Load seal text detection model.

    Args:
        seal_model_path: Path to .pth model file.
        seal_yaml_path: Path to YAML config.

    Returns:
        SealDetector instance.
    """
    from ptstructure.seal.seal_det import SealDetector

    if seal_model_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        seal_model_path = os.path.join(repo_root, 'models/structurev3/ptocr_seal_det.pth')

    detector = SealDetector()
    detector.load_weights(seal_model_path, seal_yaml_path)
    return detector


def detect_seal_text(seal_detector, ocr_system, crop):
    """Detect and recognize text in a seal region.

    Args:
        seal_detector: SealDetector instance.
        ocr_system: TextSystem instance for OCR.
        crop: BGR image crop of seal region.

    Returns:
        Recognized seal text string.
    """
    if crop.size == 0:
        return ''

    # Detect text regions in seal
    boxes, scores = seal_detector.detect(crop)

    if not boxes:
        return '[seal]'

    # OCR each detected text region
    texts = []
    for box in boxes:
        x1, y1, x2, y2 = box
        text_crop = crop[y1:y2, x1:x2]
        if text_crop.size > 0:
            text = _ocr_crop(ocr_system, text_crop)
            if text:
                texts.append(text)

    if texts:
        return '[seal: {}]'.format(' '.join(texts))
    return '[seal]'


def main():
    parser = argparse.ArgumentParser(description='PP-StructureV3 Document Structure Analysis')
    parser.add_argument('--image_dir', type=str, required=True, help='Input image file or directory.')
    parser.add_argument('--output_dir', type=str, default='./output/ppstructurev3/', help='Output directory.')

    # Preprocessing: document orientation correction
    parser.add_argument('--use_doc_orientation', action='store_true', default=False,
                        help='Enable document image orientation classification (doc_ori).')
    parser.add_argument('--doc_orientation_model_path', type=str, default=None,
                        help='Path to doc_ori .pth model.')
    parser.add_argument('--doc_orientation_yaml_path', type=str, default=None,
                        help='Path to doc_ori YAML config.')

    # Preprocessing: document unwarping
    parser.add_argument('--use_doc_unwarping', action='store_true', default=False,
                        help='Enable document image unwarping (UVDoc).')
    parser.add_argument('--doc_unwarping_model_path', type=str, default=None,
                        help='Path to UVDoc .pth model.')

    # OCR: text line orientation classification
    parser.add_argument('--use_angle_cls', action='store_true', default=False,
                        help='Enable text line orientation classification (textline_ori).')
    parser.add_argument('--cls_model_path', type=str, default=None,
                        help='Path to cls .pth model.')
    parser.add_argument('--cls_yaml_path', type=str, default=None,
                        help='Path to cls YAML config.')

    # Layout detection
    parser.add_argument('--layout_model_path', type=str,
                        default=None, help='Path to layout detection .pth model.')
    parser.add_argument('--layout_variant', type=str, default='S',
                        choices=['S', 'M', 'L'], help='Layout model variant (S/M/L).')

    # OCR models
    parser.add_argument('--det_model_path', type=str,
                        default='../models/v6/ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth')
    parser.add_argument('--det_yaml_path', type=str,
                        default='configs/det/PP-OCRv6/PP-OCRv6_small_det.yml')
    parser.add_argument('--rec_model_path', type=str,
                        default='../models/v6/ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth')
    parser.add_argument('--rec_yaml_path', type=str,
                        default='configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml')
    parser.add_argument('--rec_char_dict_path', type=str,
                        default='pytorchocr/utils/dict/ppocrv6_dict.txt')
    parser.add_argument('--table_model_path', type=str,
                        default='models/structurev3/ptocr_slanext_wired.pth')
    parser.add_argument('--layout_score_thresh', type=float, default=0.3,
                        help='Layout detection score threshold.')
    parser.add_argument('--layout_nms_thresh', type=float, default=0.5,
                        help='Layout detection NMS threshold.')

    # Formula recognition
    parser.add_argument('--use_formula', action='store_true', default=False,
                        help='Enable formula recognition (PP-FormulaNet).')
    parser.add_argument('--formula_model_path', type=str, default=None,
                        help='Path to formula recognition .pth model.')
    parser.add_argument('--formula_variant', type=str, default='M',
                        choices=['S', 'M'], help='Formula model variant (S/M).')
    parser.add_argument('--formula_tokenizer_path', type=str, default=None,
                        help='Path to UniMERNet tokenizer.json.')

    # Seal detection
    parser.add_argument('--use_seal', action='store_true', default=False,
                        help='Enable seal text detection.')
    parser.add_argument('--seal_model_path', type=str, default=None,
                        help='Path to seal detection .pth model.')
    parser.add_argument('--seal_yaml_path', type=str, default=None,
                        help='Path to seal detection YAML config.')
    parser.add_argument('--seal_det_thresh', type=float, default=0.1,
                        help='Seal detection pixel threshold (lower = more sensitive).')
    parser.add_argument('--seal_det_box_thresh', type=float, default=0.3,
                        help='Seal detection box score threshold.')
    parser.add_argument('--seal_det_unclip_ratio', type=float, default=0.3,
                        help='Seal detection unclip ratio.')
    args = parser.parse_args()

    # Collect images
    if os.path.isfile(args.image_dir):
        image_list = [args.image_dir]
    else:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_list = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                             if os.path.splitext(f)[1].lower() in exts])
    logger.info('Found {} image(s)'.format(len(image_list)))

    # Load models
    logger.info('Loading models...')
    t0 = time.time()

    # Preprocessing: doc orientation
    doc_ori = None
    if args.use_doc_orientation:
        from ptstructure.doc_preprocess.doc_orientation import DocOrientationClassifier
        doc_ori = DocOrientationClassifier(device='cpu')
        doc_ori.load_weights(args.doc_orientation_model_path, args.doc_orientation_yaml_path)

    # Preprocessing: doc unwarping
    doc_unwarp = None
    if args.use_doc_unwarping:
        from ptstructure.doc_preprocess.unwarp import UVDocUnwarper
        doc_unwarp = UVDocUnwarper(device='cpu')
        doc_unwarp.load_weights(args.doc_unwarping_model_path)

    # Layout detection
    layout_det = load_layout_detector(args.layout_model_path, variant=args.layout_variant)

    # OCR with optional textline orientation
    ocr_sys = load_ocr_system(
        args.det_model_path, args.det_yaml_path,
        args.rec_model_path, args.rec_yaml_path,
        args.rec_char_dict_path,
        use_angle_cls=args.use_angle_cls,
        cls_model_path=args.cls_model_path,
        cls_yaml_path=args.cls_yaml_path,
    )

    # Table recognition
    table_rec = load_table_recognizer(args.table_model_path)

    formula_rec = None
    if args.use_formula:
        formula_rec = load_formula_recognizer(
            args.formula_model_path, args.formula_variant, args.formula_tokenizer_path)

    seal_det = None
    if args.use_seal:
        from ptstructure.seal.seal_det import SealDetector
        seal_det = SealDetector(
            det_db_thresh=args.seal_det_thresh,
            det_db_box_thresh=args.seal_det_box_thresh,
            det_db_unclip_ratio=args.seal_det_unclip_ratio,
        )
        seal_det.load_weights(args.seal_model_path, args.seal_yaml_path)

    n_models = 3 + (1 if args.use_formula else 0) + (1 if args.use_seal else 0)
    n_models += (1 if args.use_doc_orientation else 0) + (1 if args.use_doc_unwarping else 0)
    n_models += (1 if args.use_angle_cls else 0)
    logger.info('{} models loaded in {:.1f}s'.format(n_models, time.time() - t0))

    from ptstructure.utils.reading_order import recover_reading_order
    from ptstructure.utils.markdown import blocks_to_markdown, blocks_to_json
    from ptstructure.utils.visualize import draw_structure_result

    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in image_list:
        logger.info('\nProcessing: {}'.format(os.path.basename(img_path)))
        img = cv2.imread(img_path)
        if img is None:
            logger.error('Cannot read: {}'.format(img_path))
            continue
        oh, ow = img.shape[:2]
        t_start = time.time()

        # 0. Preprocessing: document orientation correction
        if doc_ori is not None:
            img = doc_ori.correct_orientation(img)

        # 0b. Preprocessing: document unwarping
        if doc_unwarp is not None:
            img = doc_unwarp.unwarp(img)

        # 1. Layout Detection (PyTorch PPDocLayout)
        layout_boxes = detect_layout(layout_det, img,
                                      score_thresh=args.layout_score_thresh,
                                      nms_thresh=args.layout_nms_thresh)
        logger.info('  Layout: {} regions'.format(len(layout_boxes)))

        # 2. Global OCR on full image (aligned with PaddleX PP-StructureV3)
        text_res = global_ocr(ocr_sys, img)
        logger.info('  Global OCR: {} text boxes'.format(len(text_res)))

        # 3. Process each layout region
        blocks = []
        for i, lb in enumerate(layout_boxes):
            label = lb['label']
            x1, y1, x2, y2 = [max(0, int(v)) for v in lb['bbox']]
            x1, y1 = min(x1, ow - 1), min(y1, oh - 1)
            x2, y2 = min(x2, ow), min(y2, oh)
            crop = img[y1:y2, x1:x2]
            bbox = [x1, y1, x2, y2]
            content = ''

            if label == 'table' and crop.size > 0:
                content = recognize_table(table_rec, crop)
            elif label == 'formula' and formula_rec is not None and crop.size > 0:
                latex = recognize_formula(formula_rec, crop)
                content = '$${}$$'.format(latex) if latex else '[formula]'
            elif label == 'seal' and crop.size > 0:
                if seal_det is not None:
                    content = detect_seal_text(seal_det, ocr_sys, crop)
                # Fallback: filter global OCR text by seal bbox
                if (not content or content == '[seal]'):
                    filtered = filter_text_by_bbox(text_res, bbox)
                    text = format_text_res(filtered)
                    content = '[seal: {}]'.format(text) if text else '[seal]'
            elif label in ('figure', 'chart', 'image'):
                content = '[{}]'.format(label)
            else:
                # Filter global OCR results by this region's bbox
                filtered = filter_text_by_bbox(text_res, bbox)
                content = format_text_res(filtered)

            blocks.append({'block_id': i, 'block_label': label,
                           'block_bbox': bbox,
                           'block_content': content,
                           'confidence': lb['score'], 'block_order': 0})

        # 3. Reading Order + Output
        blocks = recover_reading_order(blocks)
        md = blocks_to_markdown(blocks)
        js = blocks_to_json(blocks, img_path)

        base = os.path.splitext(os.path.basename(img_path))[0]
        with open(os.path.join(args.output_dir, '{}_v3.md'.format(base)), 'w') as f:
            f.write(md)
        with open(os.path.join(args.output_dir, '{}_v3.json'.format(base)), 'w') as f:
            f.write(js)
        vis = draw_structure_result(img, blocks)
        cv2.imwrite(os.path.join(args.output_dir, '{}_v3.jpg'.format(base)), vis)

        n_text = sum(1 for b in blocks if b['block_content'])
        n_table = sum(1 for b in blocks if b['block_label'] == 'table')
        n_formula = sum(1 for b in blocks if b['block_label'] == 'formula' and b['block_content'].startswith('$$'))
        n_seal = sum(1 for b in blocks if b['block_label'] == 'seal')
        logger.info('  Blocks: {} ({} text, {} tables, {} formulas, {} seals) in {:.1f}s'.format(
            len(blocks), n_text, n_table, n_formula, n_seal, time.time() - t_start))
        if md:
            logger.info('  Markdown preview:\n{}'.format(md[:300]))

    logger.info('\nDone! Output: {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
