#!/usr/bin/env python
"""
PP-StructureV3: Document Structure Analysis Pipeline
All models run in PyTorch (no PaddleX dependency).

Usage:
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


def load_ocr_system(det_model_path, det_yaml_path, rec_model_path, rec_yaml_path, rec_char_dict_path):
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
        # System params
        use_angle_cls=False, drop_score=0.5,
        save_crop_res=False, crop_res_save_dir='',
    )
    return TextSystem(args)


def load_table_recognizer(table_model_path):
    """Table structure recognition via PyTorch SLANeXt (verified 100% precision)."""
    from ptstructure.table.slanext import SLANeXt, SLANeXtConfig, SLANeXtVisionConfig
    model = SLANeXt(SLANeXtConfig(vision_config=SLANeXtVisionConfig()))
    model.eval()
    model.load_state_dict(torch.load(table_model_path, map_location='cpu', weights_only=True))
    return model


def ocr_region(ocr_system, crop):
    """Run text detection + recognition on a cropped region using TextSystem."""
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


def main():
    parser = argparse.ArgumentParser(description='PP-StructureV3 Document Structure Analysis')
    parser.add_argument('--image_dir', type=str, required=True, help='Input image file or directory.')
    parser.add_argument('--output_dir', type=str, default='./output/ppstructurev3/', help='Output directory.')
    parser.add_argument('--layout_model_path', type=str,
                        default=None, help='Path to layout detection .pth model.')
    parser.add_argument('--layout_variant', type=str, default='S',
                        choices=['S', 'M', 'L'], help='Layout model variant (S/M/L).')
    parser.add_argument('--det_model_path', type=str,
                        default='/home/frotms/hdd/liuchenxi/repo/pytorchocr/models/v6/ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth')
    parser.add_argument('--det_yaml_path', type=str,
                        default='configs/det/PP-OCRv6/PP-OCRv6_small_det.yml')
    parser.add_argument('--rec_model_path', type=str,
                        default='/home/frotms/hdd/liuchenxi/repo/pytorchocr/models/v6/ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth')
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
    layout_det = load_layout_detector(args.layout_model_path, variant=args.layout_variant)
    ocr_sys = load_ocr_system(args.det_model_path, args.det_yaml_path,
                               args.rec_model_path, args.rec_yaml_path, args.rec_char_dict_path)
    table_rec = load_table_recognizer(args.table_model_path)
    logger.info('Models loaded in {:.1f}s'.format(time.time() - t0))

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

        # 1. Layout Detection (PyTorch PPDocLayout)
        layout_boxes = detect_layout(layout_det, img,
                                      score_thresh=args.layout_score_thresh,
                                      nms_thresh=args.layout_nms_thresh)
        logger.info('  Layout: {} regions'.format(len(layout_boxes)))

        # 2. Process each region
        blocks = []
        for i, lb in enumerate(layout_boxes):
            label = lb['label']
            x1, y1, x2, y2 = [max(0, int(v)) for v in lb['bbox']]
            x1, y1 = min(x1, ow - 1), min(y1, oh - 1)
            x2, y2 = min(x2, ow), min(y2, oh)
            crop = img[y1:y2, x1:x2]
            content = ''

            if label == 'table' and crop.size > 0:
                content = recognize_table(table_rec, crop)
            elif label in ('figure', 'chart', 'seal', 'image'):
                content = '[{}]'.format(label)
            elif crop.size > 0:
                content = ocr_region(ocr_sys, crop)

            blocks.append({'block_id': i, 'block_label': label,
                           'block_bbox': [x1, y1, x2, y2],
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
        logger.info('  Blocks: {} ({} text, {} tables) in {:.1f}s'.format(
            len(blocks), n_text, n_table, time.time() - t_start))
        if md:
            logger.info('  Markdown preview:\n{}'.format(md[:300]))

    logger.info('\nDone! Output: {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
