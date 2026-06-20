"""
Layout detection post-processing utilities.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union


# PP-DocLayout-M 23-class label taxonomy (from PaddleX inference.yml)
LAYOUT_LABELS = [
    'paragraph_title',  # 0
    'image',            # 1
    'text',             # 2
    'number',           # 3
    'abstract',         # 4
    'content',          # 5
    'figure_title',     # 6
    'formula',          # 7
    'table',            # 8
    'table_title',      # 9
    'reference',        # 10
    'doc_title',        # 11
    'footnote',         # 12
    'header',           # 13
    'algorithm',        # 14
    'footer',           # 15
    'seal',             # 16
    'chart_title',      # 17
    'chart',            # 18
    'formula_number',   # 19
    'header_image',     # 20
    'footer_image',     # 21
    'aside_text',       # 22
]

def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> List[int]:
    """Non-Maximum Suppression.

    Args:
        boxes: [N, 4] array of [x1, y1, x2, y2].
        scores: [N] array of confidence scores.
        threshold: IoU threshold for suppression.

    Returns:
        Indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


class LayoutPostProcess:
    """Layout detection post-processor.

    Converts raw model outputs to formatted layout detection results.
    """
    def __init__(
        self,
        labels: Optional[List[str]] = None,
        scale_size: Tuple[int, int] = (800, 800),
        threshold: float = 0.5,
        layout_nms: bool = True,
        nms_threshold: float = 0.5,
        unclip_ratio: Tuple[float, float] = (1.0, 1.0),
    ):
        self.labels = labels or LAYOUT_LABELS
        self.scale_size = scale_size
        self.threshold = threshold
        self.layout_nms = layout_nms
        self.nms_threshold = nms_threshold
        self.unclip_ratio_h, self.unclip_ratio_w = (
            unclip_ratio if isinstance(unclip_ratio, (tuple, list))
            else (unclip_ratio, unclip_ratio)
        )

    def __call__(
        self,
        preds: List[Dict],
        datas: List[Dict],
        threshold: Optional[float] = None,
    ) -> List[List[Dict]]:
        """Process detection predictions.

        Args:
            preds: List of per-image prediction dicts with 'boxes' array.
            datas: List of per-image data dicts with 'ori_img' and shape info.
            threshold: Override default score threshold.

        Returns:
            List of per-image lists of detection results.
        """
        if threshold is None:
            threshold = self.threshold

        results = []
        for idx, pred in enumerate(preds):
            data = datas[idx] if idx < len(datas) else {}
            ori_img = data.get('ori_img')
            img_shape = ori_img.shape if ori_img is not None else (800, 800)

            boxes = pred.get('boxes', None)
            if boxes is None or len(boxes) == 0:
                results.append([])
                continue

            # boxes format: [class_id, score, x1, y1, x2, y2, ...]
            det_results = []
            if isinstance(boxes, np.ndarray):
                labels_arr = boxes[:, 0].astype(int)
                scores = boxes[:, 1]
                bboxes = boxes[:, 2:6]
            else:
                labels_arr = np.array([b[0] for b in boxes], dtype=int)
                scores = np.array([b[1] for b in boxes])
                bboxes = np.array([b[2:6] for b in boxes])

            # Filter by threshold
            valid = scores > threshold
            labels_arr = labels_arr[valid]
            scores = scores[valid]
            bboxes = bboxes[valid]

            if len(bboxes) == 0:
                results.append([])
                continue

            # Apply NMS per class
            if self.layout_nms:
                keep_indices = []
                unique_labels = np.unique(labels_arr)
                for label in unique_labels:
                    cls_mask = labels_arr == label
                    cls_boxes = bboxes[cls_mask]
                    cls_scores = scores[cls_mask]
                    cls_keep = nms(cls_boxes, cls_scores, self.nms_threshold)
                    orig_indices = np.where(cls_mask)[0][cls_keep]
                    keep_indices.extend(orig_indices.tolist())

                labels_arr = labels_arr[keep_indices]
                scores = scores[keep_indices]
                bboxes = bboxes[keep_indices]

            # Scale back to original image size
            scale_h = img_shape[0] / self.scale_size[1]
            scale_w = img_shape[1] / self.scale_size[0]
            bboxes[:, [0, 2]] *= scale_w
            bboxes[:, [1, 3]] *= scale_h

            for i in range(len(bboxes)):
                label_idx = int(labels_arr[i])
                label_name = self.labels[label_idx] if label_idx < len(self.labels) else 'unknown'
                score = float(scores[i])
                bbox = bboxes[i].tolist()

                det_results.append({
                    'label': label_name,
                    'label_id': label_idx,
                    'score': score,
                    'bbox': [int(b) for b in bbox],
                })

            results.append(det_results)

        return results
