import numpy as np
import torch



# 相交矩形的面积
def intersect(box_a, box_b):
    """计算两组矩形两两之间相交区域的面积
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        ious: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]
    union = area_a + area_b - inter
    return inter / union  # [A, B]



def _matrix_nms(bboxes, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class bboxes.
    Args:
        bboxes (Tensor): shape (n, 4)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gaussian'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
    iou_matrix = jaccard(bboxes, bboxes)   # shape: [n_samples, n_samples]
    iou_matrix = iou_matrix.triu(diagonal=1)   # 只取上三角部分

    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)   # shape: [n_samples, n_samples]
    # 第i行第j列表示的是第i个预测框和第j个预测框的类别id是否相同。我们抑制的是同类的预测框。
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)   # shape: [n_samples, n_samples]

    # IoU compensation
    # 非同类的iou置为0，同类的iou保留。逐列取最大iou
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)   # shape: [n_samples, ]
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)   # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    decay_iou = iou_matrix * label_matrix   # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # 更新分数
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update




def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.):
    inds = (scores > score_threshold)
    cate_scores = scores[inds]
    if len(cate_scores) == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0

    inds = inds.nonzero()
    cate_labels = inds[:, 1]
    bboxes = bboxes[inds[:, 0]]

    # sort and keep top nms_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if nms_top_k > 0 and len(sort_inds) > nms_top_k:
        sort_inds = sort_inds[:nms_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    # Matrix NMS
    kernel = 'gaussian' if use_gaussian else 'linear'
    cate_scores = _matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)

    # filter.
    keep = cate_scores >= post_threshold
    if keep.sum() == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0
    bboxes = bboxes[keep, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # sort and keep keep_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    cate_scores = cate_scores.unsqueeze(1)
    cate_labels = cate_labels.unsqueeze(1).float()

    # pred = torch.cat([cate_labels, cate_scores, bboxes], 1)
    # return pred
    return bboxes, cate_scores, cate_labels.type(torch.int32)


# yolo_box
# https://github.com/miemie2013/Pytorch-PPYOLO/blob/master/model/head.py
def yolo_box(conv_output, anchors, stride, num_classes, scale_x_y, im_size, clip_bbox, conf_thresh, use_gpu=False):
    conv_output = conv_output.permute(0, 2, 3, 1)
    conv_shape       = conv_output.shape
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = conv_output.reshape((batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    rows = torch.arange(0, output_size, dtype=torch.float32, device=conv_raw_dxdy.device)
    cols = torch.arange(0, output_size, dtype=torch.float32, device=conv_raw_dxdy.device)
    rows = rows[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis].repeat((1, output_size, 1, 1, 1))
    cols = cols[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis].repeat((1, 1, output_size, 1, 1))
    offset = torch.cat([rows, cols], dim=-1)
    offset = offset.repeat((batch_size, 1, 1, anchor_per_scale, 1))
    # Grid Sensitive
    pred_xy = (scale_x_y * torch.sigmoid(conv_raw_dxdy) + offset - (scale_x_y - 1.0) * 0.5 ) * stride

    # _anchors = torch.Tensor(anchors, device=conv_raw_dxdy.device)   # RuntimeError: legacy constructor for device type: cpu was passed device type: cuda, but device type must be: cpu
    _anchors = torch.Tensor(anchors)
    if use_gpu and torch.cuda.is_available():
        _anchors = _anchors.cuda()
    pred_wh = (torch.exp(conv_raw_dwdh) * _anchors)

    pred_xyxy = torch.cat([pred_xy - pred_wh / 2, pred_xy + pred_wh / 2], dim=-1)   # 左上角xy + 右下角xy
    pred_conf = torch.sigmoid(conv_raw_conf)
    # mask = (pred_conf > conf_thresh).float()
    pred_prob = torch.sigmoid(conv_raw_prob)
    pred_scores = pred_conf * pred_prob
    # pred_scores = pred_scores * mask
    # pred_xyxy = pred_xyxy * mask

    # paddle中实际的顺序
    # pred_xyxy = pred_xyxy.permute(0, 3, 1, 2, 4)
    # pred_scores = pred_scores.permute(0, 3, 1, 2, 4)

    pred_xyxy = pred_xyxy.reshape((batch_size, output_size*output_size*anchor_per_scale, 4))
    pred_scores = pred_scores.reshape((batch_size, pred_xyxy.shape[1], num_classes))
    im_size = torch.from_numpy(im_size)
    _im_size_h = im_size[:, 0:1]
    _im_size_w = im_size[:, 1:2]
    _im_size = torch.cat([_im_size_w, _im_size_h], 1)
    _im_size = _im_size.unsqueeze(1)
    _im_size = _im_size.repeat((1, pred_xyxy.shape[1], 1))

    pred_x0y0 = pred_xyxy[:, :, 0:2] / output_size / stride * _im_size
    pred_x1y1 = pred_xyxy[:, :, 2:4] / output_size / stride * _im_size
    if clip_bbox:
        x0 = pred_x0y0[:, :, 0:1]
        y0 = pred_x0y0[:, :, 1:2]
        x1 = pred_x1y1[:, :, 0:1]
        y1 = pred_x1y1[:, :, 1:2]
        x0 = torch.where(x0 < 0, x0 * 0, x0)
        y0 = torch.where(y0 < 0, y0 * 0, y0)
        x1 = torch.where(x1 > _im_size[:, :, 0:1], _im_size[:, :, 0:1], x1)
        y1 = torch.where(y1 > _im_size[:, :, 1:2], _im_size[:, :, 1:2], y1)
        pred_xyxy = torch.cat([x0, y0, x1, y1], -1)
    else:
        pred_xyxy = torch.cat([pred_x0y0, pred_x1y1], -1)
    return pred_xyxy, pred_scores
