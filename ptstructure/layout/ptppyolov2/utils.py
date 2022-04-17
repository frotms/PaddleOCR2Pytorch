import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _decode(feature, anchors, class_num, image_size):
    anchor_num = len(anchors)
    map_deep = class_num + 5
    batch_size = feature.shape[0]
    feature_h = feature.shape[1]
    feature_w = feature.shape[2]

    scale_h = image_size[1] / feature_h
    scale_w = image_size[0] / feature_w

    scaled_anchors = [(a_w / scale_w, a_h / scale_h) for a_w, a_h in anchors]

    # prediction  = feature.reshape(batch_size, anchor_num, map_deep, feature_h, feature_w)
    prediction = feature.reshape(batch_size, feature_h, feature_w, anchor_num, map_deep)
    prediction  = prediction.transpose([0, 1, 3, 4, 2])

    x = sigmoid(prediction[..., 0])
    y = sigmoid(prediction[..., 1])
    w = prediction[..., 2]
    h = prediction[..., 3]

    conf = sigmoid(prediction[..., 4])
    prob = sigmoid(prediction[..., 5:])

    grid_x = np.arange(feature_w).reshape(1, feature_w).repeat(feature_h, 0)[np.newaxis, ...] \
        .repeat(batch_size * anchor_num, 0).transpose(1, 2, 0)[np.newaxis, ...].astype(np.float32)
    grid_y = np.arange(feature_h).reshape(feature_h, 1).repeat(feature_w, 1)[np.newaxis, ...] \
        .repeat(batch_size * anchor_num, 0).transpose(1, 2, 0)[np.newaxis, ...].astype(np.float32)

    anchor_w = np.array(scaled_anchors)[:, :1]
    anchor_h = np.array(scaled_anchors)[:, 1:]
    anchor_w = anchor_w.repeat(batch_size, 1)[..., np.newaxis].repeat(feature_w * feature_h, -1).transpose(1, 2,
                                                                                                           0).reshape(
        w.shape)
    anchor_h = anchor_h.repeat(batch_size, 1)[..., np.newaxis].repeat(feature_w * feature_h, -1).transpose(1, 2,
                                                                                                           0).reshape(
        h.shape)

    pred_boxes = np.zeros_like(prediction[..., :4])
    w = np.exp(w)
    h = np.exp(h)
    width = np.where(w > 50, 50, w) * anchor_w
    height = np.where(h > 50, 50, h) * anchor_h
    pred_boxes[..., 0] = x + grid_x - width / 2.
    pred_boxes[..., 1] = y + grid_y - height / 2.
    pred_boxes[..., 2] = x + grid_x + width / 2.
    pred_boxes[..., 3] = y + grid_y + height / 2.

    _scale = np.array([scale_w, scale_h] * 2, dtype=np.float32)
    output = np.concatenate((pred_boxes.reshape(batch_size, -1, 4) * _scale,
                             conf.reshape(batch_size, -1, 1),
                             prob.reshape(batch_size, -1, class_num)),
                            axis=-1)

    return output


def decode_process(features, anchors, class_num, image_size):
    # _anchors = [anchors[6:9], anchors[3:6], anchors[0:3]]
    _anchors = [anchors[0:3], anchors[3:6], anchors[6:9]]
    outputs = []
    for feature, anchor in zip(features, _anchors):
        outputs.append(_decode(feature, anchor, class_num, image_size))
    output = np.concatenate(outputs, 1)
    bboxes, confs, probs = np.split(output, [4, 5], axis=-1)
    return bboxes, confs, probs

def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
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

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=30, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0:
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label



# yolov4
# -*- coding: utf-8 -*-
import random
import colorsys
import cv2
import threading
import os
import numpy as np


class Decode(object):
    def __init__(self, obj_threshold, nms_threshold, input_shape, _yolo, all_classes, scale_x_y=1.0):
        self._t1 = obj_threshold
        self._t2 = nms_threshold
        self.input_shape = input_shape
        self.all_classes = all_classes
        self.num_classes = len(self.all_classes)
        self.scale_x_y = scale_x_y
        self._yolo = _yolo

    # 处理一张图片
    def detect_image(self, image, draw_image):
        pimage = self.process_image(np.copy(image))

        boxes, scores, classes = self.predict(pimage, image.shape)
        if boxes is not None and draw_image:
            self.draw(image, boxes, scores, classes)
        return image, boxes, scores, classes

    # 多线程后处理
    def multi_thread_post(self, batch_img, outs, i, draw_image, result_image, result_boxes, result_scores, result_classes):
        a1 = np.reshape(outs[0][i], (1, self.input_shape[0] // 32, self.input_shape[1] // 32, 3, 5 + self.num_classes))
        a2 = np.reshape(outs[1][i], (1, self.input_shape[0] // 16, self.input_shape[1] // 16, 3, 5 + self.num_classes))
        a3 = np.reshape(outs[2][i], (1, self.input_shape[0] // 8, self.input_shape[1] // 8, 3, 5 + self.num_classes))
        boxes, scores, classes = self._yolo_out([a1, a2, a3], batch_img[i].shape)
        if boxes is not None and draw_image:
            self.draw(batch_img[i], boxes, scores, classes)
        result_image[i] = batch_img[i]
        result_boxes[i] = boxes
        result_scores[i] = scores
        result_classes[i] = classes

    # 处理一批图片
    def detect_batch(self, batch_img, draw_image):
        batch_size = len(batch_img)
        result_image, result_boxes, result_scores, result_classes = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size
        batch = []

        for image in batch_img:
            pimage = self.process_image(np.copy(image))
            batch.append(pimage)
        batch = np.concatenate(batch, axis=0)
        outs = self._yolo.predict(batch)

        # 多线程
        threads = []
        for i in range(batch_size):
            t = threading.Thread(target=self.multi_thread_post, args=(
                batch_img, outs, i, draw_image, result_image, result_boxes, result_scores, result_classes))
            threads.append(t)
            t.start()
        # 等待所有线程任务结束。
        for t in threads:
            t.join()
        return result_image, result_boxes, result_scores, result_classes

    def draw(self, image, boxes, scores, classes):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[cl]
            # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            bbox_thick = 1
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    def process_image(self, img):
        h, w = img.shape[:2]
        scale_x = float(self.input_shape[1]) / w
        scale_y = float(self.input_shape[0]) / h
        resized_img = cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        pimage = rgb_img.astype(np.float32) / 255.
        pimage = np.expand_dims(pimage, axis=0)
        return pimage

    def pure_postprocess(self, yolo_infer_out, org_hw_shape, anchors):
        # numpy后处理
        a1 = np.reshape(yolo_infer_out[0], (1, self.input_shape[0] // 32, self.input_shape[1] // 32, 3, 5 + self.num_classes))
        a2 = np.reshape(yolo_infer_out[1], (1, self.input_shape[0] // 16, self.input_shape[1] // 16, 3, 5 + self.num_classes))
        a3 = np.reshape(yolo_infer_out[2], (1, self.input_shape[0] // 8, self.input_shape[1] // 8, 3, 5 + self.num_classes))
        boxes, scores, classes = self._yolo_out([a1, a2, a3], org_hw_shape, anchors)

        return boxes, scores, classes

    def predict(self, image, shape):
        outs = self._yolo.predict(image)

        # numpy后处理
        a1 = np.reshape(outs[0], (1, self.input_shape[0]//32, self.input_shape[1]//32, 3, 5+self.num_classes))
        a2 = np.reshape(outs[1], (1, self.input_shape[0]//16, self.input_shape[1]//16, 3, 5+self.num_classes))
        a3 = np.reshape(outs[2], (1, self.input_shape[0]//8, self.input_shape[1]//8, 3, 5+self.num_classes))
        boxes, scores, classes = self._yolo_out([a1, a2, a3], shape)

        return boxes, scores, classes


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _process_feats(self, out, anchors, mask, scale_x_y=1.0):
        grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])
        anchors = [anchors[i] for i in mask]
        anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

        # Reshape to batch, height, width, num_anchors, box_params.
        out = out[0]
        box_xy = self._sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4])
        box_wh = box_wh * anchors_tensor

        box_confidence = self._sigmoid(out[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = self._sigmoid(out[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_shape
        box_wh *= scale_x_y
        box_xy -= (box_wh / 2.)   # 坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self._t1)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _soft_nms(self, boxes, scores, iou_thresh=0.3, sigma=0.5, thresh=0.005, method=0):
        '''
        https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
        :param dets: [[x1, y1, x2, y2, score]，[x1, y1, x2, y2, score]，[x1, y1, x2, y2, score]]
        :param iou_thresh: iou thresh
        :param sigma: std of gaussian
        :param thresh: the last score thresh
        :param method: 1、linear 2、gaussian 3、originl nms
        :return: keep bboxes
        '''

        dets = np.concatenate([boxes, scores[:,None]], axis=1)

        N = dets.shape[0]  # the size of bboxes
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        for i in range(N):
            temp_box = dets[i, :4]
            temp_score = dets[i, 4]
            temp_area = areas[i]
            pos = i + 1

            if i != N - 1:
                maxscore = np.max(dets[:, 4][pos:])
                maxpos = np.argmax(dets[:, 4][pos:])
            else:
                maxscore = dets[:, 4][-1]
                maxpos = -1

            if temp_score < maxscore:
                dets[i, :4] = dets[maxpos + i + 1, :4]
                dets[maxpos + i + 1, :4] = temp_box

                dets[i, 4] = dets[maxpos + i + 1, 4]
                dets[maxpos + i + 1, 4] = temp_score

                areas[i] = areas[maxpos + i + 1]
                areas[maxpos + i + 1] = temp_area

            xx1 = np.maximum(x1[i], x1[pos:])
            xx2 = np.minimum(x2[i], x2[pos:])
            yy1 = np.maximum(y1[i], y1[pos:])
            yy2 = np.minimum(y2[i], y2[pos:])

            w = np.maximum(xx2 - xx1 + 1.0, 0.)
            h = np.maximum(yy2 - yy1 + 1.0, 0.)

            inters = w * h
            ious = inters / (areas[i] + areas[pos:] - inters)

            if method == 1: # linear
                weight = np.ones(ious.shape)
                weight[ious > iou_thresh] = weight[ious > iou_thresh] - ious[ious > iou_thresh]
            elif method == 2: # gaussian
                weight = np.exp(-ious * ious / sigma)
            else: # original nms
                weight = np.ones(ious.shape)
                weight[ious > iou_thresh] = 0

            dets[pos:, 4] = dets[pos:, 4] * weight

        inds = np.argwhere(dets[:, 4] > thresh)
        keep = inds.astype(int)[0]

        return keep


    def _nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._t2)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep


    def _yolo_out(self, outs, shape,
                  anchors=[[10, 13], [16, 30], [33, 23],
                           [30, 61], [62, 45], [59, 119],
                           [116, 90], [156, 198], [373, 326]],
                  ):
        masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # anchors = [[10, 13], [16, 30], [33, 23],
        #     [30, 61], [62, 45], [59, 119],
        #     [116, 90], [156, 198], [373, 326]]
        scale_x_y = self.scale_x_y
        boxes, classes, scores = [], [], []
        for out, mask in zip(outs, masks):
            b, c, s = self._process_feats(out, anchors, mask, scale_x_y=scale_x_y)
            b, c, s = self._filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        # boxes坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
        # Scale boxes back to original image shape.
        w, h = shape[1], shape[0]

        image_dims = [w, h, w, h]
        boxes = boxes * image_dims

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self._nms_boxes(b, s)
            # keep = self._soft_nms(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        # 换坐标
        boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

        return boxes, scores, classes


