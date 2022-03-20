import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import torch
import torch.nn as nn
from .ppyolov2_base import PPYOLOv2Base
from .utils import Decode
from .pt_utils import yolo_box, matrix_nms

ALL_CLASSES = [
'person', 'bicycle', 'car','motorcycle','airplane','bus','train', 'truck', 'boat','traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench','bird','cat','dog','horse','sheep',
'cow', 'elephant','bear','zebra','giraffe','hat','umbrella', 'handbag','tie','suitcase',
'frisbee','skis','snowboard','sports ball','kite', 'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
'bed','dining table','window','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
'oven', 'sink','refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','tooth brush',
               ]



class PPYOLOv2:
    def __init__(self, **kwargs):
        self._init(**kwargs)

    def _init(self, **kwargs):
        model_path = kwargs.get('INIT_model_path', 'vehicle_yolov3_darknet.pth')#'pedestrian_yolov3_darknet.pth')#'vehicle_yolov3_darknet.pth')
        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            raise FileNotFoundError('{} is not existed.'.format(model_path))
        use_gpu = kwargs.get('INIT_use_gpu', True)
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.input_shape = kwargs.get('INIT_det_input_shape', (640,640))
        self.conf_thresh = kwargs.get('INIT_det_thresh', 0.05)
        # self.nms_thresh = kwargs.get('INIT_det_nms_thresh', 0.45)
        self.keep_top_k = kwargs.get('INIT_det_keep_top_k', 100)
        self.score_threshold = kwargs.get('INIT_score_threshold', 0.01)
        self.post_threshold = kwargs.get('INIT_post_threshold', 0.01)
        self.all_classes = ALL_CLASSES
        self.num_classes = kwargs.get('INIT_num_classes', 80)
        self.scale_x_y = kwargs.get('INIT_scale_x_y', 1.05)
        self.downsample_ratio = kwargs.get('INIT_downsample_ratio', 32)
        self.clip_bbox = True

        self.use_visualization = kwargs.get('INIT_use_visualization', True)
        # rgb
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        kwargs['INIT_num_classes'] = self.num_classes
        self.net = PPYOLOv2Base(**kwargs)
        self.anchors = self.net.anchors

        if model_path.endswith('.pdparams'):
            self.net.load_paddle_weights(model_path)
        else:
            self.net.load_pytorch_weights(model_path)
        # self.net.load_paddle_weights('ptyolov3/vehicle_yolov3_darknet.pdparams')
        # self.net.load_paddle_weights('ptyolov3/pedestrian_yolov3_darknet.pdparams')
        self.net.eval()
        # self.net.save_pytorch_weights('pedestrian_yolov3_darknet.pth')
        # self.PostProcessor = BBoxPostProcess(
        #     num_classes=80,
        #     decode=None,
        #     nms=None,
        # )

        self.decoder = Decode(self.conf_thresh,
                              0.45,
                              self.input_shape,
                              _yolo=None,
                              all_classes=self.all_classes,
                              scale_x_y=self.scale_x_y,
                              )


        if self.use_gpu:
            self.net.cuda()

    def preprocess(self, image):
        # not keep ratio
        im_shape = image.shape
        resize_h, resize_w = self.input_shape
        im_scale_y = resize_h / im_shape[0]
        im_scale_x = resize_w / im_shape[1]
        resized_img = cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=cv2.INTER_CUBIC # default 2:INTER_CUBIC      INTER_LINEAR
        )
        normalized_img = self.normalize(resized_img)
        return normalized_img, (im_scale_x, im_scale_y)

    def normalize(self, image):
        # rgb
        return (image[:,:,::-1].astype(np.float32) / 255 - self.mean) / self.std

    def __call__(self, image_data, **kwargs):
        if len(image_data.shape) != 4:
            image_data = np.expand_dims(image_data, 0)

        if image_data.shape[-1] == 3:
            image_data = image_data.transpose([0,3,1,2])

        inp = torch.from_numpy(image_data)
        if self.use_gpu:
            inp = inp.cuda()
        with torch.no_grad():
            outputs = self.net(inp)

        return outputs

    def run(self, image_data, **kwargs):
        if image_data is None or not isinstance(image_data, np.ndarray):
            return None, image_data
        if len(image_data.shape) != 3 or image_data.shape[-1] != 3:
            return None, image_data

        org_h, org_w = image_data.shape[:2]
        inp = self.decoder.process_image(image_data)

        infer = self(inp)
        with torch.no_grad():
            boxes, scores, classes = self.pt_postprocess(infer, np.array([[int(org_h), int(org_w)]], dtype=np.float32))
        boxes, scores, classes = boxes[0].cpu().numpy(), scores[0].cpu().numpy(), classes[0].cpu().numpy()
        classes = classes.squeeze(-1)

        visualization = self.visualize(image_data, boxes, scores, classes) if self.use_visualization else image_data

        if boxes is None:
            return [], image_data

        detections = []
        clip = lambda x,y,z:int(min(max(x,y),z))
        for box, score, cls in zip(boxes, scores, classes):
            x1,y1,x2,y2 = box
            x1 = clip(x1, 0, org_w)
            y1 = clip(y1, 0, org_h)
            x2 = clip(x2, 0, org_w)
            y2 = clip(y2, 0, org_h)
            score = float(score)
            cls = int(cls)
            detections.append([x1,y1,x2,y2,score,cls])

        return detections, visualization


    def pt_postprocess(self, outputs, org_im_size):
        boxes = []
        scores = []
        for i, output in enumerate(outputs):
            box, score = yolo_box(output, np.array(self.net.anchors)[self.net.anchor_mask[i]], self.downsample_ratio,
                              self.num_classes, self.scale_x_y, org_im_size, self.clip_bbox,
                              conf_thresh=self.conf_thresh, use_gpu=self.use_gpu)
            boxes.append(box)
            scores.append(score)
        yolo_boxes = torch.cat(boxes, dim=1)
        yolo_scores = torch.cat(scores, dim=1)

        # nms
        preds = []
        nms_cfg = {
            'score_threshold': self.score_threshold,
            'post_threshold': self.post_threshold,
            'nms_top_k': self.keep_top_k,
            'keep_top_k': self.keep_top_k,
            'use_gaussian': False,
            'gaussian_sigma': 2.,

        }

        # nms_cfg = copy.deepcopy(self.nms_cfg)
        # nms_type = nms_cfg.pop('nms_type')
        batch_size = yolo_boxes.shape[0]

        boxes, scores, classes = [], [], []
        # matrix nms
        for i in range(batch_size):
            # pred = matrix_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
            # preds.append(pred)
            box, score, cls = matrix_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
            boxes.append(box)
            scores.append(score)
            classes.append(cls)

        # print(preds)
        # exit()
        return boxes, scores, classes


    def visualize(self, image_data, boxes, scores, classes):
        visualized_img = image_data.copy()
        if boxes is not None:
            self.decoder.draw(visualized_img, boxes, scores, classes)
        return visualized_img

    def run_path(self, image_path):
        image = cv2.imread(image_path)
        return self.run(image)

