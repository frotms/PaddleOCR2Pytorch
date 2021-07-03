
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time
import sys

import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20
import tools.infer.pytorchocr_utility as utility
from pytorchocr.utils.utility import get_image_file_list, check_and_read_gif
from pytorchocr.data import create_operators, transform
from pytorchocr.postprocess import build_post_process


class TextE2E(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.args = args
        self.e2e_algorithm = args.e2e_algorithm
        pre_process_list = [{
            'E2EResizeForTest': {}
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.e2e_algorithm == "PGNet":
            pre_process_list[0] = {
                'E2EResizeForTest': {
                    'max_side_len': args.e2e_limit_side_len,
                    'valid_set': 'totaltext'
                }
            }
            postprocess_params['name'] = 'PGPostProcess'
            postprocess_params["score_thresh"] = args.e2e_pgnet_score_thresh
            postprocess_params["character_dict_path"] = args.e2e_char_dict_path
            postprocess_params["valid_set"] = args.e2e_pgnet_valid_set
            postprocess_params["mode"] = args.e2e_pgnet_mode
            self.e2e_pgnet_polygon = args.e2e_pgnet_polygon
        else:
            print("unknown e2e_algorithm:{}".format(self.e2e_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.weights_path = args.e2e_model_path
        self.yaml_path = args.e2e_yaml_path
        network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path)
        super(TextE2E, self).__init__(network_config, **kwargs)

        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):

        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.cuda()
            outputs = self.net(inp)

        preds = {}
        if self.e2e_algorithm == 'PGNet':
            preds['f_border'] = outputs['f_border'].cpu().numpy()
            preds['f_char'] = outputs['f_char'].cpu().numpy()
            preds['f_direction'] = outputs['f_direction'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
        else:
            raise NotImplementedError
        post_result = self.postprocess_op(preds, shape_list)
        points, strs = post_result['points'], post_result['texts']
        dt_boxes = self.filter_tag_det_res_only_clip(points, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, strs, elapse


if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextE2E(args)
    count = 0
    total_time = 0
    draw_img_save = "./inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        points, strs, elapse = text_detector(img)
        if count > 0:
            total_time += elapse
        count += 1
        print("Predict time of {}: {}".format(image_file, elapse))
        src_im = utility.draw_e2e_res(points, strs, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "e2e_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        print("The visualized image saved in {}".format(img_path))
    if count > 1:
        print("Avg Time: {}".format(total_time / (count - 1)))
