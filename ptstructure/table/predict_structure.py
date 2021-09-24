import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time
import torch
import tools.infer.pytorchocr_utility as utility
from pytorchocr.base_ocr_v20 import BaseOCRV20
from pytorchocr.data import create_operators, transform
from pytorchocr.postprocess import build_post_process
from pytorchocr.utils.logging import get_logger
from pytorchocr.utils.utility import get_image_file_list, check_and_read_gif
from ptstructure.utility import parse_args

logger = get_logger()


class TableStructurer(BaseOCRV20):
    def __init__(self, args, **kwargs):
        pre_process_list = [{
            'ResizeTableImage': {
                'max_len': args.table_max_len
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'PaddingTableImage': None
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image']
            }
        }]
        postprocess_params = {
            'name': 'TableLabelDecode',
            "character_type": args.table_char_type,
            "character_dict_path": args.table_char_dict_path,
        }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu

        self.weights_path = args.table_model_path
        self.yaml_path = args.table_yaml_path
        network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path)

        super(TableStructurer, self).__init__(network_config, **kwargs)

        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        starttime = time.time()

        # self.input_tensor.copy_from_cpu(img)
        # self.predictor.run()
        # outputs = []
        # for output_tensor in self.output_tensors:
        #     output = output_tensor.copy_to_cpu()
        #     outputs.append(output)
        del img
        img = np.load('inp.npy')
        print('inp ==> ', np.sum(img), np.mean(img), np.max(img), np.min(img))
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.cuda()
            outputs = self.net(inp)
        preds = {}
        preds['structure_probs'] = outputs['structure_probs'].cpu().numpy()#outputs[1]
        preds['loc_preds'] = outputs['loc_preds'].cpu().numpy()#outputs[0]
        aa = preds['structure_probs']
        print('==> ', np.sum(aa), np.mean(aa), np.max(aa), np.min(aa))
        aa = preds['loc_preds']
        print('==> ', np.sum(aa), np.mean(aa), np.max(aa), np.min(aa));

        post_result = self.postprocess_op(preds)

        structure_str_list = post_result['structure_str_list']
        res_loc = post_result['res_loc']
        imgh, imgw = ori_im.shape[0:2]
        res_loc_final = []

        for rno in range(len(res_loc[0])):
            x0, y0, x1, y1 = res_loc[0][rno]
            left = max(int(imgw * x0), 0)
            top = max(int(imgh * y0), 0)
            right = min(int(imgw * x1), imgw - 1)
            bottom = min(int(imgh * y1), imgh - 1)
            res_loc_final.append([left, top, right, bottom])

        structure_str_list = structure_str_list[0][:-1]
        structure_str_list = ['<html>', '<body>', '<table>'] + structure_str_list + ['</table>', '</body>', '</html>']

        elapse = time.time() - starttime
        return (structure_str_list, res_loc_final), elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    table_structurer = TableStructurer(args)
    count = 0
    total_time = 0
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        structure_res, elapse = table_structurer(img)

        logger.info("result: {}".format(structure_res))

        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    main(parse_args())
