import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import numpy as np
import math
import time
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20
import tools.infer.pytorchocr_utility as utility
from pytorchocr.postprocess import build_post_process
from pytorchocr.utils.utility import get_image_file_list, check_and_read_gif


class TextRecognizer(BaseOCRV20):
    def __init__(self, config):
        OCR_CFG = utility.get_default_config()
        OCR_CFG.update(config)
        self.config = OCR_CFG

        use_gpu = OCR_CFG['use_gpu']
        self.use_gpu = torch.cuda.is_available() and use_gpu

        self.weights_path = OCR_CFG['rec_model_path']
        network_config = utility.AnalysisConfig(self.weights_path)
        super(TextRecognizer, self).__init__(network_config)

        self.limited_max_width = OCR_CFG['limited_max_width']
        self.limited_min_width = OCR_CFG['limited_min_width']

        self.rec_image_shape = [int(v) for v in OCR_CFG['rec_image_shape'].split(",")]
        self.character_type = OCR_CFG['rec_char_type']
        self.rec_batch_num = OCR_CFG['rec_batch_num']
        self.rec_algorithm = OCR_CFG['rec_algorithm']
        self.use_zero_copy_run = OCR_CFG['use_zero_copy_run']
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": OCR_CFG['rec_char_type'],
            "character_dict_path": OCR_CFG['rec_char_dict_path'],
            "use_space_char": OCR_CFG['use_space_char']
        }
        self.postprocess_op = build_post_process(postprocess_params)

        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()


    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                # norm_img = self.resize_norm_img(img_list[ino], max_wh_ratio)
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            starttime = time.time()

            with torch.no_grad():
                inp = torch.Tensor(norm_img_batch)
                if self.use_gpu:
                    inp = inp.cuda()
                prob_out = self.net(inp)
            preds = prob_out.cpu().numpy()

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse


def main(config, image_dir):
    image_file_list = get_image_file_list(image_dir)
    text_recognizer = TextRecognizer(config)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, predict_time = text_recognizer(img_list)
    except:
        print(
            "ERROR!!!! \n"
            "Please read the FAQ：https://github.com/PaddlePaddle/PaddleOCR#faq \n"
            "If your model has tps module:  "
            "TPS does not support variable shape.\n"
            "Please set --rec_image_shape='3,32,100' and --rec_char_type='en' ")
        exit()
    for ino in range(len(img_list)):
        print("Predicts of {}:{}".format(valid_image_file_list[ino], rec_res[
            ino]))
    print("Total predict time for {} images, cost: {:.3f}".format(
        len(img_list), predict_time))


if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    DEFAULT_MODEL_PATH = './ch_ptocr_server_v2.0_rec_infer.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--image_dir", type=str, help='Assign the image directory')
    parser.add_argument('-m', "--model_path", type=str, help='Assign the model path', default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    param_dict = {}
    param_dict['rec_model_path'] = args.model_path
    param_dict['drop_score'] = 0.5
    main(param_dict, args.image_dir)