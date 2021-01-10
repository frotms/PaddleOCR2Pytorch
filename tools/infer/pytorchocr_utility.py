import os, sys
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def get_default_config():
    dc = {}
    # params for prediction engine
    dc['use_gpu'] = True
    # dc['ir_optim'] = True
    # dc['use_tensorrt'] = False
    # dc['use_fp16'] = False
    dc['max_batch_size'] = 10
    # dc['gpu_mem'] = 8000

    # params for text detector
    dc['det_algorithm'] = 'DB'
    dc['det_model_path'] = ''
    dc['det_limit_side_len'] = 960 # det_max_side_len
    dc['det_limit_type'] = 'max'

    # DB params
    dc['det_db_thresh'] = 0.3
    dc['det_db_box_thresh'] = 0.5
    dc['det_db_unclip_ratio'] = 1.6

    # EAST parmas
    dc['det_east_score_thresh'] = 0.8
    dc['det_east_cover_thresh'] = 0.1
    dc['det_east_nms_thresh'] = 0.2

    # SAST parmas
    dc['det_sast_score_thresh'] = 0.5
    dc['det_sast_nms_thresh'] = 0.2
    dc['det_sast_polygon'] = False

    # params for text recognizer
    dc['rec_algorithm'] = 'CRNN'
    dc['rec_model_path'] = ''
    dc['rec_image_shape'] = '3, 32, 320'
    dc['rec_char_type'] = 'ch'
    dc['rec_batch_num'] = 6
    dc['max_text_length'] = 25
    dc['rec_char_dict_path'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'pytorchocr/utils/ppocr_keys_v1.txt')
    dc['use_space_char'] = True
    dc['vis_font_path'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'doc', 'simfang.ttf')
    dc['drop_score'] = 0.5
    dc['limited_max_width'] = 1280
    dc['limited_min_width'] = 16

    # params for text classifier
    dc['use_angle_cls'] = False
    dc['cls_model_path'] = ''
    dc['cls_image_shape'] = '3, 48, 192'
    dc['label_list'] = ['0', '180']
    dc['cls_batch_num'] = 30
    dc['cls_thresh'] = 0.9

    dc['enable_mkldnn'] = False
    dc['use_zero_copy_run'] = False
    dc['use_pdserving'] = False

    return dc


def AnalysisConfig(weights_path):
    if not os.path.exists(os.path.abspath(weights_path)):
        raise FileNotFoundError('{} is not found.'.format(weights_path))

    weights_basename = os.path.basename(weights_path)
    weights_name = weights_basename.lower()

    supported_weights = ['ch_ptocr_server_v2.0_det_infer.pth',
                         'ch_ptocr_server_v2.0_rec_infer.pth',
                         'ch_ptocr_mobile_v2.0_det_infer.pth',
                         'ch_ptocr_mobile_v2.0_rec_infer.pth',
                         'ch_ptocr_mobile_v2.0_cls_infer.pth',
                       ]
    assert weights_name in supported_weights, \
        "supported weights are {} but input weights is {}".format(supported_weights, weights_name)

    if weights_name == 'ch_ptocr_server_v2.0_det_infer.pth':
        network_config = {'model_type':'det',
                          'algorithm':'DB',
                          'Transform':None,
                          'Backbone':{'name':'ResNet', 'layers':18, 'disable_se':True},
                          'Neck':{'name':'DBFPN', 'out_channels':256},
                          'Head':{'name':'DBHead', 'k':50}}

    elif weights_name == 'ch_ptocr_server_v2.0_rec_infer.pth':
        network_config = {'model_type':'rec',
                          'algorithm':'CRNN',
                          'Transform':None,
                          'Backbone':{'name':'ResNet', 'layers':34},
                          'Neck':{'name':'SequenceEncoder', 'hidden_size':256, 'encoder_type':'rnn'},
                          'Head':{'name':'CTCHead', 'fc_decay': 4e-05}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'DBFPN', 'out_channels': 96},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_rec_infer.pth':
        network_config = {'model_type':'rec',
                          'algorithm':'CRNN',
                          'Transform':None,
                          'Backbone':{'model_name':'small', 'name':'MobileNetV3', 'scale':0.5, 'small_stride':[1,2,2,2]},
                          'Neck':{'name':'SequenceEncoder', 'hidden_size':48, 'encoder_type':'rnn'},
                          'Head':{'name':'CTCHead', 'fc_decay': 4e-05}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_cls_infer.pth':
        network_config = {'model_type':'cls',
                          'algorithm':'CLS',
                          'Transform':None,
                          'Backbone':{'name':'MobileNetV3', 'model_name':'small', 'scale':0.35},
                          'Neck':None,
                          'Head':{'name':'ClsHead', 'class_dim':2}}

    else:
        raise NotImplementedError

    return network_config


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)