
# Reasoning based on Python prediction engine

We first introduce how to convert a `paddle` trained model into a `pytorch` model, and then we will introduce text detection, text recognition, angle class, and the concatenation of them based on inference model.

- [CONVERT PADDLE-OCR MODEL TO PYTORCH MODEL](#CONVERT)
    - [CHINESE AND ENGLISH GENERAL OCR MODELS](#CHINESE AND ENGLISH GENERAL OCR MODELS)
    - [MULTILINGUAL MODELS](#MULTILINGUAL MODELS)


- [INFERENCE IN PYTORCH](#INFERENCE IN PYTORCH)
    - [TEXT DETECTION MODEL INFERENCE](#TEXT DETECTION MODEL INFERENCE)
    - [TEXT RECOGNITION MODEL INFERENCE](#TEXT RECOGNITION MODEL INFERENCE)
    - [TEXT DIRECTION CLASSIFICATION MODEL IN INFERENCE](#TEXT DIRECTION CLASSIFICATION MODEL IN INFERENCE)
    - [TEXT DETECTION ANGLE CLASSIFICATION AND RECOGNITION INFERENCE CONCATENATION](#CONCATENATION)
    - [OTHER MODEL INFERENCE](#OTHER MODEL INFERENCE)
    - [PARSER LIST](#PARSER LIST)

- [References](#References)

<a name="CONVERT"></a>

## CONVERT PADDLE-OCR MODEL TO PYTORCH MODEL

The PyTorch models are converted from trained models in PaddleOCR.

PaddleOCR models in BaiduPan: https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g code：lmv7 

<a name="CHINESE AND ENGLISH GENERAL OCR MODELS"></a>

### CHINESE AND ENGLISH GENERAL OCR MODELS

```bash
python3 ./converter/ch_ppocr_mobile_v2.0_det_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_det_train_dir

python3 ./converter/ch_ppocr_server_v2.0_det_converter.py --src_model_path paddle_ch_ppocr_server_v2.0_det_train_dir

python3 ./converter/ch_ppocr_mobile_v2.0_rec_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_rec_train_dir

python3 ./converter/ch_ppocr_server_v2.0_rec_converter.py --src_model_path paddle_ch_ppocr_server_v2.0_rec_train_dir

python3 ./converter/ch_ppocr_mobile_v2.0_cls_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_cls_train_dir
```

<a name="MULTILINGUAL MODELS"></a>

### MULTILINGUAL MODELS

```bash
python3 ./converter/multilingual_mobile_v2.0_rec_converter.py --src_model_path paddle_multilingual_mobile_v2.0_rec_train_dir
```

<a name="INFERENCE IN PYTORCH"></a>

## INFERENCE IN PYTORCH

PyTorch models in BaiduPan: https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg code：6clx

You can also get the pytorch models with the converter by yourself. 

<a name="TEXT DETECTION MODEL INFERENCE"></a>

### TEXT DETECTION MODEL INFERENCE

```bash
python3 ./tools/infer/predict_det.py --image_dir ./doc/imgs --model_path your_det_pth_path.pth
```

<a name="TEXT RECOGNITION MODEL INFERENCE"></a>

### TEXT RECOGNITION MODEL INFERENCE

#### Chinese and English General OCR Models

```bash
python3 ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words --model_path your_rec_pth_path.pth
```

#### Multilingual Recognition Models

If you need to predict other language models, when using inference model prediction, you need to specify the dictionary path used by `--rec_char_dict_path`. At the same time, in order to get the correct visualization results,
You need to specify the visual font path through `--vis_font_path`. There are small language fonts provided by default under the `doc/fonts` path

```bash
python ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words/spanish/es_1.jpg --rec_model_dir ../rec_models/multi_language/spanish/es_mobile_v2.0_rec_infer/ --rec_char_type your_multilingual_char_type --rec_char_dict_path ./ppocr/utils/dict/your_multilingual_dict.txt

# rec_char_type
# support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'it', 'es', 'pt', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs_latin',
            'oc', 'rs_cyrillic', 'bg', 'uk', 'be', 'te', 'kn', 'ch_tra', 'hi',
            'mr', 'ne', 'EN'
        ]
```

<a name="TEXT DIRECTION CLASSIFICATION MODEL IN INFERENCE"></a>

### TEXT DIRECTION CLASSIFICATION MODEL IN INFERENCE

```bash
python3 ./tools/infer/predict_cls.py --image_dir ./doc/imgs_words --model_path your_cls_pth_path.pth
```

<a name="CONCATENATION"></a>

### TEXT DETECTION ANGLE CLASSIFICATION AND RECOGNITION INFERENCE CONCATENATION

#### Chinese and English OCR Models

```bash
# use direction classifier
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path your_det_pth_path.pth --rec_model_path your_rec_pth_path.pth --use_angle_cls --cls_model_path your_cls_pth_path.pth --vis_font_path ./doc/fonts/your_lang_font.ttf --rec_char_type your_char_type --rec_char_dict_path ./ppocr/utils/dict/your_dict.txt

# not use use direction classifier
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path your_det_pth_path.pth --rec_model_path your_rec_pth_path.pth --vis_font_path ./doc/fonts/your_lang_font.ttf --rec_char_type your_char_type --rec_char_dict_path ./ppocr/utils/dict/your_dict.txt
```

After executing the command, the recognition result image is as follows:

![](../../doc/imgs_results/system_res_00018069.jpg)

<a name="OTHER MODEL INFERENCE"></a>

### OTHER MODEL INFERENCE

If you want to try other detection algorithms or recognition algorithms, please refer to the above text detection model inference and text recognition model inference, update the corresponding configuration and model.

<a name="PARSER LIST"></a>

### PARSER LIST

```bash
def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    # parser.add_argument("--ir_optim", type=str2bool, default=True)
    # parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    # parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_path", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_path", type=str)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)

    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)
    parser.add_argument("--limited_max_width", type=int, default=1280)
    parser.add_argument("--limited_min_width", type=int, default=16)

    parser.add_argument(
        "--vis_font_path", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'doc/fonts/simfang.ttf'))
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'pytorchocr/utils/ppocr_keys_v1.txt'))

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_path", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)

    return parser.parse_args()
```

<a name="References"></a>

## References

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_en/inference_en.md)