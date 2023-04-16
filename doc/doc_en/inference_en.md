
# Reasoning based on Python prediction engine

We first introduce how to convert a `paddle` trained model into a `pytorch` model, and then we will introduce text detection, text recognition, angle class, and the concatenation of them based on inference model.

- [CONVERT PADDLE-OCR MODEL TO PYTORCH MODEL](#CONVERT)
    - [CHINESE AND ENGLISH GENERAL OCR MODELS](#GENERAL)
    - [MULTILINGUAL MODELS](#MULTILINGUAL)
    - [END2END MODELS](#E2E_MODELS)
    - [SUPER RESOLUTION MODELS](#SR_MODELS)
    - [OTHER DETECTION MODELS](#CVT_DETECTION)
    - [OTHER RECOGNITION MODELS](#CVT_RECOGNITION)


- [INFERENCE IN PYTORCH](#INFERENCE)
    - [TEXT DETECTION MODEL INFERENCE](#DETECTION)
    - [TEXT RECOGNITION MODEL INFERENCE](#RECOGNITION)
    - [TEXT DIRECTION CLASSIFICATION MODEL INFERENCE](#CLASSIFICATION)
    - [TEXT DETECTION ANGLE CLASSIFICATION AND RECOGNITION INFERENCE CONCATENATION](#CONCATENATION)
    - [END2END MODEL INFERENCE](#E2E_INFERENCE)
    - [SUPER RESOLUTION MODEL INFERENCE](#SR_INFERENCE)
    - [OTHER MODEL INFERENCE](#OTHER)
    - [PARSER LIST](#PARSER)

- [References](#References)

<a name="CONVERT"></a>

## CONVERT PADDLE-OCR MODEL TO PYTORCH MODEL

The PyTorch models are converted from trained models in PaddleOCR.

PaddleOCR models in BaiduPan: https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g code：lmv7 

<a name="GENERAL"></a>

### CHINESE AND ENGLISH GENERAL OCR MODELS

```bash
python3 ./converter/ch_ppocr_mobile_v2.0_det_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_det_train_dir

python3 ./converter/ch_ppocr_server_v2.0_det_converter.py --src_model_path paddle_ch_ppocr_server_v2.0_det_train_dir

python3 ./converter/ch_ppocr_mobile_v2.0_rec_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_rec_train_dir

python3 ./converter/ch_ppocr_server_v2.0_rec_converter.py --src_model_path paddle_ch_ppocr_server_v2.0_rec_train_dir

python3 ./converter/ch_ppocr_mobile_v2.0_cls_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_cls_train_dir

#ppocr v2
python3 ./converter/ch_ppocr_v2_det_converter.py --src_model_path ./paddle_ch_PP-OCRv2_det_distill_train_dir

python ./converter/ch_ppocr_v2_rec_converter.py --src_model_path ./paddle_ch_PP-OCRv2_rec_train_dir

#ppocr v3
# det
# ch_PP-OCRv3_rec_train, en_PP-OCRv3_det_distill_train, Multilingual_PP-OCRv3_det_distill_train
python ./converter/ch_ppocr_v3_det_converter.py --src_model_path paddle_ch_PP-OCRv3_rec_train_dir

python ./converter/ch_ppocr_v3_rec_converter.py --src_model_path paddle_ch_PP-OCRv3_rec_train_dir
```

<a name="MULTILINGUAL"></a>

### MULTILINGUAL MODELS

```bash
python3 ./converter/multilingual_mobile_v2.0_rec_converter.py --src_model_path paddle_multilingual_mobile_v2.0_rec_train_dir

# v3
# en_PP-OCRv3_rec, multilingual_PP-OCRv3_rec
python ./converter/multilingual_ppocr_v3_rec_converter.py --src_model_path paddle_multilingual_PP-OCRv3_rec_train_dir
```

<a name="E2E_MODELS"></a>

### END2END MODELS

```bash
# en_server_pgnetA
python ./converter/e2e_converter.py --yaml_path ./configs/e2e/e2e_r50_vd_pg.yml --src_model_path your_ppocr_e2e_models_en_server_pgnetA_train_dir
```

<a name="SR_MODELS"></a>

### SUPER RESOLUTION MODELS

```bash
# sr_telescope
python ./converter/sr_converter.py --yaml_path ./configs/sr/sr_telescope.yml --src_model_path your_ppocr_sr_telescope_train_dir
```

<a name="cvt_DETECTION"></a>

### OTHER DETECTION MODELS

```bash
# det_mv3_db
python3 ./converter/det_converter.py --yaml_path ./configs/det/det_mv3_db.yml --src_model_path your_ppocr_det_mv3_db_v2.0_train_dir

# det_mv3_east
python3 ./converter/det_converter.py --yaml_path ./configs/det/det_mv3_east.yml --src_model_path your_ppocr_det_mv3_east_v2.0_train_dir

# det_r50_vd_db
python3 ./converter/det_converter.py --yaml_path ./configs/det/det_r50_vd_db.yml --src_model_path your_ppocr_det_r50_vd_db_v2.0_train_dir

# det_r50_vd_east
python3 ./converter/det_converter.py --yaml_path ./configs/det/det_r50_vd_east.yml --src_model_path your_ppocr_det_r50_vd_east_v2.0_train_dir

# det_r50_vd_sast_icdar15
 python3 ./converter/det_converter.py --yaml_path ./configs/det/det_r50_vd_sast_icdar15.yml --src_model_path your_ppocr_det_r50_vd_sast_icdar15_v2.0_train_dir
 
# det_r50_vd_sast_totaltext
python3 ./converter/det_converter.py --yaml_path ./configs/det/det_r50_vd_sast_totaltext.yml --src_model_path your_ppocr_det_r50_vd_sast_totaltext_v2.0_train_dir

# det_mv3_pse
python3 ./converter/det_converter.py --yaml_path ./configs/det/det_mv3_pse.yml --src_model_path your_ppocr_det_mv3_pse_v2.0_train_dir

# det_r50_vd_pse
python3 ./converter/det_converter.py --yaml_path ./configs/det/det_r50_vd_pse.yml --src_model_path your_ppocr_det_r50_vd_pse_v2.0_train_dir

# det_fcenet
python3 ./converter/det_fcenet_converter.py --src your_det_r50_dcn_fce_ctw/det_r50_dcn_fce_ctw_v2.0_train_dir

# db++
python3 ./converter/det_converter.py --yaml_path ./configs/det/det_r50_db++_icdar15.yml --src_model_path your_ppocr_det_r50_db++_icdar15_or_td_tr_train_dir
```

<a name="CVT_RECOGNITION"></a>

### OTHER RECOGNITION MODELS

```bash
# rec_mv3_none_none_ctc
python3 ./converter/rec_converter.py --yaml_path ./configs/rec/rec_mv3_none_none_ctc.yml --src_model_path your_ppocr_rec_mv3_none_none_ctc_v2.0_train_dir

# rec_r34_vd_none_none_ctc
python3 ./converter/rec_converter.py --yaml_path ./configs/rec/rec_r34_vd_none_none_ctc.yml --src_model_path your_ppocr_rec_r34_vd_none_none_ctc_v2.0_train_dir

# rec_mv3_none_bilstm_ctc
python3 ./converter/rec_converter.py --yaml_path ./configs/rec/rec_mv3_none_bilstm_ctc.yml --src_model_path your_ppocr_rec_mv3_none_bilstm_ctc_v2.0_train_dir

# rec_r34_vd_none_bilstm_ctc
python3 ./converter/rec_converter.py --yaml_path ./configs/rec/rec_r34_vd_none_bilstm_ctc.yml --src_model_path your_ppocr_rec_r34_vd_none_bilstm_ctc_v2.0_train_dir

# rec_mv3_tps_bilstm_ctc
python ./converter/rec_converter.py --yaml_path ./configs/rec/rec_mv3_tps_bilstm_ctc.yml --src_model_path your_ppocr_rec_mv3_tps_bilstm_ctc_v2.0_train_dir

# rec_r34_vd_tps_bilstm_ctc
python ./converter/rec_converter.py --yaml_path ./configs/rec/rec_r34_vd_tps_bilstm_ctc.yml --src_model_path your_ppocr_rec_r34_vd_tps_bilstm_ctc_v2.0_train_dir

# rec_mv3_tps_bilstm_att
python ./converter/rec_converter.py --yaml_path ./configs/rec/rec_mv3_tps_bilstm_att.yml --src_model_path your_ppocr_rec_mv3_tps_bilstm_att_v2.0_train_dir

# rec_r34_vd_tps_bilstm_att
python ./converter/rec_converter.py --yaml_path ./configs/rec/rec_r34_vd_tps_bilstm_att.yml --src_model_path your_ppocr_rec_r34_vd_tps_bilstm_att_v2.0_train_dir

# rec_r50_vd_srn
python ./converter/srn_converter.py --yaml_path ./configs/rec/rec_r50_fpn_srn.yml --src_model_path your_ppocr_rec_r50_vd_srn_train_dir

# NRTR
python ./rec_nrtr_mtb_converter.py --yaml_path ../configs/rec/rec_mtb_nrtr.yml --dict_path ../pytorchocr/utils/EN_symbol_dict.txt --src_model_path your_ppocr_rec_mtb_nrtr_train_dir

# SAR
python ./converter/rec_sar_converter.py --yaml_path ./configs/rec/rec_r31_sar.yml --dict_path ./pytorchocr/utils/dict90.txt --src_model_path your_rec_r31_sar_train_dir

# SVTR
python ./converter/rec_svtr_converter.py --yaml_path ./configs/rec/rec_svtr/rec_svtr_tiny_6local_6global_stn_en.yml --src_model_path your_rec_svtr_tiny_none_ctc_en_train_dir

# ViTSTR
python ./converter/rec_vitstr_converter.py --yaml_path ./configs/rec/rec_vitstr_none_ce.yml --src_model_path your_rec_vitstr_none_ce_train_dir

# CAN
python3 ./converter/rec_can_converter.py --yaml_path ./configs/rec/rec_d28_can.yml --src_model_path your_rec_d28_can_train_dir
```

<a name="INFERENCE"></a>

## INFERENCE IN PYTORCH

PyTorch models in BaiduPan: https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg code：6clx

You can also get the pytorch models with the converter by yourself. 

<a name="DETECTION"></a>

### TEXT DETECTION MODEL INFERENCE

```bash
python3 ./tools/infer/predict_det.py --image_dir ./doc/imgs --model_path your_det_pth_path.pth

# v3
python ./tools/infer/predict_det.py --det_model_path your_ch_ptocr_v3_det_infer_path.pth --image_dir ./doc/imgs/1.jpg
```

![](../../doc/imgs_results/det_res_img_10_db.jpg)

![](../../doc/imgs_results/det_res_img623_sast.jpg)

#### Multilingual Detection Models

```bash
# v3
# en_ptocr_v3_det_infer.pth, multilingual_ptocr_v3_det_infer.pth
python ./tools/infer/predict_det.py --det_algorithm DB --det_yaml_path ./configs/det/det_ppocr_v3.yml --det_model_path your_multilingual_ptocr_v3_det_infer_path.pth --image_dir ./doc/imgs/1.jpg
```

<a name="RECOGNITION"></a>

### TEXT RECOGNITION MODEL INFERENCE

#### Chinese and English General OCR Models

```bash
python3 ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words --model_path your_rec_pth_path.pth

# v3
python ./tools/infer/predict_rec.py --rec_model_path your_ch_ptocr_v3_rec_infer_path.pth --rec_image_shape 3,48,320 --image_dir ./doc/imgs_words/en/word_1.png
```

![](../../doc/imgs_words/ch/word_4.jpg)

```
Predicts of ./doc/imgs_words/ch/word_4.jpg:('实力活力', 0.98458153)
```

#### Multilingual Recognition Models

If you need to predict other language models, when using inference model prediction, you need to specify the dictionary path used by `--rec_char_dict_path`. At the same time, in order to get the correct visualization results,
You need to specify the visual font path through `--vis_font_path`. There are small language fonts provided by default under the `doc/fonts` path

```bash
# python3 ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words/spanish/es_1.jpg --rec_model_dir ../rec_models/multi_language/spanish/es_mobile_v2.0_rec_infer/ --rec_char_type your_multilingual_char_type --rec_char_dict_path ./ppocr/utils/dict/your_multilingual_dict.txt

python3 ./tools/infer/predict_rec.py --rec_model_path your_japan_mobile_v2.0_rec_infer_path.pth --rec_char_type japan --rec_char_dict_path ./pytorchocr/utils/dict/japan_dict.txt --image_dir ./doc/imgs_words/japan/1.jpg

# rec_char_type
# rec_char_type
# support_character_type = [ 
#             # release/2.0
#             'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
#             'it', 'es', 'pt', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs_latin',
#             'oc', 'rs_cyrillic', 'bg', 'uk', 'be', 'te', 'kn', 'ch_tra', 'hi',
#             'mr', 'ne', 'EN'
#             # release/2.1
#             'xi', 'pu', 'rs', 'rsc', 'ka', 'chinese_cht', 'latin', 'arabic',
#             'cyrillic', 'devanagari'
#         ]

# v3
python ./tools/infer/predict_rec.py --rec_model_path your_en_ptocr_v3_rec_infer_path.pth --rec_image_shape 3,48,320 --rec_yaml_path ./configs/rec/PP-OCRv3/en_ptocr_v3_rec.yml --rec_char_dict_path ./pytorchocr/utils/en_dict.txt  --image_dir ./doc/imgs_words/en/word_1.png 

python ./tools/infer/predict_rec.py --rec_model_path your_japan_ptocr_v3_rec_infer_path.pth --rec_image_shape 3,48,320 --rec_yaml_path ./configs/rec/PP-OCRv3/multi_language/japan_PP-OCRv3_rec.yml --rec_char_dict_path ./pytorchocr/utils/dict/japan_dict.txt  --image_dir ./doc/imgs_words/japan/1.jpg
```

refer to [paddleocr.py](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/paddleocr.py#L283)

![](../../doc/imgs_words/korean/1.jpg)

```
Predicts of ./doc/imgs_words/korean/1.jpg:('바탕으로', 0.9948904)
```

<a name="CLASSIFICATION"></a>

### TEXT DIRECTION CLASSIFICATION MODEL IN INFERENCE

```bash
python3 ./tools/infer/predict_cls.py --image_dir ./doc/imgs_words --model_path your_cls_pth_path.pth
```

![](../../doc/imgs_words/ch/word_1.jpg)

```
Predicts of ./doc/imgs_words/ch/word_4.jpg:['0', 0.9999982]
```

<a name="CONCATENATION"></a>

### TEXT DETECTION ANGLE CLASSIFICATION AND RECOGNITION INFERENCE CONCATENATION

#### Chinese and English OCR Models

```bash
# use direction classifier
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path your_det_pth_path.pth --rec_model_path your_rec_pth_path.pth --use_angle_cls --cls_model_path your_cls_pth_path.pth --vis_font_path ./doc/fonts/your_lang_font.ttf --rec_char_type your_char_type --rec_char_dict_path ./ppocr/utils/dict/your_dict.txt

# not use use direction classifier
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path your_det_pth_path.pth --rec_model_path your_rec_pth_path.pth --vis_font_path ./doc/fonts/your_lang_font.ttf --rec_char_type your_char_type --rec_char_dict_path ./ppocr/utils/dict/your_dict.txt

# v3
# other rec-models: use --rec_char_dict_path and --rec_yaml_path
python ./tools/infer/predict_system.py --image_dir ./doc/imgs/1.jpg --det_model_path your_ch_ptocr_v3_det_infer_path.pth --rec_image_shape 3,48,320 --rec_model_path your_ch_ptocr_v3_rec_infer_path.pth
```

After executing the command, the recognition result image is as follows:

![](../../doc/imgs_results/system_res_00018069.jpg)

<a name="E2E_INFERENCE"></a>

### END2END MODEL INFERENCE

```bash
# en_server_pgnetA
python tools/infer/predict_e2e.py --e2e_model_path ./en_server_pgnetA_infer.pth --image_dir ./doc/imgs_en/img623.jpg --e2e_algorithm PGNet --e2e_pgnet_polygon True --e2e_char_dict_path ./pytorchocr/utils/ic15_dict.txt --e2e_yaml_path ./configs/e2e/e2e_r50_vd_pg.yml
```

![](../../doc/imgs_results/e2e_res_img623_pgnet.jpg)

<a name="SR_INFERENCE"></a>

### SUPER RESOLUTION MODEL INFERENCE

```bash
# sr_telescope
python ./tools/infer/predict_sr.py --sr_yaml_path ./configs/sr/sr_telescope.yml --sr_model_path your_sr_telescope_infer_path.pth --image_dir
./doc/imgs_words_en/word_52.png
```

![](C:/workspace/repo/deep_learning_framework/github/PaddleOCR2Pytorch/doc/imgs_words_en/word_52.png)

![](C:/workspace/repo/deep_learning_framework/github/PaddleOCR2Pytorch/doc/imgs_results/sr/sr_word_52.png)

<a name="OTHER"></a>

### OTHER MODEL INFERENCE

If you want to try other detection algorithms or recognition algorithms, please refer to the above text detection model inference and text recognition model inference, update the corresponding configuration and model.

```bash
# detection
# det_mv3_db
python3 ./tools/infer/predict_det.py --det_model_path your_det_mv3_db_v2.0_infer_path.pth --image_dir ./doc/imgs_en/img_195.jpg  --det_algorithm DB --det_yaml_path ./configs/det/det_mv3_db.yml

# det_mv3_east
python3 ./tools/infer/predict_det.py --det_model_path your_det_mv3_east_v2.0_infer_path.pth --image_dir ./doc/imgs_en/img_195.jpg  --det_algorithm EAST --det_yaml_path ./configs/det/det_mv3_east.yml

# det_r50_vd_db
python3 ./tools/infer/predict_det.py --det_model_path your_det_r50_vd_db_v2.0_infer_path.pth --image_dir ./doc/imgs_en/img_195.jpg  --det_algorithm DB --det_yaml_path ./configs/det/det_r50_vd_db.yml

# det_r50_vd_east
python3 ./tools/infer/predict_det.py --det_model_path your_det_r50_vd_east_v2.0_infer_path.pth --image_dir ./doc/imgs_en/img_195.jpg  --det_algorithm EAST --det_yaml_path ./configs/det/det_r50_vd_east.yml

# det_r50_vd_sast_icdar15
python ./tools/infer/predict_det.py --det_model_path your_det_r50_vd_sast_icdar15_v2.0_infer_path.pth --image_dir ./doc/imgs/00006737.jpg  --det_algorithm SAST --det_yaml_path ./configs/det/det_r50_vd_sast_icdar15.yml

# det_r50_vd_sast_totaltext
python3 ./tools/infer/predict_det.py --det_model_path your_det_r50_vd_sast_totaltext_v2.0_infer_path.pth --image_dir ./doc/imgs/00006737.jpg  --det_algorithm SAST --det_yaml_path ./configs/det/det_r50_vd_sast_totaltext.yml

# det_mv3_pse
python3 ./tools/infer/predict_det.py --det_model_path your_det_mv3_pse_v2.0_infer_path.pth --image_dir ./doc/imgs_en/img_195.jpg  --det_algorithm PSE --det_yaml_path ./configs/det/det_mv3_pse.yml

# det_r50_vd_pse
python3 ./tools/infer/predict_det.py --det_model_path your_det_r50_vd_pse_v2.0_infer_path.pth --image_dir ./doc/imgs_en/img_195.jpg  --det_algorithm PSE --det_yaml_path ./configs/det/det_r50_vd_pse.yml

# ppocr_v3_det
python ./tools/infer/predict_det.py --use_gpu false --det_algorithm DB --det_model_path your_xx_ptocr_v3_det_infer_path.pth --image_dir ./doc/imgs/1.jpg

# det_fcenet
python3.7 ./tools/infer/predict_det.py --det_algorithm FCE --det_model_path your_det_r50_dcn_fce_ctw_v2.0_infer_path.pth --det_fce_box_type poly --det_yaml_path ./configs/det/det_r50_vd_dcn_fce_ctw.yml --image_dir ./doc/imgs_en/img_10.jpg

# db++
python3 tools/infer/predict_det.py --image_dir ./doc/imgs_en/img_10.jpg --det_model_path your_det_r50_db++_icdar15_infer_path.pth --det_algorithm="DB++"  --det_yaml_path ./configs/det/det_r50_db++_icdar15.yml
python3 tools/infer/predict_det.py --image_dir ./doc/imgs_en/img_10.jpg --det_model_path your_det_r50_db++_td_tr_infer_path.pth --det_algorithm="DB++"  --det_yaml_path ./configs/det/det_r50_db++_td_tr.yml


# recognition
# rec_mv3_none_none_ctc
python3 ./tools/infer/predict_rec.py --rec_model_path your_rec_mv3_none_none_ctc_v2.0_infer_path.pth --image_dir ./doc/imgs_words_en/word_10.png --rec_char_dict_path ./pytorchocr/utils/dict/en_dict.txt --rec_char_type en --rec_yaml_path ./configs/rec/rec_mv3_none_none_ctc.yml

# rec_r34_vd_none_none_ctc
 python3 ./tools/infer/predict_rec.py --rec_model_path your_rec_r34_vd_none_none_ctc_v2.0_infer_path.pth --image_dir ./doc/imgs_words_en/word_201.png --rec_char_dict_path ./pytorchocr/utils/dict/en_dict.txt --rec_char_type en --rec_yaml_path ./configs/rec/rec_r34_vd_none_none_ctc.yml

# rec_mv3_none_bilstm_ctc
python3 ./tools/infer/predict_rec.py --rec_model_path your_rec_mv3_none_bilstm_ctc_v2.0_infer_path.pth --image_dir ./doc/imgs_words_en/word_10.png --rec_char_dict_path ./pytorchocr/utils/dict/en_dict.txt --rec_char_type en --rec_yaml_path ./configs/rec/rec_mv3_none_bilstm_ctc.yml

# rec_r34_vd_none_bilstm_ctc
python3 ./tools/infer/predict_rec.py --rec_model_path your_rec_r34_vd_none_bilstm_ctc_v2.0_infer_path.pth --image_dir ./doc/imgs_words_en/word_10.png --rec_char_dict_path ./pytorchocr/utils/dict/en_dict.txt --rec_char_type en --rec_yaml_path ./configs/rec/rec_r34_vd_none_bilstm_ctc.yml
 
# rec_mv3_tps_bilstm_ctc
python ./tools/infer/predict_rec.py --rec_model_path your_rec_mv3_tps_bilstm_ctc_v2.0_infer_path.pth --image_dir ./doc/imgs_words_en/word_401.png --rec_image_shape 3,32,100 --rec_char_type en --rec_yaml_path ./configs/rec/rec_mv3_tps_bilstm_ctc.yml
 
# rec_r34_vd_tps_bilstm_ctc
python ./tools/infer/predict_rec.py --rec_model_path your_rec_r34_vd_tps_bilstm_ctc_v2.0_infer_path.pth --image_dir ./doc/imgs_words_en/word_401.png --rec_image_shape 3,32,100 --rec_char_type en --rec_yaml_path ./configs/rec/rec_r34_vd_tps_bilstm_ctc.yml

# rec_mv3_tps_bilstm_att
python ./tools/infer/predict_rec.py --rec_model_path your_rec_mv3_tps_bilstm_att_v2.0_infer_path.pth --image_dir ./doc/imgs_words_en/word_461.png --rec_image_shape 3,32,100 --rec_char_type en  --rec_algorithm RARE --rec_yaml_path ./configs/rec/rec_mv3_tps_bilstm_att.yml

# rec_r34_vd_tps_bilstm_att
python ./tools/infer/predict_rec.py --rec_model_path your_rec_r34_vd_tps_bilstm_att_v2.0_infer_path.pth --image_dir ./doc/imgs_words_en/word_461.png  --rec_image_shape 3,32,100 --rec_char_type en  --rec_algorithm RARE --rec_yaml_path ./configs/rec/rec_r34_vd_tps_bilstm_att.yml

# rec_r50_vd_srn
python ./tools/infer/predict_rec.py --rec_model_path your_rec_r50_vd_srn_infer_path.pth --image_dir ./doc/imgs_words_en/word_401.png --rec_image_shape 1,64,256 --rec_char_type en --rec_algorithm SRN --rec_yaml_path ./configs/rec/rec_r50_fpn_srn.yml

# NRTR
python ./tools/infer/predict_rec.py --rec_char_dict_path ./pytorchocr/utils/EN_symbol_dict.txt --rec_algorithm NRTR --rec_model_path your_rec_mtb_nrtr_infer_path.pth --rec_yaml_path ./configs/rec/rec_mtb_nrtr.yml --image_dir ./doc/imgs_words_en/word_10.png --rec_image_shape 1,32,100

# SAR
python ./tools/infer/predict_rec.py --rec_char_dict_path ./pytorchocr/utils/dict90.txt --max_text_length 30 --rec_yaml_path ./configs/rec/rec_r31_sar.yml --rec_algorithm SAR --rec_image_shape 3,48,48,160 --use_space_char false --rec_model_path your_rec_r31_sar_infer_path.pth --image_dir ./doc/imgs_words/en/word_1.png

# SVTR
# en
python ./tools/infer/predict_rec.py --rec_model_path yout_rec_svtr_tiny_none_ctc_en_infer_path.pth --rec_algorithm SVTR --rec_image_shape 3,64,256 --rec_char_dict_path ./pytorchocr/utils/ic15_dict.txt --rec_yaml_path ./configs/rec/rec_svtr/rec_svtr_tiny_6local_6global_stn_en.yml --image_dir ./doc/imgs_words_en/word_10.png
# ch
python ./tools/infer/predict_rec.py --rec_model_path yout_rec_svtr_tiny_none_ctc_ch_infer_path.pth --rec_algorithm SVTR --rec_image_shape 3,64,256 --rec_char_dict_path ./pytorchocr/utils/ppocr_keys_v1.txt --rec_yaml_path ./configs/rec/rec_svtr/rec_svtr_tiny_6local_6global_stn_ch.yml --image_dir ./doc/imgs_words/ch/word_1.jpg

# ViTSTR
python tools/infer/predict_rec.py --rec_model_path your_rec_vitstr_none_ce_infer_path.pth --rec_algorithm ViTSTR --rec_image_shape 1,224,224  --rec_char_dict_path ./pytorchocr/utils/EN_symbol_dict.txt --rec_yaml_path ./configs/rec/rec_vitstr_none_ce.yml --image_dir ./doc/imgs_words_en/word_10.png

# CAN
python3 ./tools/infer/predict_rec.py --image_dir="./doc/datasets/crohme_demo/hme_00.jpg" --rec_algorithm="CAN" --rec_batch_num=1 --rec_model_path your_rec_d28_can_infer_path.pth --rec_char_dict_path="./pytorchocr/utils/dict/latex_symbol_dict.txt" --rec_yaml_path ./configs/rec/rec_d28_can.yml
```

<a name="PARSER"></a>

### PARSER LIST

```bash
def init_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    # parser.add_argument("--ir_optim", type=str2bool, default=True)
    # parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    # parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--warmup", type=str2bool, default=False)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_path", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=str2bool, default=False)

    # PSE parmas
    parser.add_argument("--det_pse_thresh", type=float, default=0)
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    parser.add_argument("--det_pse_min_area", type=float, default=16)
    parser.add_argument("--det_pse_box_type", type=str, default='box')
    parser.add_argument("--det_pse_scale", type=int, default=1)

    # FCE parmas
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fourier_degree", type=int, default=5)
    parser.add_argument("--det_fce_box_type", type=str, default='poly')

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

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_path", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'pytorchocr/utils/ic15_dict.txt'))
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_polygon", type=bool, default=True)
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # SR parmas
    parser.add_argument("--sr_model_path", type=str)
    parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    parser.add_argument("--sr_batch_num", type=int, default=1)

    # params .yaml
    parser.add_argument("--det_yaml_path", type=str, default=None)
    parser.add_argument("--rec_yaml_path", type=str, default=None)
    parser.add_argument("--cls_yaml_path", type=str, default=None)
    parser.add_argument("--e2e_yaml_path", type=str, default=None)
    parser.add_argument("--sr_yaml_path", type=str, default=None)

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--show_log", type=str2bool, default=True)

    return parser
```

<a name="References"></a>

## References

- [PaddleOCR release/2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_en/inference_en.md)
- [PaddleOCR release/2.1](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/inference_en.md)
- [PaddleOCR release/2.5](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/inference_en.md)
- [PaddleOCR release/2.6](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/inference_en.md)

