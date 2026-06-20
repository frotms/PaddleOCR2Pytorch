# 基于Python预测引擎推理

首先介绍如何将`PaddleOCR`训练的模型转换成`pytorch`模型，然后将依次介绍文本检测、文本角度分类器、文本识别以及三者串联在CPU、GPU上的预测方法。


- [一、PaddleOCR训练模型转PyTorch模型](#PaddleOCR训练模型转PyTorch模型)
    - [中英文通用OCR](#中英文通用OCR)
    - [多语言识别模型](#多语言识别模型)
    - [文档预处理模型](#文档预处理模型)
    - [文档结构化解析模型（PP-StructureV3）](#文档结构化解析模型pp-structurev3)
    - [端到端模型](#端到端模型)
    - [超分辨率模型](#超分辨率模型)
    - [其他检测模型](#其他检测模型)
    - [其他识别模型](#其他识别模型)
- [二、PyTorch推理](#PyTorch推理)
    - [文本检测模型推理](#文本检测模型推理)
    - [文本识别模型推理](#文本识别模型推理)
    - [文本方向分类模型推理](#文本方向分类模型推理)
    - [文档预处理模型推理](#文档预处理模型推理)
    - [文档结构化解析推理（PP-StructureV3）](#文档结构化解析推理pp-structurev3)
    - [文本检测、方向分类和文字识别串联推理](#文本检测、方向分类和文字识别串联推理)
    - [端到端模型推理](#端到端模型推理)
    - [超分辨率模型推理](#超分辨率模型推理)
    - [其他模型推理](#其他模型推理)
    - [参数列表](#参数列表)

- [参考](#参考)

<a name="PaddleOCR训练模型转PyTorch模型"></a>

## 一、PaddleOCR训练模型转PyTorch模型

**转换模型使用PaddleOCR的*训练模型***。

模型路径详见**PaddleOCR对应模型**或者**百度网盘链接**：https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g 
提取码：lmv7 

<a name="中英文通用OCR"></a>

### 中英文通用OCR

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
# det v3
# ch_PP-OCRv3_rec_train, en_PP-OCRv3_det_distill_train, Multilingual_PP-OCRv3_det_distill_train
python ./converter/ch_ppocr_v3_det_converter.py --src_model_path paddle_ch_PP-OCRv3_det_train_dir

python ./converter/ch_ppocr_v3_rec_converter.py --src_model_path paddle_ch_PP-OCRv3_rec_train_dir

#ppocr v4

# det v4
# ch_PP-OCRv4_det
python ./converter/ch_ppocr_v4_det_converter.py --yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml --src_model_path ch_PP-OCRv4_det_train_dir
# ch_PP-OCRv4_server_det
python ./converter/ch_ppocr_v4_det_server_converter.py --yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml --src_model_path ./models_v2.7/ppocrv4/ch_PP-OCRv4_det_server_train_dir

# rec v4
# ch_PP-OCRv4_rec
python ./converter/ch_ppocr_v4_rec_converter.py --yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml --src_model_path ch_PP-OCRv4_rec_train_dir
# ch_PP-OCRv4_server_rec
python ./converter/ch_ppocr_v4_rec_server_converter.py --yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml --src_model_path ch_PP-OCRv4_rec_server_train_dir

# PP-OCRv5
# PP-OCRv5_mobile_det
python ./converter/ppocr_v5_det_converter.py --yaml_path configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml --src_model_path PP-OCRv5_mobile_det_pretrained.pdparams
# PP-OCRv5_server_det
python ./converter/ppocr_v5_det_converter.py --yaml_path configs/det/PP-OCRv5/PP-OCRv5_server_det.yml --src_model_path PP-OCRv5_server_det_pretrained.pdparams
# PP-OCRv5_mobile_rec
python ./converter/ppocr_v5_rec_converter.py --yaml_path configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml --src_model_path PP-OCRv5_mobile_rec_pretrained.pdparams
# PP-OCRv5_server_rec
python ./converter/ppocr_v5_rec_converter.py --yaml_path configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml --src_model_path PP-OCRv5_server_rec_pretrained.pdparams

# PP-OCRv6
# 检测模型（tiny / small / medium 三档）
python ./converter/ppocr_v6_det_converter.py --yaml_path configs/det/PP-OCRv6/PP-OCRv6_tiny_det.yml --src_model_path PP-OCRv6_tiny_det_pretrained.pdparams
python ./converter/ppocr_v6_det_converter.py --yaml_path configs/det/PP-OCRv6/PP-OCRv6_small_det.yml --src_model_path PP-OCRv6_small_det_pretrained.pdparams
python ./converter/ppocr_v6_det_converter.py --yaml_path configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml --src_model_path PP-OCRv6_medium_det_pretrained.pdparams
# 识别模型（tiny / small / medium 三档）
python ./converter/ppocr_v6_rec_converter.py --yaml_path configs/rec/PP-OCRv6/PP-OCRv6_tiny_rec.yml --src_model_path PP-OCRv6_tiny_rec_pretrained.pdparams
python ./converter/ppocr_v6_rec_converter.py --yaml_path configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml --src_model_path PP-OCRv6_small_rec_pretrained.pdparams
python ./converter/ppocr_v6_rec_converter.py --yaml_path configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml --src_model_path PP-OCRv6_medium_rec_pretrained.pdparams
```

### 文档预处理模型

```bash
# 文档方向分类模型 (doc_ori)：判断文档的0°/90°/180°/270°旋转
python ./converter/pplcnet_cls_converter.py --yaml_path configs/cls/doc_ori/PP-LCNet_x1_0_doc_ori.yml --src_model_path PP-LCNet_x1_0_doc_ori_pretrained.pdparams

# 文本行方向分类模型 (textline_ori)：判断文本行0°/180°倒置
# 轻量版 (0.96MB)
python ./converter/pplcnet_cls_converter.py --yaml_path configs/cls/textline_ori/PP-LCNet_x0_25_textline_ori.yml --src_model_path PP-LCNet_x0_25_textline_ori_pretrained.pdparams
# 标准版 (6.5MB)
python ./converter/pplcnet_cls_converter.py --yaml_path configs/cls/textline_ori/PP-LCNet_x1_0_textline_ori.yml --src_model_path PP-LCNet_x1_0_textline_ori_pretrained.pdparams

# UVDoc 文档图像矫正模型：去弯曲/透视校正
python ./converter/uvdoc_converter.py --src_model_path UVDoc_pretrained.pdparams
```

<a name="多语言识别模型"></a>

### 多语言识别模型

```bash
python3 ./converter/multilingual_mobile_v2.0_rec_converter.py --src_model_path paddle_multilingual_mobile_v2.0_rec_train_dir

# v3
# en_PP-OCRv3_rec, multilingual_PP-OCRv3_rec
python ./converter/multilingual_ppocr_v3_rec_converter.py --src_model_path paddle_multilingual_PP-OCRv3_rec_train_dir

# v4
# en_PP-OCRv4_rec
python ./converter/ch_ppocr_v4_rec_converter.py --yaml_path ./configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml --src_model_path en_PP-OCRv4_rec_train_dir
```

<a name="文档结构化解析模型pp-structurev3"></a>

### 文档结构化解析模型（PP-StructureV3）

PP-StructureV3 是文档结构化解析系统，包含布局检测、OCR、表格识别三个模型。

**1. 布局检测模型转换**

```bash
# PP-DocLayout-S (轻量，1.2M)
python converter/ppstructure_layout_converter.py \
    --src_model_path=PP-DocLayout-S_pretrained.pdparams \
    --dst_model_path=ptocr_ppdoclayout_s.pth \
    --variant=S

# PP-DocLayout-M (推荐，5.8M)
python converter/ppstructure_layout_converter.py \
    --src_model_path=PP-DocLayout-M_pretrained.pdparams \
    --dst_model_path=ptocr_ppdoclayout_m.pth \
    --variant=M
```

**2. 表格结构识别模型转换**

```bash
python converter/ppstructure_slanext_converter.py \
    --src_model_path=SLANeXt_wired_pretrained.pdparams \
    --dst_model_path=ptocr_slanext_wired.pth
```

<a name="端到端模型"></a>

### 端到端模型

```bash
# en_server_pgnetA
python ./converter/e2e_converter.py --yaml_path ./configs/e2e/e2e_r50_vd_pg.yml --src_model_path your_ppocr_e2e_models_en_server_pgnetA_train_dir
```

<a name="超分辨率模型"></a>

### 超分辨率模型

```bash
# sr_telescope
python ./converter/sr_converter.py --yaml_path ./configs/sr/sr_telescope.yml --src_model_path your_ppocr_sr_telescope_train_dir
```

<a name="其他检测模型"></a>

### 其他检测模型

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

<a name="其他识别模型"></a>

### 其他识别模型

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

<a name="PyTorch推理"></a>

## 二、PyTorch推理

PyTorch模型下载链接：https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg 提取码：6clx

或者自行转换模型。

<a name="文本检测模型推理"></a>

### 文本检测模型推理

```bash
python3 ./tools/infer/predict_det.py --image_dir ./doc/imgs --model_path your_det_pth_path.pth

# v3
python ./tools/infer/predict_det.py --det_model_path your_ch_ptocr_v3_det_infer_path.pth --image_dir ./doc/imgs/1.jpg

# ppocrv4_det
python ./tools/infer/predict_det.py --image_dir ./doc/imgs/00009282.jpg --det_model_path your_ch_ptocr_v4_det_infer_path.pth --det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml

# ppocrv4_det server
python3 ./tools/infer/predict_det.py --image_dir ./doc/imgs/00009282.jpg --det_model_path your_ch_ptocr_v4_det_server_infer_path.pth --det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml

# PP-OCRv5
# PP-OCRv5_mobile_det
python ./tools/infer/predict_det.py --det_yaml_path configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml --det_model_path ./ptocr_v5_mobile_det.pth --image_dir ./doc/imgs/00009282.jpg
# PP-OCRv5_server_det
python ./tools/infer/predict_det.py --use_gpu false --det_algorithm DB --det_yaml_path configs/det/PP-OCRv5/PP-OCRv5_server_det.yml --det_model_path ./ptocr_v5_server_det.pth --image_dir ./doc/imgs/00009282.jpg

# PP-OCRv6
# PP-OCRv6_tiny_det
python ./tools/infer/predict_det.py --det_algorithm DB --det_yaml_path configs/det/PP-OCRv6/PP-OCRv6_tiny_det.yml --det_model_path ./models/v6/ptocr_v6_det_PP-OCRv6_tiny_det_pretrained.pth --image_dir ./doc/imgs/1.jpg
# PP-OCRv6_small_det
python ./tools/infer/predict_det.py --det_algorithm DB --det_yaml_path configs/det/PP-OCRv6/PP-OCRv6_small_det.yml --det_model_path ./models/v6/ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth --image_dir ./doc/imgs/1.jpg
# PP-OCRv6_medium_det
python ./tools/infer/predict_det.py --det_algorithm DB --det_yaml_path configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml --det_model_path ./models/v6/ptocr_v6_det_PP-OCRv6_medium_det_pretrained.pth --image_dir ./doc/imgs/1.jpg
```

![](../imgs_results/det_res_img_10_db.jpg)

![](../imgs_results/det_res_img623_sast.jpg)

#### 多语言检测模型

```bash
# v3
# en_ptocr_v3_det_infer.pth, multilingual_ptocr_v3_det_infer.pth
python ./tools/infer/predict_det.py --det_algorithm DB --det_yaml_path ./configs/det/det_ppocr_v3.yml --det_model_path your_multilingual_ptocr_v3_det_infer_path.pth --image_dir ./doc/imgs/1.jpg

# v4
# en_PP-OCRv4_rec
python ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words/en/word_1.png --rec_model_path your_en_ptocr_v4_rec_infer_path.pth --rec_yaml_path ./configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/en_dict.txt
```

<a name="文本识别模型推理"></a>

### 文本识别模型推理

#### 中英文模型

```bash
python3 ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words --model_path your_rec_pth_path.pth

# v3
python ./tools/infer/predict_rec.py --rec_model_path your_ch_ptocr_v3_rec_infer_path.pth --rec_image_shape 3,48,320 --image_dir ./doc/imgs_words/en/word_1.png

# ppocrv4_rec
python ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words/ch/word_1.jpg --rec_model_path your_ch_ptocr_v4_rec_infer_path.pth --rec_yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml --rec_image_shape='3,48,320'

# ppocrv4_rec server
python ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words/ch/word_1.jpg --rec_model_path your_ch_ptocr_v4_rec_server_infer_path.pth --rec_yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml --rec_image_shape='3,48,320'

# PP-OCRv5
# PP-OCRv5_mobile_rec
python ./tools/infer/predict_rec.py --rec_yaml_path configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv5_dict.txt --rec_model_path ./ptocr_v5_mobile_rec.pth --image_dir ./doc/imgs_words/ch/word_1.jpg
# PP-OCRv5/PP-OCRv5_server_rec
python ./tools/infer/predict_rec.py --use_gpu false --rec_yaml_path configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv5_dict.txt --rec_model_path ./ptocr_v5_server_rec.pth --image_dir ./doc/imgs_words/ch/word_1.jpg

# PP-OCRv6
# PP-OCRv6_tiny_rec（使用tiny字典，49语言）
python ./tools/infer/predict_rec.py --rec_algorithm CRNN --rec_yaml_path configs/rec/PP-OCRv6/PP-OCRv6_tiny_rec.yml --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv6_tiny_dict.txt --rec_model_path ./models/v6/ptocr_v6_rec_PP-OCRv6_tiny_rec_pretrained.pth --image_dir ./doc/imgs_words/ch/word_1.jpg
# PP-OCRv6_small_rec（使用v6标准字典，50语言）
python ./tools/infer/predict_rec.py --rec_algorithm CRNN --rec_yaml_path configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv6_dict.txt --rec_model_path ./models/v6/ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth --image_dir ./doc/imgs_words/ch/word_1.jpg
# PP-OCRv6_medium_rec（使用v6标准字典，50语言）
python ./tools/infer/predict_rec.py --rec_algorithm CRNN --rec_yaml_path configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv6_dict.txt --rec_model_path ./models/v6/ptocr_v6_rec_PP-OCRv6_medium_rec_pretrained.pth --image_dir ./doc/imgs_words/ch/word_1.jpg
```

![](../imgs_words/ch/word_4.jpg)

```
Predicts of ./doc/imgs_words/ch/word_4.jpg:('实力活力', 0.98458153)
```

#### 多语言识别模型

如果您需要预测的是其他语言模型，在使用inference模型预测时，需要通过`--rec_char_dict_path`指定使用的字典路径, 同时为了得到正确的可视化结果，
需要通过 `--vis_font_path` 指定可视化的字体路径，`doc/fonts/` 路径下有默认提供的小语种字体

```bash
# python3 ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words/spanish/es_1.jpg --rec_model_dir ../rec_models/multi_language/spanish/es_mobile_v2.0_rec_infer/ --rec_char_type your_multilingual_char_type --rec_char_dict_path ./ppocr/utils/dict/your_multilingual_dict.txt

python3 ./tools/infer/predict_rec.py --rec_model_path your_japan_mobile_v2.0_rec_infer_path.pth --rec_char_type japan --rec_char_dict_path ./pytorchocr/utils/dict/japan_dict.txt --image_dir ./doc/imgs_words/japan/1.jpg

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
python ./tools/infer/predict_rec.py --rec_model_path your_en_ptocr_v3_rec_infer_path.pth --rec_image_shape 3,48,320 --rec_yaml_path ./configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml --rec_char_dict_path ./pytorchocr/utils/en_dict.txt  --image_dir ./doc/imgs_words/en/word_1.png 

python ./tools/infer/predict_rec.py --rec_model_path your_japan_ptocr_v3_rec_infer_path.pth --rec_image_shape 3,48,320 --rec_yaml_path ./configs/rec/PP-OCRv3/multi_language/japan_PP-OCRv3_rec.yml --rec_char_dict_path ./pytorchocr/utils/dict/japan_dict.txt  --image_dir ./doc/imgs_words/japan/1.jpg
```

参考：[paddleocr.py](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/paddleocr.py#L283)

![](../imgs_words/korean/1.jpg)

```
Predicts of ./doc/imgs_words/korean/1.jpg:('바탕으로', 0.9948904)
```

<a name="文本方向分类模型推理"></a>

### 文本方向分类模型推理

```bash
python3 ./tools/infer/predict_cls.py --image_dir ./doc/imgs_words --model_path your_cls_pth_path.pth
```

![](../imgs_words/ch/word_1.jpg)

```
Predicts of ./doc/imgs_words/ch/word_4.jpg:['0', 0.9999982]
```

<a name="文档预处理模型推理"></a>

### 文档预处理模型推理

#### 文档方向分类模型推理 (doc_ori)

判断整张文档图像的旋转角度（0°/90°/180°/270°），用于旋转校正：

```bash
# doc_ori 使用 predict_cls.py，标签为 0, 90, 180, 270
python ./tools/infer/predict_cls.py \
    --cls_yaml_path configs/cls/doc_ori/PP-LCNet_x1_0_doc_ori.yml \
    --cls_model_path pretrained/PP-LCNet_x1_0_doc_ori_infer.pth \
    --cls_image_shape 3,224,224 \
    --label_list "0" "90" "180" "270" \
    --image_dir ./doc/imgs_words/ch/word_1.jpg
```

#### 文本行方向分类模型推理 (textline_ori)

判断单个文本行是否倒置（0°/180°），用于翻转校正：

```bash
# textline_ori 轻量版（0.96MB），标签为 0, 180
python ./tools/infer/predict_cls.py \
    --cls_yaml_path configs/cls/textline_ori/PP-LCNet_x0_25_textline_ori.yml \
    --cls_model_path pretrained/PP-LCNet_x0_25_textline_ori_infer.pth \
    --cls_image_shape 3,224,224 \
    --label_list "0" "180" \
    --image_dir ./doc/imgs_words/ch/word_1.jpg

# textline_ori 标准版（6.5MB）
python ./tools/infer/predict_cls.py \
    --cls_yaml_path configs/cls/textline_ori/PP-LCNet_x1_0_textline_ori.yml \
    --cls_model_path pretrained/PP-LCNet_x1_0_textline_ori_infer.pth \
    --cls_image_shape 3,224,224 \
    --label_list "0" "180" \
    --image_dir ./doc/imgs_words/ch/word_1.jpg
```

#### UVDoc 文档图像矫正模型推理

矫正弯曲/透视变形的文档图像。UVDoc 为独立模型，可通过 `test_new_models.py` 测试或参考以下代码使用：

```bash
# 运行测试脚本
python ./tools/test_new_models.py
```

```python
# 或通过 Python API 使用
from pytorchocr.modeling.architectures.uvdoc_model import UVDocModel
import torch, cv2
model = UVDocModel()
model.load_state_dict(torch.load('pretrained/UVDoc_infer.pth', weights_only=False))
model.eval()
img = cv2.cvtColor(cv2.imread('warped_doc.jpg'), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (712, 488))
x = torch.from_numpy(img.transpose(2,0,1).astype(np.float32)).unsqueeze(0)
unwarped, _ = model.unwarp(x)
cv2.imwrite('output.jpg', cv2.cvtColor(unwarped[0].permute(1,2,0).numpy().clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR))
```

<a name="文本检测、方向分类和文字识别串联推理"></a>

### 文本检测、方向分类和文字识别串联推理

#### 中英文模型推理

```bash
# 使用方向分类器
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path your_det_pth_path.pth --rec_model_path your_rec_pth_path.pth --use_angle_cls --cls_model_path your_cls_pth_path.pth --vis_font_path ./doc/fonts/your_lang_font.ttf

# 不使用方向分类器
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path your_det_pth_path.pth --rec_model_path your_rec_pth_path.pth

# v3
# other rec-models: use --rec_char_dict_path and --rec_yaml_path
python ./tools/infer/predict_system.py --image_dir ./doc/imgs/1.jpg --det_model_path your_ch_ptocr_v3_det_infer_path.pth --rec_image_shape 3,48,320 --rec_model_path your_ch_ptocr_v3_rec_infer_path.pth

# v4
# other det-models: use --det_yaml_path
# other rec-models: use --rec_char_dict_path, --rec_yaml_path and --rec_yaml_path
python ./tools/infer/predict_system.py --image_dir ./doc/imgs/1.jpg --det_model_path your_ch_ptocr_v4_det_infer_path.pth --det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml --rec_image_shape 3,48,320 --rec_model_path your_ch_ptocr_v4_rec_infer_path.pth --rec_yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml

# PP-OCRv5
# mobile
python ./tools/infer/predict_system.py --use_gpu false --det_yaml_path configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml --det_model_path ./ptocr_v5_mobile_det.pth --rec_yaml_path configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml --rec_model_path ./ptocr_v5_mobile_rec.pth --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv5_dict.txt  --image_dir ./doc/imgs/1.jpg
# server
python ./tools/infer/predict_system.py --use_gpu false --det_yaml_path configs/det/PP-OCRv5/PP-OCRv5_server_det.yml --det_model_path ./ptocr_v5_server_det.pth --rec_yaml_path configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml --rec_model_path ./ptocr_v5_server_rec.pth --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv5_dict.txt  --image_dir ./doc/imgs/1.jpg

# PP-OCRv6（检测+识别串联）
# tiny（端侧，~1.5M参数，极速推理）
python ./tools/infer/predict_system.py --use_gpu false --det_algorithm DB --det_yaml_path configs/det/PP-OCRv6/PP-OCRv6_tiny_det.yml --det_model_path ./models/v6/ptocr_v6_det_PP-OCRv6_tiny_det_pretrained.pth --rec_algorithm CRNN --rec_yaml_path configs/rec/PP-OCRv6/PP-OCRv6_tiny_rec.yml --rec_model_path ./models/v6/ptocr_v6_rec_PP-OCRv6_tiny_rec_pretrained.pth --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv6_tiny_dict.txt --image_dir ./doc/imgs/1.jpg
# small（移动端/桌面，~7.8M参数，平衡精度与速度）
python ./tools/infer/predict_system.py --use_gpu false --det_algorithm DB --det_yaml_path configs/det/PP-OCRv6/PP-OCRv6_small_det.yml --det_model_path ./models/v6/ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth --rec_algorithm CRNN --rec_yaml_path configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml --rec_model_path ./models/v6/ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv6_dict.txt --image_dir ./doc/imgs/1.jpg
# medium（服务端，~35M参数，最高精度）
python ./tools/infer/predict_system.py --use_gpu false --det_algorithm DB --det_yaml_path configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml --det_model_path ./models/v6/ptocr_v6_det_PP-OCRv6_medium_det_pretrained.pth --rec_algorithm CRNN --rec_yaml_path configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml --rec_model_path ./models/v6/ptocr_v6_rec_PP-OCRv6_medium_rec_pretrained.pth --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/dict/ppocrv6_dict.txt --image_dir ./doc/imgs/1.jpg
```

执行命令后，识别结果图像如下：

![](../../doc/imgs_results/system_res_00018069.jpg)

<a name="文档结构化解析推理pp-structurev3"></a>

### 文档结构化解析推理（PP-StructureV3）

完整的文档结构化解析管道，输入一张文档图片，输出 Markdown/JSON 结构化文档。

```bash
python ptstructure/predict_structure.py \
    --image_dir=./doc/table/ \
    --output_dir=./output/ \
    --layout_variant=M \
    --layout_score_thresh=0.2 \
    --layout_nms_thresh=0.5 \
    --det_model_path=./models/v6/ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth \
    --det_yaml_path=configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    --rec_model_path=./models/v6/ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth \
    --rec_yaml_path=configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
    --rec_char_dict_path=pytorchocr/utils/dict/ppocrv6_dict.txt \
    --table_model_path=./models/structurev3/ptocr_slanext_wired.pth
```

**管道流程**：输入图片 → 布局检测(PPDocLayout) → 文本OCR(PP-OCRv6) → 表格识别(SLANeXt) → 阅读顺序恢复 → Markdown/JSON

**参数说明**：
- `--layout_variant`: 布局模型变体 S(1.2M) / M(5.8M)
- `--layout_score_thresh`: 布局检测置信度阈值（默认0.2）
- `--layout_nms_thresh`: NMS IoU 阈值（默认0.5）

<a name="端到端模型推理"></a>

### 端到端模型推理

```bash
# en_server_pgnetA
python tools/infer/predict_e2e.py --e2e_model_path ./en_server_pgnetA_infer.pth --image_dir ./doc/imgs_en/img623.jpg --e2e_algorithm PGNet --e2e_pgnet_polygon True --e2e_char_dict_path ./pytorchocr/utils/ic15_dict.txt --e2e_yaml_path ./configs/e2e/e2e_r50_vd_pg.yml
```

![](../../doc/imgs_results/e2e_res_img623_pgnet.jpg)

<a name="超分辨率模型推理"></a>

### 超分辨率模型推理

```bash
# sr_telescope
python ./tools/infer/predict_sr.py --sr_yaml_path ./configs/sr/sr_telescope.yml --sr_model_path your_sr_telescope_infer_path.pth --image_dir
./doc/imgs_words_en/word_52.png
```

![](../../doc/imgs_words_en/word_52.png)

![](../../doc/imgs_results/sr/sr_word_52.png)

<a name="其他模型推理"></a>

### 其他模型推理

如果想尝试使用其他检测算法或者识别算法，请参考上述文本检测模型推理和文本识别模型推理，更新相应配置和模型。

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
python ./tools/infer/predict_det.py --det_algorithm DB --det_model_path your_xx_ptocr_v3_det_infer_path.pth --image_dir ./doc/imgs/1.jpg

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

### 参数列表

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
    parser.add_argument("--det_box_type", type=str, default="quad")

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
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
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
        "--e2e_char_dict_path", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'pytorchocr/utils/ic15_dict.txt'))
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_polygon", type=bool, default=True)
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # SR parmas
    parser.add_argument("--sr_model_path", type=str)
    parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    parser.add_argument("--sr_batch_num", type=int, default=1)

    #
    parser.add_argument("--draw_img_save_dir", type=str, default="./inference_results")
    parser.add_argument("--save_crop_res", type=str2bool, default=False)
    parser.add_argument("--crop_res_save_dir", type=str, default="./output")

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

    # extended function
    parser.add_argument(
        "--return_word_box",
        type=str2bool,
        default=False,
        help="Whether return the bbox of each word (split by space) or chinese character. Only used in ppstructure for layout recovery",
    )

    return parser
```

<a name="参考"></a>

## 参考

- [PaddleOCR release/2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_ch/inference.md)
- [PaddleOCR release/2.1](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/inference.md)
- [PaddleOCR release/2.5](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/inference_ppocr.md)
- [PaddleOCR release/2.6](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/inference_ppocr.md)
- [PaddleOCR release/2.7](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/inference_ppocr.md)