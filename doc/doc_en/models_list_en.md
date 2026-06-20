## OCR Model List

**Note**: PyTorch `.pth` model download link: https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg passcode: 6clx

PaddleOCR model download link: https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g passcode: lmv7

- [1. Text Detection Models](#text-detection-models)
- [2. Text Recognition Models](#text-recognition-models)
- [3. Text Direction Classification Models](#text-direction-classification-models)
- [4. Document Preprocessing Models](#document-preprocessing-models)
- [5. Document Structure Analysis Models (PP-StructureV3)](#document-structure-analysis-models)

<a name="text-detection-models"></a>
## 1. Text Detection Models

| Model | Description | Size | Download |
| --- | --- | --- | --- |
| PP-OCRv6_medium_det | Medium model, PPLCNetV4 + RepLKPAN, 50 languages | 61M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams) / [Converter](../../converter/ppocr_v6_det_converter.py) |
| PP-OCRv6_small_det | Small model, PPLCNetV4 + RepLKFPN, 50 languages | 9.8M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_det_pretrained.pdparams) / [Converter](../../converter/ppocr_v6_det_converter.py) |
| PP-OCRv6_tiny_det | Tiny model, PPLCNetV4 + RepLKFPN(k=5), 49 languages | 1.9M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_det_pretrained.pdparams) / [Converter](../../converter/ppocr_v6_det_converter.py) |
| PP-OCRv5_server_det | Server model, PPLCNetV4 + RepLKPAN | 110M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams) / [Converter](../../converter/ppocr_v5_det_converter.py) |
| PP-OCRv5_mobile_det | Mobile model | 4.7M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams) / [Converter](../../converter/ppocr_v5_det_converter.py) |
| ch_PP-OCRv4_det | Ultra-lightweight | 4.7M | [Training](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar) |
| ch_PP-OCRv4_server_det | High-precision | 110M | [Training](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_train.tar) |
| ch_PP-OCRv3_det | Original lightweight | 3.8M | [Training](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) |

<a name="text-recognition-models"></a>
## 2. Text Recognition Models

| Model | Description | Size | Download |
| --- | --- | --- | --- |
| PP-OCRv6_medium_rec | Medium model, 50 languages | 73M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams) / [Converter](../../converter/ppocr_v6_rec_converter.py) |
| PP-OCRv6_small_rec | Small model, 50 languages | 20M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams) / [Converter](../../converter/ppocr_v6_rec_converter.py) |
| PP-OCRv6_tiny_rec | Tiny model, 49 languages | 4.3M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_rec_pretrained.pdparams) / [Converter](../../converter/ppocr_v6_rec_converter.py) |
| PP-OCRv5_server_rec | Server model, SVTR_HGNet | 99M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams) / [Converter](../../converter/ppocr_v5_rec_converter.py) |
| PP-OCRv5_mobile_rec | Mobile model, SVTR_LCNet | 15M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams) / [Converter](../../converter/ppocr_v5_rec_converter.py) |
| ch_PP-OCRv4_rec | Ultra-lightweight | 8.5M | [Training](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar) |
| ch_PP-OCRv4_server_rec | High-precision | 88M | [Training](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_train.tar) |

<a name="text-direction-classification-models"></a>
## 3. Text Direction Classification Models

| Model | Description | Size | Download |
| --- | --- | --- | --- |
| ch_ppocr_mobile_v2.0_cls | Original model | 1.38M | [Training](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |

<a name="document-preprocessing-models"></a>
## 4. Document Preprocessing Models

| Model | Description | Size | Download |
| --- | --- | --- | --- |
| PP-LCNet_x1_0_doc_ori | Document orientation (0°/90°/180°/270°) | 7M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams) / [Converter](../../converter/pplcnet_cls_converter.py) |
| PP-LCNet_x0_25_textline_ori | Text line orientation (0°/180°), ultra-lightweight | 0.96M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams) / [Converter](../../converter/pplcnet_cls_converter.py) |
| PP-LCNet_x1_0_textline_ori | Text line orientation (0°/180°) | 6.5M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_textline_ori_pretrained.pdparams) / [Converter](../../converter/pplcnet_cls_converter.py) |
| UVDoc | Document unwarping (CGU-Net), SIGGRAPH Asia 2023 | 30.3M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams) / [Converter](../../converter/uvdoc_converter.py) |

<a name="document-structure-analysis-models"></a>
## 5. Document Structure Analysis Models (PP-StructureV3)

### 5.1 Layout Detection Models

| Model | Description | Params | Download |
| --- | --- | --- | --- |
| PP-DocLayout-M | **[Recommended]** PicoDet, LCNet(scale=2.0)+LCPAN+PicoHeadV2, 23 layout classes | 5.8M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams) / [Converter](../../converter/ppstructure_layout_converter.py) |
| PP-DocLayout-S | Lightweight PicoDet, LCNet(scale=0.75)+LCPAN+PicoHeadV2, 23 layout classes | 1.2M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams) / [Converter](../../converter/ppstructure_layout_converter.py) |

### 5.2 Table Structure Recognition Models

| Model | Description | Params | Download |
| --- | --- | --- | --- |
| SLANeXt_wired | ViT Encoder + GRU Attention Decoder, HTML output | ~90M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams) / [Converter](../../converter/ppstructure_slanext_converter.py) |

### 5.3 Formula Recognition Models

| Model | Description | Params | Download |
| --- | --- | --- | --- |
| PP-FormulaNet_plus-M | **[Recommended]** PPHGNetV2_B6 + MBart Decoder (6 layers), LaTeX output | ~250M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams) / [Converter](../../converter/ppstructure_formula_converter.py) |
| PP-FormulaNet-S | Lightweight, PPHGNetV2_B4 + MBart Decoder (2 layers), LaTeX output | ~100M | [Training](https://paddleocr.bj.bcebos.com/pretrained/PP-FormulaNet-S_pretrained.pdparams) / [Converter](../../converter/ppstructure_formula_converter.py) |

### 5.4 Seal Text Detection Models

| Model | Description | Params | Download |
| --- | --- | --- | --- |
| PP-OCRv4_mobile_seal_det | Seal text detection, PPLCNetV3+RSEFPN+DBHead | ~1.5M | [Training](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams) / [Converter](../../converter/ppstructure_seal_converter.py) |

### 5.5 Pipeline Usage

```bash
python ptstructure/predict_structure.py \
    --image_dir=./doc/table/ \
    --output_dir=./output/ \
    --layout_variant=M \
    --use_formula --use_seal
```

See [PP-StructureV3 Porting Guide](../../skills/ppstructurev3_porting_guide.md) for details.
