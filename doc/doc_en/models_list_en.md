## OCR model list
**Note** : **PyTorch models in BaiduPan: https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg code：6clx**

PaddleOCR models in BaiduPan:  https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g code：lmv7 

- [1. Text Detection Model](#Detection)
- [2. Text Recognition Model](#Recognition)
    - [Chinese Recognition Model](#Chinese)
    - [English Recognition Model](#English)
    - [Multilingual Recognition Model](#Multilingual)
- [3. Text Angle Classification Model](#Angle)

The downloadable models provided by PaddleOCR include `inference model`, `trained model`, `pre-trained model` and `slim model`. **`slim models` are not supported here**. The differences between the models are as follows:

|model type|model format|description|
|--- | --- | --- |
|inference model|inference.pdmodel、inference.pdiparams|Used for reasoning based on Python prediction engine，[detail](./inference_en.md)|
|trained model, pre-trained model|\*.pdparams、\*.pdopt、\*.states |The checkpoints model saved in the training process, which stores the parameters of the model, mostly used for model evaluation and continuous training.|

<a name="Detection"></a>
## 1. Text Detection Model

### Chinese Detection Model

|model name|description|model size|download|
| --- | --- | --- | --- |
|ch_PP-OCRv4_det|[New] Original lightweight model, supporting Chinese, English, multilingual text detection|4.70M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar)|
|ch_PP-OCRv4_server_det|[New] Original general model, supporting Chinese, English, multilingual text detection|110M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_train.tar)|
|ch_PP-OCRv3_det|Original lightweight model, supporting Chinese, English, multilingual text detection|3.8M|inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar)|
|ch_ppocr_mobile_v2.0_det|Original lightweight model, supporting Chinese, English, multilingual text detection|3M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|
|ch_ppocr_server_v2.0_det|General model, which is larger than the lightweight model, but achieved better performance|47M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)|
|ch_PP-OCRv2_det|Original lightweight model, supporting Chinese, English, multilingual text detection|3M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar)|

### English Detection Model

| model name      | description                                                  | model size | download                                                     |
| --------------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| en_PP-OCRv3_det | [New] Original lightweight detection model, supporting English | 3.8M       | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) |

### Multilingual Detection Model

| model name      | description                                                  | model size | download                                                     |
| --------------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| ml_PP-OCRv3_det | [New] Original lightweight detection model, supporting English | 3.8M       | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) |

<a name="Recognition"></a>

## 2. Text Recognition Model

<a name="Chinese"></a>
### Chinese Recognition Model

|model name|description|model size|download|
| --- | --- | --- | --- |
|ch_PP-OCRv4_rec|[New] Original lightweight model, supporting Chinese, English, multilingual text recognition|10M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar)|
|ch_PP-OCRv4_server_rec|[New] Original general model, supporting Chinese, English, multilingual text recognition|88M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_train.tar)|
|ch_PP-OCRv3_rec|Original lightweight model, supporting Chinese, English, multilingual text recognition|12.4M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar)|
|ch_ppocr_mobile_v2.0_rec|Original lightweight model, supporting Chinese, English and number recognition|3.71M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
|ch_ppocr_server_v2.0_rec|General model, supporting Chinese, English and number recognition|94.8M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |
|ch_PP-OCRv2_rec|Original lightweight model, supporting Chinese, English, multilingual text detection|8.5M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |

**Note:** The `trained model` is finetuned on the `pre-trained model` with real data and synthsized vertical text data, which achieved better performance in real scene. The `pre-trained model` is directly trained on the full amount of real data and synthsized data, which is more suitable for finetune on your own dataset.

<a name="English"></a>

### English Recognition Model

|model name|description|model size|download|
| --- | --- | --- | --- |
|en_PP-OCRv4_rec|[New] Original lightweight model, supporting english, English, multilingual text recognition|9.7M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar)|
|en_PP-OCRv3_rec|Original lightweight model, supporting english, English, multilingual text recognition|9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar)|
|en_number_mobile_v2.0_rec|Original lightweight model, supporting English and number recognition(v2.0)|2.56M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |
|en_number_mobile_v2.0_rec|Original lightweight model, supporting English and number recognition(v2.1 the performance is optimized)|2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |

<a name="Multilingual"></a>

### Multilingual Recognition Model（Updating...）

#### [release/2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_en/models_list_en.md#2-generate-italian-configuration-file-to-train-your-own-data)

|model name|description|model size|download|
| --- | --- | --- | --- |
| french_mobile_v2.0_rec |Lightweight model for French recognition|2.65M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar) |
| german_mobile_v2.0_rec |Lightweight model for German recognition|2.65M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar) |
| korean_mobile_v2.0_rec |Lightweight model for Korean recognition|3.9M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar) |
| japan_mobile_v2.0_rec |Lightweight model for Japanese recognition|4.23M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar) |
| it_mobile_v2.0_rec |Lightweight model for Italian recognition|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_train.tar) |
| es_mobile_v2.0_rec |Lightweight model for Spanish recognition|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/es_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/es_mobile_v2.0_rec_train.tar) |
| pt_mobile_v2.0_rec |Lightweight model for Portuguese recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pt_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pt_mobile_v2.0_rec_train.tar) |
| ru_mobile_v2.0_rec |Lightweight model for Russia recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_train.tar) |
| ar_mobile_v2.0_rec |Lightweight model for Arabic recognition|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_train.tar) |
| hi_mobile_v2.0_rec |Lightweight model for Hindi recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_train.tar) |
| ch_tra_mobile_v2.0_rec |Lightweight model for chinese traditional recognition|5.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ch_tra_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ch_tra_mobile_v2.0_rec_train.tar) |
| ug_mobile_v2.0_rec |Lightweight model for Uyghur recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_train.tar) |
| fa_mobile_v2.0_rec |Lightweight model for Persian recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_train.tar) |
| ur_mobile_v2.0_rec |Lightweight model for Urdu recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_train.tar) |
| rs_latin_mobile_v2.0_rec |Lightweight model for Serbian(latin) recognition|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_latin_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_latin_mobile_v2.0_rec_train.tar) |
| oc_mobile_v2.0_rec |Lightweight model for Occitan recognition|2.53M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_train.tar) |
| mr_mobile_v2.0_rec |Lightweight model for Marathi recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_train.tar) |
| ne_mobile_v2.0_rec |Lightweight model for Nepali recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_train.tar) |
| rs_cyrillic_mobile_v2.0_rec |Lightweight model for Serbian(cyrillic) recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_cyrillic_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_cyrillic_mobile_v2.0_rec_train.tar) |
| bg_mobile_v2.0_rec |Lightweight model for Bulgarian recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_train.tar) |
| uk_mobile_v2.0_rec |Lightweight model for Ukranian recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_train.tar) |
| be_mobile_v2.0_rec |Lightweight model for Belarusian recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_train.tar) |
| te_mobile_v2.0_rec |Lightweight model for Telugu recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar) |
| kn_mobile_v2.0_rec |Lightweight model for Kannada recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/kn_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/kn_mobile_v2.0_rec_train.tar) |
| ta_mobile_v2.0_rec |Lightweight model for Tamil recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar) |

#### [release/2.1](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/models_list_en.md#2-generate-italian-configuration-file-to-train-your-own-data)

| model name                  | description                                   | model size | download                                                     |
| --------------------------- | --------------------------------------------- | ---------- | ------------------------------------------------------------ |
| french_mobile_v2.0_rec      | Lightweight model for French recognition      | 2.65M      | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar) |
| german_mobile_v2.0_rec      | Lightweight model for German recognition      | 2.65M      | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar) |
| korean_mobile_v2.0_rec      | Lightweight model for Korean recognition      | 3.9M       | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar) |
| japan_mobile_v2.0_rec       | Lightweight model for Japanese recognition    | 4.23M      | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar) |
| chinese_cht_mobile_v2.0_rec | Lightweight model for chinese cht recognition | 5.63M      | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_train.tar) |
| te_mobile_v2.0_rec          | Lightweight model for Telugu recognition      | 2.63M      | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar) |
| ka_mobile_v2.0_rec          | Lightweight model for Kannada recognition     | 2.63M      | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_train.tar) |
| ta_mobile_v2.0_rec          | Lightweight model for Tamil recognition       | 2.63M      | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar) |
| latin_mobile_v2.0_rec       | Lightweight model for latin recognition       | 2.6M       | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_train.tar) |
| arabic_mobile_v2.0_rec      | Lightweight model for arabic recognition      | 2.6M       | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_train.tar) |
| cyrillic_mobile_v2.0_rec    | Lightweight model for cyrillic recognition    | 2.6M       | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_train.tar) |
| devanagari_mobile_v2.0_rec  | Lightweight model for devanagari recognition  | 2.6M       | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_train.tar) |

For more supported languages, please refer to : 

- [Multi-language model](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/multi_languages_en.md)
- [lang](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/paddleocr.py#L283)

<a name="Angle"></a>

## 3. Text Angle Classification Model

|model name|description|model size|download|
| --- | --- | --- | --- |
|ch_ppocr_mobile_v2.0_cls|Original model|1.38M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |
