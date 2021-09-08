# [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)

English | [简体中文](README.md)

## Introduction
Converting PaddleOCR to PyTorch.

This repository aims to 

- learn PaddleOCR
- use models in PyTorch which are trained in Paddle
- give a guideline for Paddle2PyTorch

## TODO

- [ ] PP-OCRv2. The inference speed of PP-OCRv2 is 220% higher than that of PP-OCR server in CPU device. The F-score of PP-OCRv2 is 7% higher than that of PP-OCR mobile.
- [ ] a new structured documents analysis toolkit, i.e., [PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/ppstructure/README.md), support layout analysis and table recognition (One-key to export chart images to Excel files).

## Notice

`PytorchOCR` models are converted from `PaddleOCRv2.0`.

**Recent updates**

- 2021.06.01 update SRN
- 2021.04.25 update AAAI 2021 end-to-end algorithm PGNet
- 2021.04.24 update RARE
- 2021.04.12 update STARNET
- 2021.04.08 update DB, SAST, EAST, ROSETTA, CRNN
- 2021.04.03 update more than 25+ multilingual recognition models [models list](./doc/doc_en/models_list_en.md), including：English, Chinese, German, French, Japanese，Spanish，Portuguese Russia Arabic and so on.  Models for more languages will continue to be updated [Develop Plan](https://github.com/PaddlePaddle/PaddleOCR/issues/1048).
- 2021.01.10 upload Chinese and English general OCR models.

## Features
- PTOCR series of high-quality pre-trained models, comparable to commercial effects
    - Ultra lightweight ptocr_mobile series models
    - General ptocr_server series models
    - Support Chinese, English, and digit recognition, vertical text recognition, and long text recognition
    - Support multi-language recognition: Korean, Japanese, German, French, etc.

## [Model List](./doc/doc_en/models_list_en.md) (updating)

PyTorch models in BaiduPan：https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg code：6clx

PaddleOCR models in BaiduPan：https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g code：lmv7 

If you want to get more models including multilingual models，please refer to [PTOCR  series](./doc/doc_en/models_list_en.md).

## Tutorials
- [Installation](./doc/doc_en/installation_en.md)
- [Inferences](./doc/doc_en/inference_en.md)
- [PP-OCR Pipeline](#PP-OCR-Pipeline)
- [Visualization](#Visualization)
- [Reference documents](./doc/doc_en/reference_en.md)
- [FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_en/FAQ_en.md)
- [References](#References)



<a name="PP-OCR-Pipeline"></a>

## PP-OCR Pipeline

<div align="center">
    <img src="./doc/framework.png" width="800">
</div>


PP-OCR is a practical ultra-lightweight OCR system. It is mainly composed of three parts: DB text detection[2], detection frame correction and CRNN text recognition[7]. The system adopts 19 effective strategies from 8 aspects including backbone network selection and adjustment, prediction head design, data augmentation, learning rate transformation strategy, regularization parameter selection, pre-training model use, and automatic model tailoring and quantization to optimize and slim down the models of each module. The final results are an ultra-lightweight Chinese and English OCR model with an overall size of 3.5M and a 2.8M English digital OCR model. For more details, please refer to the PP-OCR technical article (https://arxiv.org/abs/2009.09941). Besides, The implementation of the FPGM Pruner [8] and PACT quantization [9] is based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim).


## Visualization
- Chinese OCR model
<div align="center">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/11.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/00015504.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/00056221.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/1.jpg" width="800">
</div>


- English OCR model
<div align="center">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/img_12.jpg" width="800">
</div>


- Multilingual OCR model
<div align="center">
    <img src="./doc/imgs_results/french_0.jpg" width="800">
    <img src="./doc/imgs_results/korean.jpg" width="800">
</div>


<a name="Reference"></a>

## References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR)
- [Paddle](https://github.com/PaddlePaddle)
- [Pytorch](https://pytorch.org/)
- [https://github.com/frotms/image_classification_pytorch](https://github.com/frotms/image_classification_pytorch)
- [https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_ch/models_list.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_ch/models_list.md)