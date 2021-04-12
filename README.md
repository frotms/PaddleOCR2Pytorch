# [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)

简体中文 | [English](README_en.md)

## 简介
**”白嫖“**[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)。

本项目旨在：

- 学习PaddleOCR
- 让PaddleOCR训练的模型在pytorch上使用
- 为paddle转pytorch提供参考

## TODO

- [ ] AAAI 2021论文端到端识别算法PGNet
- [ ] 其他文本识别模型: RARE, SRN

## 注意

`PytorchOCR`由`PaddleOCRv2.0`动态图版本移植。

**近期更新**

- 2021.4.12 更新STARNET
- 2021.4.8 更新DB, SAST, EAST, ROSETTA, CRNN
- 2021.4.3 更新多语言识别模型，目前支持语种超过27种，[多语言模型下载](./doc/doc_ch/models_list.md)，包括中文简体、中文繁体、英文、法文、德文、韩文、日文、意大利文、西班牙文、葡萄牙文、俄罗斯文、阿拉伯文等，后续计划可以参考[多语言研发计划](https://github.com/PaddlePaddle/PaddleOCR/issues/1048)
- 2021.1.10 白嫖中英文通用OCR模型

## 特性

高质量推理模型，准确的识别效果

- 超轻量ptocr_mobile移动端系列
- 通用ptocr_server系列
- 支持中英文数字组合识别、竖排文本识别、长文本识别
- 支持多语言识别：韩语、日语、德语、法语等

<a name="模型下载"></a>

## [模型列表](./doc/doc_ch/models_list.md)（更新中）

PyTorch模型下载链接：https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg 提取码：6clx

PaddleOCR模型百度网盘链接：https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g 提取码：lmv7 

更多模型下载（包括多语言），可以参考[PT-OCR v2.0 系列模型下载](./doc/doc_ch/models_list.md)

## 文档教程
- [快速安装](./doc/doc_ch/installation.md)
- [模型预测](./doc/doc_ch/inference.md)
- [Pipline](#Pipline)
- [效果展示](#效果展示)
- [参考文献](./doc/doc_ch/reference.md)
- [FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_en/FAQ_en.md)
- [参考](#参考)

<a name="Pipline"></a>

## PP-OCR Pipline
<div align="center">
    <img src="./doc/framework.png" width="800">
</div>


PP-OCR是一个实用的超轻量OCR系统。主要由DB文本检测[2]、检测框矫正和CRNN文本识别三部分组成[7]。该系统从骨干网络选择和调整、预测头部的设计、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型自动裁剪量化8个方面，采用19个有效策略，对各个模块的模型进行效果调优和瘦身，最终得到整体大小为3.5M的超轻量中英文OCR和2.8M的英文数字OCR。更多细节请参考PP-OCR技术方案 https://arxiv.org/abs/2009.09941 。其中FPGM裁剪器[8]和PACT量化[9]的实现可以参考[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)。

<a name="效果展示"></a>

## 效果展示
- 中文模型
<div align="center">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/11.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/00015504.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/00056221.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/1.jpg" width="800">
</div>


- 英文模型
<div align="center">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/img_12.jpg" width="800">
</div>


- 其他语言模型
<div align="center">
    <img src="./doc/imgs_results/french_0.jpg" width="800">
    <img src="./doc/imgs_results/korean.jpg" width="800">
</div>

<a name="参考"></a>

## 参考

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR)
- [Paddle](https://github.com/PaddlePaddle)
- [Pytorch](https://pytorch.org/)
- [https://github.com/frotms/image_classification_pytorch](https://github.com/frotms/image_classification_pytorch)
- [https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_ch/models_list.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_ch/models_list.md)