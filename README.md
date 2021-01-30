# [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)

## 简介

**”真·白嫖“**[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## 注意

`PytorchOCR`由`PaddleOCR-2.0rc1+`动态图版本移植。

## 特性

高质量推理模型，准确的识别效果

- 超轻量ptocr_mobile移动端系列
- 通用ptocr_server系列
- 支持中英文数字组合识别、竖排文本识别、长文本识别

## 模型列表

`.pth`模型下载链接：https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg 提取码：6clx 

### 检测模型

| 模型名称                             | PaddleOCR对应模型                                            |
| ------------------------------------ | ------------------------------------------------------------ |
| `ch_ptocr_mobile_v2.0_det_infer.pth` | ch_ppocr_mobile_v2.0_det: [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar) |
| `ch_ptocr_server_v2.0_det_infer.pth` | ch_ppocr_server_v2.0_det: [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar) |

### 识别模型

| 模型名称                             | PaddleOCR对应模型                                            |
| ------------------------------------ | ------------------------------------------------------------ |
| `ch_ptocr_mobile_v2.0_rec_infer.pth` | ch_ppocr_mobile_v2.0_rec: [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) |
| `ch_ptocr_server_v2.0_rec_infer.pth` | ch_ppocr_server_v2.0_rec: [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) |

### 分类模型

| 模型名称                             | PaddleOCR对应模型                                            |
| ------------------------------------ | ------------------------------------------------------------ |
| `ch_ptocr_mobile_v2.0_cls_infer.pth` | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |

## 效果展示

- 中文模型

<div align="center">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/11.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/00015504.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/00056221.jpg" width="800">
    <img src="./doc/imgs_results/ch_ptocr_mobile_v2.0/1.jpg" width="800">
</div>



## Pipeline

<div align="center">
    <img src="./doc/framework.png" width="800">
</div>

PP-OCR是一个实用的超轻量OCR系统。主要由DB文本检测、检测框矫正和CRNN文本识别三部分组成。该系统从骨干网络选择和调整、预测头部的设计、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型自动裁剪量化8个方面，采用19个有效策略，对各个模块的模型进行效果调优和瘦身，最终得到整体大小为3.5M的超轻量中英文OCR和2.8M的英文数字OCR。更多细节请参考PP-OCR技术方案 https://arxiv.org/abs/2009.09941 。其中FPGM裁剪器和PACT量化的实现可以参考[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)。

## 推理

### 环境

```
shapely
numpy
pillow
pyclipper
opencv-python <= 4.2.0.32
pytorch
```

### 中文检测模型推理

```
python3 ./tools/infer/predict_det.py --image_dir ./doc/imgs --model_path your_det_pth_path.pth
```

### 中文识别模型推理

```
python3 ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words --model_path your_rec_pth_path.pth
```

### 中文方向分类模型推理

```
python3 ./tools/infer/predict_cls.py --image_dir ./doc/imgs_words --model_path your_cls_pth_path.pth
```

### 文本检测、方向分类和文字识别串联推理

#### 使用方向分类器

```
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path your_det_pth_path.pth --rec_model_path your_rec_pth_path.pth --use_angle_cls --cls_model_path your_cls_pth_path.pth
```

#### 不适用方向分类器

```
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path your_det_pth_path.pth --rec_model_path your_rec_pth_path.pth
```

## PaddleOCR2Pytorch

### 环境

```
shapely
numpy
pillow
pyclipper
opencv-python <= 4.2.0.32
pytorch
paddlepaddle==2.0rc1
```

### 模型转换

**转换模型使用PaddleOCR的*训练模型***。

模型路径详见**PaddleOCR对应模型**或者**百度网盘链接**：https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g 
提取码：lmv7 

```
python3 ./converter/ch_ppocr_mobile_v2.0_det_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_det_train_dir

python3 ./converter/ch_ppocr_server_v2.0_det_converter.py --src_model_path paddle_ch_ppocr_server_v2.0_det_train_dir

python3 ./converter/ch_ppocr_mobile_v2.0_rec_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_rec_train_dir

python3 ./converter/ch_ppocr_server_v2.0_rec_converter.py --src_model_path paddle_ch_ppocr_server_v2.0_rec_train_dir

python3 ./converter/ch_ppocr_mobile_v2.0_cls_converter.py --src_model_path paddle_ch_ppocr_mobile_v2.0_cls_train_dir
```

## [FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0-rc1-0/doc/doc_ch/FAQ.md)

## 参考

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR)
- [Paddle](https://github.com/PaddlePaddle)
- [Pytorch](https://pytorch.org/)
- [https://github.com/frotms/image_classification_pytorch](https://github.com/frotms/image_classification_pytorch)
- [https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0-rc1-0/doc/doc_ch/models_list.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0-rc1-0/doc/doc_ch/models_list.md)