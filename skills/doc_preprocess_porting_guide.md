# PaddleOCR 文档预处理模型移植指南

> 将 PaddleOCR v3.x 的 textline_ori、doc_ori 和 UVDoc 三个文档预处理模型移植到 PaddleOCR2Pytorch。

---

## 一、模型概述

| 模型 | 功能 | 骨干网络 | 分类 | 大小 |
|------|------|---------|------|------|
| PP-LCNet_x0_25_textline_ori | 文本行方向分类（0°/180°） | PP-LCNet v1 (scale=0.25) | 2类 | 0.96 MB |
| PP-LCNet_x1_0_textline_ori | 文本行方向分类（0°/180°） | PP-LCNet v1 (scale=1.0) | 2类 | 6.5 MB |
| PP-LCNet_x1_0_doc_ori | 文档方向分类（0°/90°/180°/270°） | PP-LCNet v1 (scale=1.0) | 4类 | 7 MB |
| UVDoc (CGU-Net) | 文档图像去扭曲矫正 | 全卷积双任务网络 | 回归 | 30.3 MB |

---

## 二、架构说明

### 2.1 PP-LCNet v1 骨干网络

PP-LCNet v1 是轻量级 CNN，专门为 CPU 推理优化。关键设计：

- **DepthSepConv** 基础块（无残差连接）：DW Conv → BN → H-Swish → PW Conv → BN → H-Swish
- **SE 模块**：仅最后两个 Block 使用，reduction=4
- **大卷积核**：中后部 Block 使用 5×5 DW Conv
- **1280 维投影**：GAP → 1×1 Conv(→1280) + H-Swish → Dropout

textline_ori 和 doc_ori 使用 `BaseModel` 架构组装：
- Backbone: PPLCNet (含 GAP + 1280 投影)
- Head: ClsHead (AdaptiveAvgPool2d + Linear)

### 2.2 UVDoc CGU-Net

全卷积双任务网络，不遵循 BaseModel 范式：

- **编码器**：2 层 Conv 下采样 + 3 级空洞残差块（总共 1+3+4+6=14 个 res block）
- **空间金字塔**：6 路并行空洞卷积（dilation=1,2,4,8,16,32），其中后 3 路各含 3 个 Conv+BN 子块
- **双输出头**：2D 矫正网格（45×31×2）+ 3D 形状网格（45×31×3）
- **后处理**：双线性插值上采样 + grid_sample

---

## 三、使用方法

### 3.1 权重转换

```bash
# textline_ori (2类, 轻量版)
python converter/pplcnet_cls_converter.py \
    --yaml_path=configs/cls/textline_ori/PP-LCNet_x0_25_textline_ori.yml \
    --src_model_path=pretrained/PP-LCNet_x0_25_textline_ori_pretrained.pdparams

# doc_ori (4类)
python converter/pplcnet_cls_converter.py \
    --yaml_path=configs/cls/doc_ori/PP-LCNet_x1_0_doc_ori.yml \
    --src_model_path=pretrained/PP-LCNet_x1_0_doc_ori_pretrained.pdparams

# UVDoc
python converter/uvdoc_converter.py \
    --src_model_path=pretrained/UVDoc_pretrained.pdparams
```

### 3.2 推理

```python
# textline_ori / doc_ori (使用 BaseModel)
from pytorchocr.modeling.architectures.base_model import BaseModel
import yaml, torch

cfg = yaml.safe_load(open('configs/cls/textline_ori/PP-LCNet_x0_25_textline_ori.yml'))
model = BaseModel(cfg['Architecture'])
model.load_state_dict(torch.load('pretrained/PP-LCNet_x0_25_textline_ori_infer.pth'))
model.eval()

# UVDoc (独立模型)
from pytorchocr.modeling.architectures.uvdoc_model import UVDocModel
model = UVDocModel()
model.load_state_dict(torch.load('pretrained/UVDoc_infer.pth'))
model.eval()

result = model(x)
unwarped, grid = model.unwarp(x)  # 直接得到矫正图像
```

---

## 四、新增文件清单

| 文件 | 说明 |
|------|------|
| `pytorchocr/modeling/backbones/rec_pplcnet.py` | PP-LCNet v1 骨干网络实现 |
| `pytorchocr/modeling/architectures/uvdoc_model.py` | UVDoc CGU-Net 完整实现 |
| `configs/cls/textline_ori/*.yml` | textline_ori YAML 配置 |
| `configs/cls/doc_ori/*.yml` | doc_ori YAML 配置 |
| `converter/pplcnet_cls_converter.py` | PP-LCNet 权重转换脚本 |
| `converter/uvdoc_converter.py` | UVDoc 权重转换脚本 |
| `tools/test_new_models.py` | 模型测试脚本 |

---

## 五、关键实现细节

### PP-LCNet SE 模块位置
SE 位于 DW Conv 和 PW Conv 之间，作用于**中间通道**（in_channels）。

### UVDoc 权重键名匹配
PaddlePaddle 和 PyTorch 的权重键名结构高度一致，通过匹配后缀（weight/bias/running_mean/running_var）实现自动转换。特定块（bridge 和 resnet blocks）使用额外的 Sequential 包装以保持与 Paddle 键名的兼容。

### UVDoc resnet_head 索引问题
PaddlePaddle 的 Sequential 索引有间隔（0,1,3,4），跳过了 ReLU 激活层。PyTorch 必须使用 `._modules['key']` 而非 `[index]` 方式访问。

---

## 六、ptstructure 集成

文档预处理模型已集成到 `ptstructure/doc_preprocess/`，可在管道中直接使用：

### Python API

```python
from ptstructure.doc_preprocess import DocOrientationClassifier, UVDocUnwarper

# 文档方向分类
ori = DocOrientationClassifier()
ori.load_weights('models/structurev3/ptocr_doc_ori.pth')
label, score = ori.classify(img)       # → ('90', 0.95)
img = ori.correct_orientation(img)     # 自动纠正

# 文档去扭曲
uw = UVDocUnwarper()
uw.load_weights('models/structurev3/ptocr_uvdoc.pth')
corrected = uw.unwarp(img)             # → 矫正后图像
```

### CLI

```bash
python ptstructure/predict_structure.py \
    --image_dir=./doc/imgs/ \
    --use_doc_orientation --use_doc_unwarping \
    --use_angle_cls
```

### 文件位置

```
ptstructure/doc_preprocess/
├── __init__.py
├── doc_orientation.py    # DocOrientationClassifier
└── unwarp.py             # UVDocUnwarper
```

详见 [PP-StructureV3 移植指南](ppstructurev3_porting_guide.md)
