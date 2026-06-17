# PP-OCRv6 移植到 PytorchOCR 完整指南

> 本文档描述如何将 PaddleOCR 官方的 PP-OCRv6 模型移植到 PytorchOCR 仓库，包括模型转换和推理。

## 一、PP-OCRv6 简介

PP-OCRv6 是百度 PaddlePaddle 团队于 2025 年 6 月发布的新一代 OCR 系统，提供 **三档模型**：

| 档位 | 参数规模 | 适用场景 | 检测 Hmean | 识别 Acc |
|------|---------|----------|-----------|----------|
| Tiny | 1.5M | 端侧/IoT/浏览器 | 80.6% | 73.5% |
| Small | 7.7M | 移动端/桌面端 | 84.1% | 81.3% |
| Medium | 34.5M | 服务端 | 86.2% | 83.2% |

### 核心架构创新

- **统一骨干网络 PPLCNetV4**：MetaFormer 范式 + RepDWConv 结构重参数化
- **RepLKFPN 检测颈**：大核膨胀可重参数卷积，参数减少 31%，感受野从 3×3 → 7×7
- **EncoderWithLightSVTR 识别颈**：局部-全局注意力 + 加性跳跃连接
- **多语言统一**：单模型支持 50 种语言（中文、英文、日文 + 46 种拉丁语系）

## 二、目录结构

移植后的文件组织：

```
PaddleOCR2Pytorch/
├── configs/
│   ├── det/PP-OCRv6/          # 检测模型配置
│   │   ├── PP-OCRv6_tiny_det.yml
│   │   ├── PP-OCRv6_small_det.yml
│   │   └── PP-OCRv6_medium_det.yml
│   └── rec/PP-OCRv6/          # 识别模型配置
│       ├── PP-OCRv6_tiny_rec.yml
│       ├── PP-OCRv6_small_rec.yml
│       └── PP-OCRv6_medium_rec.yml
├── pytorchocr/
│   ├── modeling/
│   │   ├── backbones/
│   │   │   └── rec_lcnetv4.py        # PPLCNetV4 骨干网络
│   │   └── necks/
│   │       ├── db_fpn.py             # 含 RepLKFPN, RepLKPAN, DilatedReparamBlock
│   │       └── rnn.py                # 含 EncoderWithLightSVTR
│   └── utils/dict/
│       ├── ppocrv6_dict.txt          # v6 标准字典 (18708 字符)
│       └── ppocrv6_tiny_dict.txt     # v6 Tiny 字典 (6902 字符)
├── converter/
│   ├── ppocr_v6_det_converter.py     # 检测模型转换脚本
│   └── ppocr_v6_rec_converter.py     # 识别模型转换脚本
└── models/v6/                        # 转换后的 PyTorch 权重
```

## 三、新增组件说明

### 3.1 PPLCNetV4 骨干网络 (`rec_lcnetv4.py`)

统一检测/识别骨干，核心结构：

```
LCNetV4Block:
  Token Mixer: RepDWConv(3x3) 或 普通 DW Conv
  Channel Mixer: Expand(2x) → GELU → Compress + 残差连接
```

**关键特性**：
- **RepDWConv**：训练时 3 分支（3x3 DW + 1x1 DW + Identity BN），推理时合并为单个 3x3 DW Conv
- **任务自适应步长**：检测模式 stride {4,8,16,32}，识别模式 stage 3-4 用非对称步长 (2,1)
- **BN 零初始化**：Channel Mixer 的 Compress 层 BN 初始化为 0，稳定训练

### 3.2 RepLKFPN 检测颈 (`db_fpn.py`)

基于 RSEFPN 优化，使用 `DilatedReparamBlock` 替换 3x3 标准卷积：

- 训练时：7x7 DW Conv + 3 个膨胀分支（5x5 dil=1, 3x3 dil=2, 3x3 dil=3）
- 推理时：所有分支合并为单个 7x7 DW Conv
- 参数减少 31%，感受野从 3×3 扩大到 7×7

### 3.3 RepLKPAN 检测颈 (`db_fpn.py`)

基于 LKPAN 优化，使用 `DilatedReparamConv`（DW 大核重参数 + 1x1 PW）：

- 替换 8 个 9x9 标准卷积为 DilatedReparamConv
- 参数减少 ~96%（训练时），推理时无额外开销

### 3.4 EncoderWithLightSVTR 识别颈 (`rnn.py`)

轻量 SVTR 变体：

- 局部上下文：1x7 深度卷积 → SiLU 激活
- 全局建模：1-2 层 Transformer（Multi-Head Self-Attention + MLP）
- 加性跳跃连接：1x1 卷积后将输入特征加到输出上

## 四、模型转换流程

### 步骤 1：下载 PaddlePaddle 预训练权重

从 PaddleOCR 官方获取 `.pdparams` 训练权重：

```bash
# 检测模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_det_pretrained.pdparams
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_det_pretrained.pdparams

# 识别模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_rec_pretrained.pdparams
```

### 步骤 2：转换检测模型

```bash
cd PaddleOCR2Pytorch

# 转换 PP-OCRv6 small 检测模型
python converter/ppocr_v6_det_converter.py \
    --yaml_path=configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    --src_model_path=./models/v6/PP-OCRv6_small_det_pretrained.pdparams

# 输出：ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth
```

### 步骤 3：转换识别模型

```bash
# 转换 PP-OCRv6 small 识别模型
python converter/ppocr_v6_rec_converter.py \
    --yaml_path=configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
    --src_model_path=./models/v6/PP-OCRv6_small_rec_pretrained.pdparams

# 输出：ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth
```

### 权重映射规则

| Paddle 参数 | PyTorch 参数 | 处理方式 |
|------------|-------------|----------|
| `._mean` | `.running_mean` | 重命名 |
| `._variance` | `.running_var` | 重命名 |
| `head.aux_binarize_p*` | (丢弃) | 辅助训练分支，推理不需要 |
| `head.aux_thresh_p*` | (丢弃) | 辅助训练分支，推理不需要 |
| `head.gtc_head.*` (rec) | (丢弃) | 蒸馏辅助头 |

## 五、推理使用

### 5.1 使用测试脚本

```bash
# 从项目根目录
cd PaddleOCR2Pytorch
python ../tests/v6_test/test_inference.py
```

测试脚本会自动：
1. 加载转换后的检测和识别模型
2. 对 `tests/v6_test/images/` 目录下的所有图片进行 OCR
3. 将标注结果保存到 `tests/v6_test/results/`

### 5.2 Python API 示例

```python
import torch
import cv2
from pytorchocr.modeling.architectures.base_model import BaseModel

# 加载模型
model = BaseModel(config)
state_dict = torch.load('ptocr_v6_det_example.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# 预处理
img = cv2.imread('test.jpg')
# ... 归一化等预处理 ...

# 推理
with torch.no_grad():
    output = model(input_tensor)
```

### 5.3 预处理参数

**检测模型**：
- 输入尺寸：动态，自动 resize 到最长边 640，padding 到 32 的倍数
- 归一化：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- 输出：[1, 1, H, W] 概率图

**识别模型**：
- 输入尺寸：高度 48，宽度按比例缩放到 320 以内
- 归一化：mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
- 输出：[1, W/8, num_classes] CTC 概率分布

### 5.4 后处理参数

**检测后处理**（DBPostProcess）：
- thresh=0.2
- box_thresh=0.45
- unclip_ratio=1.4
- max_candidates=3000

**识别后处理**（CTCLabelDecode）：
- CTC greedy decode
- 移除重复和空白标记

## 六、测试结果

### 测试环境
- CPU: Intel Xeon
- Python 3.8
- PyTorch 2.4.1 (CPU)
- 模型：PP-OCRv6 Small (检测 2.5M + 识别 5.3M)

### 测试结果

#### 测试图片 1：中文产品标签 (500x500)
| 检测文本 | 置信度 |
|---------|--------|
| 纯臻宫养护发素 | 0.971 |
| 产品信息/参数 | 0.955 |
| 【净含量】:220ml | 0.980 |
| 【品名】·纯臻宫养护发素 | 0.955 |
| 【产品编号】：YM-X-3011 | 0.947 |
| 【适用人群】：适合所有联质 | 0.941 |
| 【主要成分】：鲸蜡硬脂醇、燕麦B-葡聚 | 0.950 |
| 【主要功能】：可紧致头发磷层，从而达到 | 0.950 |
| (成品包材) | 0.946 |
| 糖，椰油酰胺内基胡菜碱，泛醍 | 0.942 |
| 前时持久改善头发光泽的效果，给干燥的头 | 0.946 |
| 发足够的滋养 | 0.959 |
| 【品牌】:代加工方式/OFMODM | 0.941 |
| 每瓶22元，1000瓶起门) | 0.946 |
| (45元/每公斤，100公斤起订) | 0.963 |

- 检测时间：255.8ms
- 识别时间：275.8ms

#### 测试图片 2：门牌/地址 (720x1150)
| 检测文本 | 置信度 |
|---------|--------|
| 上海斯格威铂尔大酒店 | 0.940 |
| 绿洲仕格维花园公寓 | 0.932 |
| 打浦路252935号 | 0.902 |
| 打浦路15号 | 0.877 |
| 一 | 0.674 |

- 检测时间：162.8ms
- 识别时间：91.5ms

### 性能分析

PP-OCRv6 Small 在 CPU 上的推理速度：
- 检测：~160-260ms（取决于图像大小）
- 识别：~15-20ms/每行文本
- 端到端：~250-530ms（取决于图像中文本区域数量）

## 七、常见问题

### Q1：转换时出现 numpy._core 错误
PaddlePaddle 3.0+ 使用 numpy 2.x 的 pickle 格式。如果使用 numpy < 2.0：
- 已在 `base_ocr_v20.py` 中添加了兼容性补丁
- 如果仍有问题，升级 numpy：`pip install numpy>=2.0`（需要 Python >= 3.9）

### Q2：识别结果不准确
- 确保使用正确的字典文件 (`ppocrv6_dict.txt`)
- 检查预处理归一化参数
- 确保 CTC 解码时正确处理 blank token (index=0)

### Q3：辅助头参数缺失
检测模型的训练权重包含 `aux_binarize_p*` 和 `aux_thresh_p*` 辅助头参数。转换脚本会自动过滤这些参数，推理时不需要这些分支。

## 八、模型大小对比

| 模型 | 检测参数 | 识别参数 | 总参数 |
|------|---------|---------|--------|
| PP-OCRv6 Tiny | ~0.5M | ~1.0M | ~1.5M |
| PP-OCRv6 Small | 2.51M | 5.28M | 7.79M |
| PP-OCRv6 Medium | ~16M | ~19M | ~35M |

## 参考链接

- [PaddleOCR 官方仓库](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv6 论文](https://arxiv.org/abs/2606.13108)
- [PaddleOCR 模型下载](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/)
- [PytorchOCR 仓库](https://github.com/frotms/PaddleOCR2Pytorch)
