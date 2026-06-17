# PaddleOCR 模型移植到 PaddleOCR2Pytorch 工作流程

> 本文档描述如何将 PaddleOCR 官方模型（PP-OCRv2/v3/v4/v5/v6）移植到 PaddleOCR2Pytorch 仓库中。

---

## 一、仓库架构总览

### 1.1 目录结构

```
PaddleOCR2Pytorch/
├── configs/                    # YAML 配置文件，按任务和版本组织
│   ├── det/                    #   文字检测配置
│   │   ├── ch_ppocr_v2.0/      #     v2 检测
│   │   ├── ch_PP-OCRv3/        #     v3 检测
│   │   ├── ch_PP-OCRv4/        #     v4 检测
│   │   └── PP-OCRv5/           #     v5 检测
│   ├── rec/                    #   文字识别配置
│   │   ├── ch_ppocr_v2.0/      #     v2 识别
│   │   ├── PP-OCRv3/           #     v3 识别
│   │   ├── PP-OCRv4/           #     v4 识别
│   │   └── PP-OCRv5/           #     v5 识别
│   ├── cls/                    #   方向分类器配置
│   ├── e2e/                    #   端到端配置
│   ├── sr/                     #   超分模型配置
│   └── table/                  #   表格识别配置
├── pytorchocr/                 # 核心 PyTorch 模型实现
│   ├── modeling/
│   │   ├── architectures/      #   顶层模型组装（BaseModel）
│   │   ├── backbones/          #   骨干网络实现
│   │   ├── necks/              #   颈部（FPN, SequenceEncoder 等）
│   │   ├── heads/              #   检测/识别/分类头
│   │   └── transforms/         #   空间变换（TPS, STN 等）
│   ├── postprocess/            # 后处理（DB, CTC, EAST 等）
│   ├── data/                   # 数据预处理/增强
│   ├── utils/                  # 工具函数和字典文件
│   └── base_ocr_v20.py         # 模型加载基类
├── converter/                  # 权重转换脚本
├── tools/infer/                # 推理入口脚本
├── ptstructure/                # PP-Structure（表格/版面/VQA）
└── misc/                       # 杂项辅助代码
```

### 1.2 核心设计模式：工厂注册

仓库使用**工厂模式**动态构建模型组件。每个组件目录的 `__init__.py` 中定义了 `build_xxx(config)` 函数，根据配置中的 `name` 字段动态实例化对应的类：

| 组件 | 构建函数 | 所在文件 | 注册方式 |
|------|----------|----------|----------|
| Transform | `build_transform(config)` | `transforms/__init__.py` | `name` → 类名 eval |
| Backbone | `build_backbone(config, model_type)` | `backbones/__init__.py` | `name` → 类名 eval，按 model_type 分流 |
| Neck | `build_neck(config)` | `necks/__init__.py` | `name` → 类名 eval |
| Head | `build_head(config)` | `heads/__init__.py` | `name` → 类名 eval |
| Model | `build_model(config)` | `architectures/__init__.py` | 统一使用 BaseModel |

### 1.3 BaseModel 数据流

```
输入图像
  ↓
Transform (可选: TPS/STN)     ← 对图像做空间矫正
  ↓
Backbone (必需)               ← 提取特征图
  ↓
Neck (可选: FPN/SequenceEncoder) ← 特征增强/序列化
  ↓
Head (必需: DBHead/CTCHead)   ← 输出预测结果
  ↓
输出 (dict 或 tensor)
```

---

## 二、各版本模型架构对照

### 2.1 文字检测模型 (det)

| 版本 | Backbone | Neck | Head | 主要特点 |
|------|----------|------|------|----------|
| PP-OCRv2 | MobileNetV3 (scale=0.5) | DBFPN (out=96) | DBHead | 轻量级 CML 蒸馏 |
| PP-OCRv3 | MobileNetV3 (scale=0.5) | RSEFPN (out=96) | DBHead | 引入 SE 注意力颈部 |
| PP-OCRv4 mobile | PPLCNetV3 (scale=0.75) | RSEFPN (out=96) | DBHead | 新轻量骨干 |
| PP-OCRv4 server | PPHGNetV2_B4 | LKPAN (out=256) | PFHeadLocal | 大模型 + PAN 颈部 |
| PP-OCRv5 mobile | PPLCNetV3 (scale=0.75) | RSEFPN (out=96) | DBHead (fix_nan) | 继承 v4 mobile |
| PP-OCRv5 server | PPHGNetV2_B4 | LKPAN (out=256, intracl) | PFHeadLocal | 继承 v4 server，新增 IntraCL |
| PP-OCRv6 small | PPLCNetV4 | RSEFPN / RepLKFPN | DBHead | 新骨干 PPLCNetV4，neck 支持大核卷积 |
| PP-OCRv6 medium | PPLCNetV4 (更大) | RSEFPN / RepLKFPN | DBHead | 同上，更大模型 |

### 2.2 文字识别模型 (rec)

| 版本 | Backbone | Neck | Head | 主要特点 |
|------|----------|------|------|----------|
| PP-OCRv2 | MobileNetV3 (scale=0.5) | SequenceEncoder (rnn) | CTCHead | CRNN 架构 |
| PP-OCRv3 | MobileNetV1Enhance (scale=0.5) | SequenceEncoder (svtr) | CTCHead | SVTR 编码器替代 RNN |
| PP-OCRv4 mobile | PPLCNetV3 (scale=0.95) | — (内置在 Head) | MultiHead (CTC+NRTR) | LCNet + SVTR Neck |
| PP-OCRv4 server | PPHGNetV2_B4 | — (内置在 Head) | MultiHead (CTC+NRTR) | HGNet + SVTR Neck |
| PP-OCRv5 mobile | PPLCNetV3 (scale=0.95) | — (内置在 Head) | MultiHead (CTC+NRTR) | 继承 v4，扩大字典 |
| PP-OCRv5 server | PPHGNetV2_B4 | — (内置在 Head) | MultiHead (CTC+NRTR) | 继承 v4，扩大字典 |
| PP-OCRv6 small | PPLCNetV4 | — (LightSVTR 内置在 Head) | MultiHead (CTC+NRTR) | 新骨干 + 轻量 SVTR 变体 |
| PP-OCRv6 medium | PPLCNetV4 (更大) | — (LightSVTR 内置在 Head) | MultiHead (CTC+NRTR) | 同上，更大模型 |

> **注意**：v4/v5/v6 识别模型不使用独立的 Neck，其 SVTR/LightSVTR 编码器和 CTC Head 合并定义在 `MultiHead` 中。MultiHead 内部包含一个 CTC 分支（SVTR/LightSVTR 编码器 + CTC Head）和一个可选的 GTC 分支（NRTR/SAR Head）。

### 2.3 方向分类器 (cls)

各版本均使用：
- **Backbone**: MobileNetV3 (scale=0.35)
- **Neck**: 无
- **Head**: ClsHead (class_dim=2，输出 0° 或 180°)

---

## 三、权重转换机制

### 3.1 转换器基类 `BaseOCRV20`

位于 `pytorchocr/base_ocr_v20.py`，提供：

- `build_net(config, **kwargs)` — 使用 `BaseModel` 构建 PyTorch 网络
- `read_paddle_weights(path)` — 读取 PaddlePaddle `.pdparams` 权重文件（兼容新旧 API）
- `save_pytorch_weights(path)` — 保存 PyTorch `.pth` 权重文件
- `load_pytorch_weights(path)` — 加载已有的 PyTorch 权重
- `print_pytorch_state_dict()` / `print_paddle_state_dict()` — 调试用

### 3.2 转换脚本模式

每个 `converter/` 中的脚本遵循固定模式：

```
1. 解析 YAML 配置文件 → 获取 Architecture 字典
2. 调用 paddle.load() 读取 PaddlePaddle 权重
3. 用 Architecture 配置构建 PyTorch BaseModel
4. 逐层映射权重：Paddle 参数名 → PyTorch 参数名
5. 处理特殊情况（线性层转置、蒸馏权重过滤等）
6. 校验推理结果一致性
7. 保存 PyTorch 权重为 .pth 文件
```

### 3.3 关键权重映射规则

| PaddlePaddle | PyTorch | 说明 |
|-------------|---------|------|
| `._mean` | `.running_mean` | BatchNorm 均值 |
| `._variance` | `.running_var` | BatchNorm 方差 |
| `fc.weight` | `fc.weight.T` | 全连接层需要转置 |
| `fc1.weight`, `fc2.weight` | 转置 | TPS 定位网络的 FC |
| `qkv.weight`, `proj.weight` | 转置 | 自注意力 QKV 投影 |
| `Student.xxx` / `Student2.xxx` | `xxx` | 蒸馏模型去除 Student 前缀 |
| `Teacher.xxx` | (丢弃) | 蒸馏模型丢弃 Teacher 权重 |
| `head.sar_head.xxx` / `head.gtc_head.xxx` | (丢弃) | 蒸馏辅助头（推理时不需要） |

---

## 四、完整移植工作流程

### 步骤 1：确定目标模型并收集信息

从 [PaddleOCR 官方仓库](https://github.com/PaddlePaddle/PaddleOCR) 确定要移植的模型：

1. **找到 PaddleOCR 训练配置文件**（`.yml`），位于 `PaddleOCR/configs/` 对应子目录
2. **从配置文件中提取 `Architecture` 部分**，记录：
   - `model_type`：det / rec / cls
   - `algorithm`：DB / CRNN / SVTR 等
   - `Transform` 及其参数
   - `Backbone` 的 `name` 和参数
   - `Neck` 的 `name` 和参数
   - `Head` 的 `name` 和参数
3. **获取预训练权重文件**（`.pdparams`），通常从 PaddleOCR 官方 Model Zoo 下载

### 步骤 2：检查 PyTorch 组件是否已实现

对照 PaddleOCR 配置中的每个组件，检查本仓库是否已有对应的 PyTorch 实现：

**检查 Backbone 注册表** (`pytorchocr/modeling/backbones/__init__.py`)：
- 检测类：MobileNetV3, ResNet, ResNet_vd, ResNet_SAST, PPLCNetV3, PPHGNet_small, PPHGNetV2_B4
- 识别类：MobileNetV1Enhance, MobileNetV3, ResNet, ResNetFPN, MTB, ResNet31, SVTRNet, ViTSTR, DenseNet, PPLCNetV3, PPHGNet_small, PPHGNetV2_B4

**检查 Neck 注册表** (`pytorchocr/modeling/necks/__init__.py`)：
- DBFPN, RSEFPN, LKPAN, FPN, EASTFPN, SASTFPN, SequenceEncoder, PGFPN, FCEFPN, TableFPN

**检查 Head 注册表** (`pytorchocr/modeling/heads/__init__.py`)：
- DBHead, PFHeadLocal, EASTHead, SASTHead, PSEHead, FCEHead, PGHead
- CTCHead, AttentionHead, SRNHead, Transformer, SARHead, CANHead, MultiHead
- ClsHead, TableAttentionHead

**检查 Transform 注册表** (`pytorchocr/modeling/transforms/__init__.py`)：
- TPS, STN_ON, TSRN, TBSRN

#### 如果缺组件 → 参见"步骤 A：实现新的 PyTorch 组件"

### 步骤 3：在 configs 目录创建 YAML 配置文件

在 `configs/<det|rec|cls>/<版本名>/` 下创建 YAML 配置，包含：

```yaml
# 参考 PaddleOCR 原始配置文件，提取以下部分：

Global:
  model_name: <模型名称>
  character_dict_path: <字典路径（rec用）>
  max_text_length: &max_text_length 25    # rec 用
  use_space_char: true

Architecture:
  model_type: det          # det / rec / cls
  algorithm: DB            # 算法标识
  Transform: null          # 或 TPS 等配置
  Backbone:
    name: PPLCNetV3        # 必须与注册表中的名称一致
    scale: 0.75            # 具体参数
    det: True              # 检测模式标志
  Neck:
    name: RSEFPN           # 必须与注册表中的名称一致
    out_channels: 96
    shortcut: True
  Head:
    name: DBHead           # 必须与注册表中的名称一致
    k: 50

# 以下是训练/推理也需要但移植时可选的部分：
Loss: ...
Optimizer: ...
PostProcess: ...
Metric: ...
Train: ...
Eval: ...
```

**关键原则**：YAML 中的 `Architecture` 部分必须与本仓库的组件注册系统完全对应，`name` 字段值必须是注册表中已注册的名称。

### 步骤 4：编写权重转换脚本

在 `converter/` 目录下创建转换脚本，参考现有模式（如 `ppocr_v5_det_converter.py`）：

#### 4.1 创建 Converter 类

继承 `BaseOCRV20`，主要需要实现 `load_paddle_weights()` 方法：

```python
class PPOCRVxDetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super().__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()
```

#### 4.2 实现权重名称映射

根据 PaddlePaddle 权重文件的实际键名，编写映射逻辑：

1. **先打印两边的 state_dict 键名**，使用调试函数 `print_paddle_state_dict()` 和 `print_pytorch_state_dict()`
2. **建立映射规则**，常见情况：
   - 直接同名映射（大多数情况）
   - BatchNorm 统计量重命名
   - 全连接层权重转置
   - 蒸馏模型前缀剥离
   - 无关参数的跳过（Teacher 分支、蒸馏辅助头等）

3. **注意 PaddlePaddle 与 PyTorch 的差异**：
   - Paddle 的 `paddle.nn.Linear` 权重形状是 `(out_features, in_features)`
   - PyTorch 的 `nn.Linear` 权重形状是 `(out_features, in_features)`
   - 两者形状相同，**通常不需要转置**
   - 但当 Paddle OCR 使用 `fluid.layers.fc` 或者权重的语义是 `(in, out)` 时需要转置
   - **实践**：逐层对比形状，形状一致则直接拷贝，不一致则转置

#### 4.3 处理蒸馏权重

很多 PaddleOCR 模型使用蒸馏训练，权重文件中同时包含 Student 和 Teacher 的参数：

- **只保留 Student 分支**的权重用于推理
- Teacher 权重、SAR Head、GTC Head 等辅助分支需要**丢弃**
- 参考 `ch_ppocr_v3_rec_converter.py` 中的 `del_invalid_state_dict()` 方法

#### 4.4 处理 MultiHead 的输出通道数

对于使用 `MultiHead` 的识别模型（v4/v5），需要从字典文件确定输出类别数：

```python
# 读取字典
with open(character_dict_path) as f:
    char_num = len(f.readlines()) + 1  # +1 for blank
# 设置 MultiHead 的输出通道
config['Architecture']['Head']['out_channels_list'] = {
    'CTCLabelDecode': char_num,
    'SARLabelDecode': char_num + 2,
    'NRTRLabelDecode': char_num + 3,
}
```

### 步骤 5：验证转换正确性

#### 5.1 数值一致性验证

用相同的随机输入，对比 PaddlePaddle 模型和 PyTorch 模型的输出：

1. 用 `np.random.seed(666)` 固定随机种子生成测试输入
2. 分别运行 PaddlePaddle 和 PyTorch 模型的前向推理
3. 比较输出的 `sum`、`mean`、`max`、`min`
4. 应该达到 float32 精度下的数值一致（误差 < 1e-5）

#### 5.2 推理验证

使用 `tools/infer/predict_system.py` 对实际图像进行推理：

```bash
python tools/infer/predict_system.py \
    --image_dir=./doc/imgs/ \
    --det_model_path=./ch_ptocr_v4_det_infer.pth \
    --rec_model_path=./ch_ptocr_v4_rec_infer.pth \
    --det_yaml_path=./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml \
    --rec_yaml_path=./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml
```

#### 5.3 注册到推理工具

在 `tools/infer/pytorchocr_utility.py` 的 `AnalysisConfig()` 函数中添加新权重的自动配置识别：

- 根据权重文件名自动匹配对应的 Architecture 配置
- 这样使用时就不需要手动指定 `--xxx_yaml_path`

### 步骤 6：更新文档和模型列表

1. 更新 `doc/doc_ch/models_list.md` 中的模型列表
2. 更新 `README.md` 中的更新日志
3. 将 PyTorch 权重上传到网盘

### 6.1 PaddleOCR 上游已存在但本仓库尚未移植的组件

根据对 PaddleOCR 上游仓库的分析，以下组件在官方 PaddleOCR 中已实现但本仓库可能尚未移植：

**Backbone（识别类）**：
- `PPLCNetV4` — PP-OCRv6 使用的新一代轻量骨干
- `SVTRv2`、`RepSVTR` — SVTR 的改进/重参数化变体
- `PPHGNetV2_B6` — 比 B4 更大的 HGNet v2
- `ViTParseQ`、`HybridTransformer` — Transformer 类骨干
- `Vary_VIT_B` — 可变尺寸 ViT
- `DonutSwinModel` — Swin Transformer 变体
- `MicroNet` — 微型网络

**Backbone（检测类）**：
- `PPLCNetV4`、`PPLCNetV2_base`
- `RepSVTR_det` — 检测专用 SVTR

**Neck**：
- `LightSVTR` — PP-OCRv6 中使用的轻量 SVTR 编码器（Neck 的一种变体）
- `RepLKFPN`、`RepLKPAN` — 带重参数化大核卷积的 FPN
- `CSPPAN` — CSP 风格的 PAN
- `CTFPN` — 中心文本 FPN
- `RFAdaptor` — 注意力适配器

**Head**：
- `ABINetHead`、`VisionLanHead`、`RobustScannerHead` — 先进的识别头
- `AsterHead` — Aster 矫正+识别头
- `CT_Head`、`DRRGHead` — 其他检测头

---

## 步骤 A：实现新的 PyTorch 组件（当缺少时）

当 PaddleOCR 模型使用了本仓库尚未实现的 Backbone/Neck/Head 时，需要新建 PyTorch 实现。

### A.1 查找 PaddlePaddle 原始实现

在 [PaddleOCR 仓库](https://github.com/PaddlePaddle/PaddleOCR) 中找到对应源码：

- Backbone: `PaddleOCR/ppocr/modeling/backbones/`
- Neck: `PaddleOCR/ppocr/modeling/necks/`
- Head: `PaddleOCR/ppocr/modeling/heads/`

### A.2 编写 PyTorch 版本

在对应目录下创建新的 `.py` 文件，遵循以下规范：

1. **类名和接口与 PaddlePaddle 版本保持一致**
2. **使用 PyTorch 等效 API 替换 PaddlePaddle API**，常见映射：

   | PaddlePaddle | PyTorch |
   |-------------|---------|
   | `paddle.nn.Conv2D` | `nn.Conv2d` |
   | `paddle.nn.BatchNorm2D` | `nn.BatchNorm2d` |
   | `paddle.nn.Linear` | `nn.Linear` |
   | `paddle.nn.LayerNorm` | `nn.LayerNorm` |
   | `paddle.nn.functional.relu` | `F.relu` |
   | `paddle.reshape / paddle.transpose` | `tensor.reshape / tensor.permute` |
   | `paddle.concat` | `torch.cat` |
   | `paddle.mean` | `torch.mean` |
   | `paddle.nn.functional.interpolate` | `F.interpolate` |
   | `paddle.fluid.dygraph.guard` | 不需要 |
   | `ParamAttr(initializer=...)` | 在 `__init__` 中用 `nn.init.*` 替代 |
   | `paddle.zeros / paddle.ones` | `torch.zeros / torch.ones` |

3. **注意维度差异**：
   - PaddlePaddle 默认 `NCHW`，与 PyTorch 相同
   - PaddlePaddle 的 LSTM 默认 `batch_first=False`（time-first）
   - Paddle 的 `transpose` 参数是 `[0, 2, 1]` 形式的 perm 列表

4. **注意算子差异**：
   - Paddle 的 `hardswish`/`hard_sigmoid` 与 PyTorch 版本行为可能略有差异
   - BatchNorm 的 `momentum` 含义不同（Paddle 是 1-momentum）
   - 需要特别注意这些细节以确保权重完全兼容

### A.3 注册新组件

在对应的 `__init__.py` 中注册：

1. **导入新类**
2. **将类名加入 `support_dict` 列表**
3. **如需区分 model_type（Backbone），在各分支中添加**

示例 — 在 `backbones/__init__.py` 中添加新 Backbone：

```python
# 添加导入
from .rec_newbackbone import NewBackbone

# 在相应 model_type 分支的 support_dict 中添加
support_dict = [
    ...现有名称...,
    'NewBackbone',   # 新增
]
```

### A.4 关键：确保权重键名兼容

PyTorch 模型的 `state_dict()` 键名必须能够与 PaddlePaddle 的键名建立明确映射。这意味着：

- **模块层级结构要与 PaddlePaddle 版本一致**
- **变量命名风格要匹配**（或提供完整的映射表）
- 如果是 `BaseModel` 组装的模型，键名会自动带上 `backbone.` / `neck.` / `head.` 前缀

---

## 五、常见问题与调试技巧

### 5.1 权重加载失败

**症状**：`KeyError` 或 shape mismatch

**调试步骤**：
1. 打印 PaddlePaddle 权重所有键名和形状
2. 打印 PyTorch 模型所有键名和形状
3. 逐层对比，找出不匹配的层
4. 确定是需要重命名还是转置

### 5.2 推理结果不一致

**症状**：相同输入，PyTorch 输出与 PaddlePaddle 输出差异大

**排查方向**：
1. 激活函数差异（hardswish 实现细节）
2. BatchNorm momentum 累积差异（推理时一般无影响，因为使用 running stats）
3. 算子实现差异（如 interpolate 的 align_corners）
4. 权重转换时的 transpose 遗漏

**调试技巧**：
- 逐层输出中间特征，对比 PaddlePaddle 和 PyTorch
- 定位到第一个出现差异的层

### 5.3 蒸馏模型处理

蒸馏模型的权重前缀规则：

| PaddleOCR 版本 | Student 前缀 | 需要丢弃的部分 |
|---------------|-------------|---------------|
| PP-OCRv3 检测 | `Student2.` | Teacher 相关全部 |
| PP-OCRv3 识别 | `Student.` | `Teacher.`、`head.sar_head.`、`head.gtc_head.` |
| PP-OCRv4 检测 CML | `Student.` + `Teacher.` | 选择保留 Student |

### 5.4 MultiHead 模型的特殊处理

PP-OCRv4/v5 的识别模型使用 `MultiHead`，其 Neck（SVTR 编码器）定义在 Head 配置内部而非独立的 Neck 配置中：

```yaml
Architecture:
  model_type: rec
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  # 注意：Neck 字段为空！
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr         # SVTR Neck 定义在这里
            dims: 120
            depth: 2
            ...
          Head:
            fc_decay: 0.00001
      - NRTRHead: ...           # 蒸馏辅助头，推理时可丢弃
```

这要求 `MultiHead` 内部自行处理特征序列化和 SVTR 编码。

---

## 六、版本间迁移速查表

### PP-OCRv4 → PP-OCRv5 的变化

| 模块 | v4 | v5 | 变化说明 |
|------|-----|-----|----------|
| 检测 Backbone | PPLCNetV3/PPHGNetV2_B4 | 同 v4 | 无变化 |
| 检测 Neck | RSEFPN/LKPAN | 同 v4，LKPAN 新增 `intracl: true` | Backbone 添加 IntraCL 模块 |
| 检测 Head | DBHead/PFHeadLocal | 同 v4，DBHead 新增 `fix_nan: True` | 数值稳定性改进 |
| 识别 Backbone | PPLCNetV3/PPHGNetV2_B4 | 同 v4 | 无变化 |
| 识别 Head | MultiHead(CTC+NRTR) | 同 v4 | 字典扩展 |
| 字典大小 | ppocr_keys_v1.txt | ppocrv5_dict.txt | v5 支持多种文字 |

### PP-OCRv3 → PP-OCRv4 的变化

| 模块 | v3 | v4 | 变化说明 |
|------|-----|-----|----------|
| 检测 Backbone | MobileNetV3 | PPLCNetV3/PPHGNetV2_B4 | 新骨干网络 |
| 检测 Neck | RSEFPN | RSEFPN/LKPAN | server 版换用 LKPAN |
| 检测 Head | DBHead | DBHead/PFHeadLocal | server 版换用 PFHeadLocal |
| 识别 Backbone | MobileNetV1Enhance | PPLCNetV3/PPHGNetV2_B4 | 新骨干网络 |
| 识别 Neck | SequenceEncoder(svtr) | 内置在 MultiHead | 架构重组 |
| 识别 Head | CTCHead | MultiHead(CTC+NRTR) | 多头设计 |

### PP-OCRv5 → PP-OCRv6 的变化

| 模块 | v5 | v6 | 变化说明 |
|------|-----|-----|----------|
| 检测 Backbone | PPLCNetV3/PPHGNetV2_B4 | PPLCNetV4 | 新一代轻量骨干网络 |
| 检测 Neck | RSEFPN/LKPAN | RSEFPN/RepLKFPN | 引入大核卷积 FPN |
| 检测 Head | DBHead/PFHeadLocal | DBHead | 统一使用 DBHead |
| 识别 Backbone | PPLCNetV3/PPHGNetV2_B4 | PPLCNetV4 | 新一代轻量骨干网络 |
| 识别 Neck(内置) | SVTR | LightSVTR | SVTR 的轻量化变体 |
| 识别 Head | MultiHead(CTC+NRTR) | MultiHead(CTC+NRTR) | 架构不变 |
| 模型规格 | mobile/server | tiny/small/medium | 三档模型规格 |

### PP-OCRv4 → PP-OCRv5 的变化

| 模块 | v4 | v5 | 变化说明 |
|------|-----|-----|----------|
| 检测 Backbone | PPLCNetV3/PPHGNetV2_B4 | 同 v4 | 无变化 |
| 检测 Neck | RSEFPN/LKPAN | 同 v4，LKPAN 新增 `intracl: true` | Neck 添加 IntraCL 模块 |
| 检测 Head | DBHead/PFHeadLocal | 同 v4，DBHead 新增 `fix_nan: True` | 数值稳定性改进 |
| 识别 Backbone | PPLCNetV3/PPHGNetV2_B4 | 同 v4 | 无变化 |
| 识别 Head | MultiHead(CTC+NRTR) | 同 v4 | 字典扩展 |
| 字典大小 | ppocr_keys_v1.txt | ppocrv5_dict.txt | v5 支持多语言+手写 |

### PP-OCRv2 → PP-OCRv3 的变化

| 模块 | v2 | v3 | 变化说明 |
|------|-----|-----|----------|
| 检测 Backbone | MobileNetV3 | MobileNetV3 | 同 v2 |
| 检测 Neck | DBFPN | RSEFPN | 引入 RSE 注意力模块 |
| 识别 Backbone | MobileNetV3 | MobileNetV1Enhance | 更换骨干 |
| 识别 Neck | SequenceEncoder(rnn) | SequenceEncoder(svtr) | RNN 换 SVTR Transformer |
| 识别 Head | CTCHead | CTCHead | 同 v2 |

---

## 七、工具脚本参考

### 现有转换脚本清单

| 脚本 | 用途 |
|------|------|
| `ch_ppocr_v2_det_converter.py` | PP-OCRv2 检测模型转换 |
| `ch_ppocr_v2_rec_converter.py` | PP-OCRv2 识别模型转换 |
| `ch_ppocr_mobile_v2.0_det_converter.py` | PP-OCRv2 移动端检测 |
| `ch_ppocr_mobile_v2.0_rec_converter.py` | PP-OCRv2 移动端识别 |
| `ch_ppocr_server_v2.0_det_converter.py` | PP-OCRv2 服务端检测 |
| `ch_ppocr_server_v2.0_rec_converter.py` | PP-OCRv2 服务端识别 |
| `ch_ppocr_v3_det_converter.py` | PP-OCRv3 检测（蒸馏权重） |
| `ch_ppocr_v3_rec_converter.py` | PP-OCRv3 识别（蒸馏权重） |
| `ch_ppocr_v4_det_converter.py` | PP-OCRv4 移动端检测 |
| `ch_ppocr_v4_det_server_converter.py` | PP-OCRv4 服务端检测 |
| `ch_ppocr_v4_rec_converter.py` | PP-OCRv4 移动端识别 |
| `ch_ppocr_v4_rec_server_converter.py` | PP-OCRv4 服务端识别 |
| `ppocr_v5_det_converter.py` | PP-OCRv5 检测 |
| `ppocr_v5_rec_converter.py` | PP-OCRv5 识别 |
| `rec_svtr_converter.py` | SVTR 通用识别 |
| `rec_can_converter.py` | CAN 手写公式识别 |
| `rec_sar_converter.py` | SAR 识别 |
| `rec_nrtr_mtb_converter.py` | NRTR 识别 |
| `rec_vitstr_converter.py` | ViTSTR 识别 |

### 推理脚本

| 脚本 | 用途 |
|------|------|
| `tools/infer/predict_system.py` | 完整 OCR Pipeline（检测+分类+识别） |
| `tools/infer/predict_det.py` | 单独文字检测 |
| `tools/infer/predict_rec.py` | 单独文字识别 |
| `tools/infer/predict_cls.py` | 单独方向分类 |
| `ptstructure/predict_system.py` | PP-Structure 文档结构化 |

---

## 八、快速移植检查清单

复制此清单，逐项勾选完成：

- [ ] 1. 从 PaddleOCR 获取 YAML 配置和 `.pdparams` 权重
- [ ] 2. 提取 `Architecture` 部分，确定 Backbone/Neck/Head 的 name 和参数
- [ ] 3. 检查 `backbones/__init__.py` — Backbone 是否已注册？
- [ ] 4. 检查 `necks/__init__.py` — Neck 是否已注册？
- [ ] 5. 检查 `heads/__init__.py` — Head 是否已注册？
- [ ] 6. 检查 `transforms/__init__.py` — Transform 是否已注册？
- [ ] 7. 若有缺失组件，编写 PyTorch 实现并注册
- [ ] 8. 在 `configs/` 下创建/复制 YAML 配置文件
- [ ] 9. 在 `converter/` 下创建权重转换脚本
- [ ] 10. 调试：打印 Paddle 和 PyTorch 的 state_dict 键名
- [ ] 11. 实现权重名映射逻辑
- [ ] 12. 处理蒸馏权重（如有）
- [ ] 13. 处理全连接层转置（如有）
- [ ] 14. 用随机输入验证输出一致性
- [ ] 15. 用真实图像验证推理结果
- [ ] 16. 在 `tools/infer/pytorchocr_utility.py` 中注册自动配置
- [ ] 17. 更新文档和模型列表
