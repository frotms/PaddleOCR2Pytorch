# PP-StructureV3 移植指南

> PaddleOCR PP-StructureV3 → PytorchOCR 完整移植，纯PyTorch推理，零Paddle依赖。

## 一、架构概览

```
输入图片 (BGR)
    │
    ▼
布局检测: PPDocLayout (PicoDet)
    ├── Backbone: LCNet (scale=0.75/2.0) → 3个特征层
    ├── Neck: LCPAN → 4个FPN层 (s8,s16,s32,s64)
    ├── Head: PicoHeadV2 (PicoFeat + PicoSE + cls_align)
    └── 输出: 23类布局区域 [label, score, bbox]
    │
    ▼
逐区域处理:
    ├── 文字区域 → OCR (PP-OCRv6: DB det + SVTR rec)
    ├── 表格区域 → 表格识别 (SLANeXt: ViT + GRU Decoder)
    └── 图片/印章 → 占位符 [image]/[seal]
    │
    ▼
阅读顺序恢复: XY-Cut 算法
    │
    ▼
格式化输出: Markdown / JSON / 可视化
```

## 二、模型详情

### 2.1 布局检测模型 (PPDocLayout)

| 属性 | S 变体 | M 变体 | L 变体 |
|------|--------|--------|--------|
| 参数量 | 1.2M | 5.8M | 20.1M |
| backbone 通道 | [16,24,48,96,192,384] | [32,64,128,256,512,1024] | [64,128,256,512,1024,2048] |
| neck/head 通道 | 96 | 160 | 256 |
| PicoFeat conv对 | 2 | 4 | 4 |
| LCNet scale | 0.75 | 2.0 | 4.0 |
| 输入尺寸 | 640×640 | 640×640 | 640×640 |

**架构**: LCNet + LCPAN + PicoHeadV2

关键设计点（与 PaddleDetection 对齐）：
- InvertedResidual **无 skip 连接**（不同于 MobileNetV3）
- blocks2 **stride=1**（不下采样）
- LCPAN **num_features=4**（3个backbone特征 + 1个生成的P7）
- **cell_offset=0.5**，**reg_max=7**
- cls_score = sqrt(sigmoid(cls_logit) × cls_align_prob + ε)

### 2.2 表格结构识别模型 (SLANeXt)

| 属性 | 值 |
|------|-----|
| 参数量 | ~90M |
| Encoder | ViT (12层, 768dim, window+global attention) |
| Decoder | GRU + Attention 自回归 |
| 输入 | 512×512 |
| 输出 | HTML token 序列 (50类) |

### 2.3 OCR 模型 (PP-OCRv6)

| 模型 | 规格 |
|------|------|
| 检测 | DB + PPLCNetV4, small(9.8M) / medium(61M) |
| 识别 | SVTR_LCNet, small(20M) / medium(73M) |

## 三、权重转换

### 布局检测

```bash
# PP-DocLayout-M（推荐，5.8M）
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams
python converter/ppstructure_layout_converter.py \
    --src_model_path=PP-DocLayout-M_pretrained.pdparams \
    --dst_model_path=ptocr_ppdoclayout_m.pth --variant=M

# PP-DocLayout-S（轻量，1.2M）
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams
python converter/ppstructure_layout_converter.py \
    --src_model_path=PP-DocLayout-S_pretrained.pdparams \
    --dst_model_path=ptocr_ppdoclayout_s.pth --variant=S
```

### 表格识别

```bash
python converter/ppstructure_slanext_converter.py \
    --src_model_path=SLANeXt_wired_pretrained.pdparams \
    --dst_model_path=ptocr_slanext_wired.pth
```

## 四、推理命令

### 完整管道

```bash
python ptstructure/predict_structure.py \
    --image_dir=./doc/table/ \
    --output_dir=./output/ \
    --layout_variant=M \
    --layout_score_thresh=0.2 \
    --layout_nms_thresh=0.5 \
    --det_model_path=../models/v6/ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth \
    --det_yaml_path=configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    --rec_model_path=../models/v6/ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth \
    --rec_yaml_path=configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
    --rec_char_dict_path=pytorchocr/utils/dict/ppocrv6_dict.txt \
    --table_model_path=models/structurev3/ptocr_slanext_wired.pth
```

### 仅布局检测

```python
from ptstructure.layout.picodet import PPDocLayout

# S 变体（1.2M）或 M 变体（5.8M），切换只需改 variant 参数
model = PPDocLayout(variant='M')
model.eval()
model.load_state_dict(torch.load('ptocr_ppdoclayout_m.pth', weights_only=True))
detections = model.detect(img, score_thresh=0.2)
```

## 五、输出格式

### JSON
```json
{
  "input_path": "doc.jpg",
  "blocks": [
    {
      "block_id": 0, "block_label": "text",
      "block_bbox": [x1,y1,x2,y2],
      "block_content": "识别的文字内容",
      "confidence": 0.85, "block_order": 0
    }
  ]
}
```

### Markdown
识别结果自动排版为 Markdown 格式，表格输出为 HTML `<table>`。

### 可视化
在每个检测框上绘制标签和文字内容，支持中/日/韩等多语言渲染。

## 六、标签体系（23类）

paragraph_title, image, text, number, abstract, content, figure_title, formula,
table, table_title, reference, doc_title, footnote, header, algorithm, footer,
seal, chart_title, chart, formula_number, header_image, footer_image, aside_text

## 七、文件结构

```
ptstructure/
├── predict_structure.py     # CLI 推理入口（主脚本）
├── layout/
│   ├── picodet.py            # PPDocLayout 模型 (PicoDet)
│   └── postprocess.py        # NMS 后处理
├── table/
│   ├── slanext.py            # SLANeXt 表格识别模型
│   └── table_utils.py
├── utils/
│   ├── reading_order.py      # XY-Cut 阅读顺序
│   ├── markdown.py           # Markdown/JSON 生成
│   └── visualize.py          # PIL 可视化（CJK字体支持）
converter/
├── ppstructure_layout_converter.py    # 布局模型转换
└── ppstructure_slanext_converter.py   # 表格模型转换
models/structurev3/
├── PP-DocLayout-S_pretrained.pdparams  # Paddle S 权重 (4.6MB)
├── PP-DocLayout-M_pretrained.pdparams  # Paddle M 权重 (22.4MB)
├── ptocr_ppdoclayout_s.pth         # PyTorch S 权重 (1.2M params)
├── ptocr_ppdoclayout_m.pth         # PyTorch M 权重 (5.8M params，推荐)
└── ptocr_slanext_wired.pth         # PyTorch 表格权重
```

## 八、已知局限

- Table 结构识别（SLANeXt）在部分图片上输出 rowspan 占位符，待优化
- PP-DocLayout-S table 检测弱于 M，3/10 表图漏检；推荐文档含表格时用 M
- PP-DocLayout-L 权重未下载测试
- 不支持公式识别、印章识别（PaddleOCR 有，但未移植）

## 九、参考

- [PaddleOCR PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleDetection PicoDet](https://github.com/PaddlePaddle/PaddleDetection)
- [PytorchOCR 仓库](https://github.com/frotms/PaddleOCR2Pytorch)
