# PP-StructureV3 移植指南

> PaddleOCR PP-StructureV3 → PytorchOCR 完整移植，纯PyTorch推理，零Paddle依赖。

## 一、架构概览

```
输入图片 (BGR)
    │
    ├── [可选] ① 文档方向校正 (doc_ori)          ← PP-LCNet, 4方向分类
    ├── [可选] ② 文档去扭曲 (UVDoc)              ← CGU-Net, 488×712
    │
    ├── ③ 版面检测 (PPDocLayout)                ← PicoDet-LCNet, 23类
    │
    ├── ④ 全局文本检测 (DB)                      ← 全图一次检测
    ├── [可选] ⑥ 文本行方向校正 (textline_ori)    ← PP-LCNet, 0°/180°
    ├── ⑦ 全局文本识别 (SVTR/CRNN)              ← 全图一次识别
    │
    └── 逐区域处理:
        ├── table   → ⑤ SLANeXt (裁切→HTML)
        ├── formula → ⑧ PP-FormulaNet (裁切→LaTeX)
        ├── seal    → ⑨ DB(seal)+OCR (裁切→文字)
        └── 其他    → 从全局OCR按bbox交集筛选
    │
    ▼
阅读顺序恢复 (XY-Cut) → Markdown / JSON / 可视化
```

**关键设计：** ③ 和 ④ 并行执行；文字区域不再逐区域裁切OCR，改为全局OCR一次完成然后用 bbox 交集筛选——与 PaddleX PP-StructureV3 完全一致。

## 二、模型详情

### 2.1 版面检测 (PPDocLayout)

| 属性 | S | M | L |
|------|---|---|---|
| 参数量 | 1.2M | 5.8M | 20.1M |
| 输入尺寸 | 640×640 | 640×640 | 640×640 |
| 架构 | LCNet+LCPAN+PicoHeadV2 | 同 | 同 |

### 2.2 OCR 模型 (PP-OCRv6)

| 模型 | 架构 | 功能 |
|------|------|------|
| 文本检测 | DB + PPLCNetV4 | 全图文字框检测 |
| 文本识别 | SVTR_LCNet | 文字框→文字串 |
| 文本行方向 | PP-LCNet (2类) | 0°/180° 纠正 (可选) |

### 2.3 表格识别 (SLANeXt)

| 属性 | 值 |
|------|-----|
| 参数量 | ~90M |
| Encoder | ViT (12层, 768dim) |
| Decoder | GRU + Attention 自回归 |
| 输入 | 512×512 |
| 输出 | HTML token 序列 |

### 2.4 公式识别 (PP-FormulaNet)

| 属性 | S | M (plus-M) |
|------|---|-----------|
| Backbone | PPHGNetV2_B4 | PPHGNetV2_B6 |
| Decoder层数 | 2 | 6 |
| Decoder维度 | 384 | 512 |
| 输入尺寸 | 384×384 | 384×384 |
| Tokenizer | UniMERNet BPE (50000) | 同 |

### 2.5 印章检测 (Seal Detection)

| 属性 | 值 |
|------|-----|
| 架构 | DB (PPLCNetV3+RSEFPN+DBHead) |
| 输入 | 自动缩放至32的倍数 |
| 检测后 | 逐框OCR识别 |

### 2.6 文档预处理

| 模型 | 功能 | 架构 | 大小 |
|------|------|------|------|
| doc_ori | 文档方向分类 | PP-LCNet (4类) | 6.5MB |
| UVDoc | 文档去扭曲 | CGU-Net | 31MB |
| textline_ori | 文本行方向分类 | PP-LCNet (2类) | 6.5MB |

## 三、文件结构

```
ptstructure/
├── predict_structure.py        # CLI 主脚本（含所有 CLI 参数）
├── doc_preprocess/             # 文档预处理
│   ├── doc_orientation.py      #   doc_ori (方向分类)
│   └── unwarp.py               #   UVDoc (去扭曲)
├── layout/                     # 版面检测
│   ├── picodet.py              #   PPDocLayout 模型
│   └── postprocess.py          #   NMS 后处理
├── table/                      # 表格识别
│   ├── slanext.py              #   SLANeXt 模型
│   └── table_utils.py
├── formula/                    # 公式识别
│   ├── unimernet_head.py       #   MBart 解码器 (PyTorch port)
│   ├── ppformulanet_head.py    #   PPFormulaNet Head
│   ├── ppformulanet.py         #   FormulaRecognizer (入口)
│   ├── tokenizer.py            #   BPE tokenizer (零依赖)
│   └── postprocess.py          #   LaTeX 后处理
├── seal/                       # 印章检测
│   └── seal_det.py             #   DB 印章检测器
└── utils/                      # 后处理工具
    ├── reading_order.py        #   XY-Cut
    ├── markdown.py             #   Markdown/JSON 生成
    └── visualize.py            #   可视化

converter/
├── ppstructure_layout_converter.py    # 版面模型转换
├── ppstructure_slanext_converter.py   # 表格模型转换
├── ppstructure_formula_converter.py   # 公式模型转换
└── ppstructure_seal_converter.py      # 印章模型转换

models/structurev3/                    # 模型权重 (PyTorch .pth)
├── ptocr_ppdoclayout_m.pth           # 版面检测 (M)
├── ptocr_ppdoclayout_s.pth           # 版面检测 (S)
├── ptocr_slanext_wired.pth           # 表格识别
├── ptocr_formulanet_m.pth            # 公式识别 (M)
├── ptocr_seal_det.pth                # 印章检测
├── ptocr_doc_ori.pth                 # 文档方向分类
├── ptocr_uvdoc.pth                   # 文档去扭曲
└── ptocr_textline_ori.pth            # 文本行方向分类
```

## 四、推理

### 全功能（所有模块启用）

```bash
python ptstructure/predict_structure.py \
    --image_dir=./doc/imgs/ \
    --output_dir=./output/ \
    --layout_variant=M \
    --use_doc_orientation --use_doc_unwarping \
    --use_angle_cls \
    --use_formula --formula_variant=M \
    --use_seal
```

### 基础功能（仅版面+OCR+表格）

```bash
python ptstructure/predict_structure.py \
    --image_dir=./doc/imgs/ \
    --output_dir=./output/ \
    --layout_variant=M
```

### 仅版面检测

```bash
python ptstructure/predict_structure.py \
    --image_dir=./doc/imgs/ \
    --output_dir=./output/ \
    --layout_variant=M \
    --det_model_path=none --rec_model_path=none --table_model_path=none
```

### Python API

```python
# 公式识别
from ptstructure.formula.ppformulanet import FormulaRecognizer
rec = FormulaRecognizer(variant='M')
rec.load_weights('models/structurev3/ptocr_formulanet_m.pth')
latex = rec.recognize(formula_crop)

# 印章检测
from ptstructure.seal.seal_det import SealDetector
det = SealDetector()
det.load_weights('models/structurev3/ptocr_seal_det.pth')
boxes, scores = det.detect(seal_crop)

# 文档方向分类
from ptstructure.doc_preprocess import DocOrientationClassifier
ori = DocOrientationClassifier()
ori.load_weights('models/structurev3/ptocr_doc_ori.pth')
label, score = ori.classify(img)

# 文档去扭曲
from ptstructure.doc_preprocess import UVDocUnwarper
uw = UVDocUnwarper()
uw.load_weights('models/structurev3/ptocr_uvdoc.pth')
corrected = uw.unwarp(img)
```

## 五、输出格式

### Markdown
- 文字区域: 纯文本
- 表格: `<table>...</table>` HTML
- 公式 (启用后): `$$...$$` LaTeX
- 印章 (启用后): `[seal: 识别文字]`
- 图片: `[image]`

### JSON
```json
{
  "input_path": "doc.jpg",
  "blocks": [
    {"block_id": 0, "block_label": "text", "block_bbox": [x1,y1,x2,y2],
     "block_content": "识别的文字", "confidence": 0.85, "block_order": 0}
  ]
}
```

## 六、权重转换

```bash
# 版面检测
python converter/ppstructure_layout_converter.py \
    --src_model_path=PP-DocLayout-M_pretrained.pdparams \
    --dst_model_path=ptocr_ppdoclayout_m.pth --variant=M

# 表格识别
python converter/ppstructure_slanext_converter.py \
    --src_model_path=SLANeXt_wired_pretrained.pdparams \
    --dst_model_path=ptocr_slanext_wired.pth

# 公式识别
python converter/ppstructure_formula_converter.py \
    --src_model_path=PP-FormulaNet_plus-M_pretrained.pdparams \
    --dst_model_path=ptocr_formulanet_m.pth --variant=M

# 印章检测
python converter/ppstructure_seal_converter.py \
    --src_model_path=PP-OCRv4_mobile_seal_det_pretrained.pdparams \
    --dst_model_path=ptocr_seal_det.pth

# 文档方向 / 文本行方向 (使用通用 CLS converter)
python converter/pplcnet_cls_converter.py \
    --yaml_path=configs/cls/doc_ori/PP-LCNet_x1_0_doc_ori.yml \
    --src_model_path=PP-LCNet_x1_0_doc_ori_pretrained.pdparams \
    --output_path=ptocr_doc_ori.pth

# 文档去扭曲
python converter/uvdoc_converter.py \
    --src_model_path=UVDoc_pretrained.pdparams \
    --output_path=ptocr_uvdoc.pth
```

## 七、标签体系（23类）

paragraph_title, image, text, number, abstract, content, figure_title, formula,
table, table_title, reference, doc_title, footnote, header, algorithm, footer,
seal, chart_title, chart, formula_number, header_image, footer_image, aside_text

## 八、已知局限

- Table 识别 (SLANeXt) 在部分图片上输出 rowspan 占位符，HTML 格式待优化
- 公式识别 (PP-FormulaNet) 模型较大 (M: 754MB)，CPU 推理慢，建议 GPU
- 印章检测器 (DB seal) 因 OpenCV 4.10 轮廓兼容性未能检出，当前 fallback 到全局 OCR
- doc_ori、UVDoc、textline_ori 预处理模型需单独下载 Paddle 权重并转换

## 九、参考

- [PaddleOCR PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleDetection PicoDet](https://github.com/PaddlePaddle/PaddleDetection)
- [PytorchOCR](https://github.com/frotms/PaddleOCR2Pytorch)
