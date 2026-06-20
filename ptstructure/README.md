# PP-StructureV3 (PyTorch)

PaddleOCR PP-StructureV3 文档结构化解析系统的 PyTorch 实现。

## 概述

PP-StructureV3 是 PaddleOCR 的文档结构化解析系统，支持以下功能：

- **布局检测 (Layout Detection)**: 检测文档中的标题、段落、表格、图片、公式、印章等 23 类区域
- **OCR 文字识别**: 对文本区域进行文字检测和识别
- **表格结构识别 (Table Recognition)**: 识别有线/无线表格结构，输出 HTML
- **阅读顺序恢复**: 使用 XY-Cut 算法恢复文档的阅读顺序
- **多格式输出**: 支持 Markdown、JSON、HTML 格式输出

## 架构

```
输入图像
  ├── 布局检测 (PP-DocLayout / PicoDet)
  ├── OCR 文字识别 (PP-OCRv5/v6)
  ├── 表格结构识别 (SLANeXt)
  ├── 阅读顺序恢复 (XY-Cut)
  └── 输出: Markdown / JSON / HTML / 可视化
```

## 目录结构

```
ptstructure/
├── __init__.py              # 模块入口
├── pipeline.py              # PPStructureV3 主管道
├── predict_structure.py     # CLI 推理脚本
├── README.md                # 本文件
├── layout/                  # 布局检测模块
│   ├── __init__.py
│   ├── picodet.py           # PicoDet 布局检测模型
│   └── postprocess.py       # 布局后处理 (NMS, 标签映射)
├── table/                   # 表格识别模块
│   ├── __init__.py
│   ├── slanext.py           # SLANeXt 表格结构识别模型
│   └── table_utils.py       # HTML 生成工具
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── reading_order.py     # 阅读顺序恢复 (XY-Cut)
│   ├── markdown.py          # Markdown/JSON 生成
│   └── visualize.py         # 结果可视化
└── data/                    # 测试数据
```

## 模型

| 模型 | 架构 | 参数量 | 功能 |
|------|------|--------|------|
| PP-DocLayout-S | PicoDet-LCNet | ~0.6M | 布局检测 (23类) |
| SLANeXt_wired | ViT + GRU Decoder | ~90M | 有线表格结构识别 |
| SLANeXt_wireless | ViT + GRU Decoder | ~90M | 无线表格结构识别 |

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision opencv-python pillow numpy pyyaml
```

### 2. 下载预训练模型

从 [PytorchOCR 模型下载](https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg) 获取转换后的模型权重。

### 3. Python API 使用

```python
from ptstructure import PPStructureV3

# 初始化管道
pipeline = PPStructureV3(
    layout_model_path='models/ppdoclayout.pth',
    det_model_path='models/ppocrv5_det.pth',
    rec_model_path='models/ppocrv5_rec.pth',
    table_model_path='models/slanext.pth',
    device='cpu',
)

# 解析文档
results = pipeline.predict('document.jpg')

for result in results:
    print(result['markdown'])
    # 保存结果
    pipeline.save_to_markdown(results, 'output.md')
    pipeline.save_to_json(results, 'output.json')
    pipeline.save_to_img(results, 'output/')
```

### 4. 命令行使用

```bash
python ptstructure/predict_structure.py \
    --image_dir=./data/ \
    --layout_model_path=./models/ppdoclayout.pth \
    --det_model_path=./models/ppocrv5_det.pth \
    --rec_model_path=./models/ppocrv5_rec.pth \
    --table_model_path=./models/slanext.pth \
    --output_dir=./output/
```

### 5. 仅使用布局检测

```python
from ptstructure.utils.visualize import draw_layout_boxes
import cv2

pipeline = PPStructureV3(
    layout_model_path='models/ppdoclayout.pth',
    # 不加载 OCR 和表格模型
)

img = cv2.imread('document.jpg')
# 获取布局检测结果
results = pipeline.predict('document.jpg')
layout_boxes = results[0]['layout_boxes']

# 绘制布局框
vis = draw_layout_boxes(img, layout_boxes)
cv2.imwrite('layout_result.jpg', vis)
```

## 模型转换

从 PaddlePaddle 权重转换为 PyTorch：

### 布局检测模型

```bash
python converter/ppstructure_layout_converter.py \
    --yaml_path=configs/layout/PP-DocLayout-S.yml \
    --src_model_path=./models/PP-DocLayout-S_pretrained.pdparams \
    --dst_model_path=./models/ptocr_ppdoclayout_s.pth
```

### 表格结构识别模型

```bash
python converter/ppstructure_slanext_converter.py \
    --yaml_path=configs/tablev3/SLANeXt_wired.yml \
    --src_model_path=./models/SLANeXt_wired_pretrained.pdparams \
    --dst_model_path=./models/ptocr_slanext_wired.pth
```

## 输出格式

### Markdown 输出示例

```markdown
# 文档标题

## 章节标题

这是正文内容，包含重要的信息。

<table>
<tr><td>列A</td><td>列B</td></tr>
<tr><td>1</td><td>2</td></tr>
</table>

### 图片标题
*[图片: 图表说明]*
```

### JSON 输出格式

```json
{
  "input_path": "document.jpg",
  "blocks": [
    {
      "block_id": 0,
      "block_label": "doc_title",
      "block_content": "文档标题",
      "block_bbox": [50, 20, 700, 80],
      "block_order": 0,
      "confidence": 0.95
    }
  ]
}
```

## 参考

- [PaddleOCR PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR)
- [PytorchOCR 仓库](https://github.com/frotms/PaddleOCR2Pytorch)
