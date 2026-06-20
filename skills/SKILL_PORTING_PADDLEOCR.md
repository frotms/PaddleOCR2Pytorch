# PaddleOCR → PytorchOCR 模型移植 SKILL

> 通用技能：将 PaddleOCR 官方模型移植到 PytorchOCR 仓库。

---

## 触发条件

当用户要求移植某个 PaddleOCR 模型到本仓库时使用此技能。

---

## 执行流程

### Phase 1: 信息收集

#### 1.1 确定目标模型
从 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 确定要移植的模型版本和规格。

#### 1.2 获取配置文件
从 PaddleOCR `configs/` 目录找到训练配置文件（`.yml`），提取 `Architecture` 部分：
- `model_type`: det / rec / cls
- `algorithm`: DB / SVTR_LCNet / ...
- `Backbone.name` 和参数
- `Neck.name` 和参数
- `Head.name` 和参数
- `Transform`（通常为 null）

#### 1.3 获取权重文件
从 PaddleOCR 文档/Model Zoo 获取 `.pdparams` 训练权重 URL（非 inference 格式）。

### Phase 2: 组件检查

#### 2.1 对照清单
检查以下文件，确认所需组件是否已实现：

| 检查项 | 文件路径 | 查找内容 |
|--------|---------|---------|
| Backbone | `pytorchocr/modeling/backbones/__init__.py` | 确认 `name` 在 `support_dict` 中 |
| Neck | `pytorchocr/modeling/necks/__init__.py` | 确认 `name` 在 `support_dict` 中 |
| Head | `pytorchocr/modeling/heads/__init__.py` | 确认 `name` 在 `support_dict` 中 |
| Transform | `pytorchocr/modeling/transforms/__init__.py` | 确认（通常为 null） |
| 字典文件 | `pytorchocr/utils/dict/` | 确认字符集字典存在 |

#### 2.2 缺失组件处理
如果缺少组件，需要从 PaddleOCR 源码移植：

1. 在 PaddleOCR 找到对应源码 (`ppocr/modeling/backbones/`, `necks/`, `heads/`)
2. 创建 PyTorch 实现（见 **PaddlePaddle → PyTorch API 对照表**）
3. 注册到对应 `__init__.py`

### Phase 3: 创建配置和转换脚本

#### 3.1 创建 YAML 配置
在 `configs/<det|rec|cls>/<版本名>/` 下创建，参考同版本 PaddleOCR 配置。

#### 3.2 创建转换脚本
在 `converter/` 下创建，参考现有脚本模式：
- 继承 `BaseOCRV20`
- 实现权重加载和映射

### Phase 4: 权重转换

#### 4.1 运行转换
```bash
python converter/<converter_script>.py \
    --yaml_path=<config_path> \
    --src_model_path=<paddle_weights_path>
```

#### 4.2 常见问题处理
- **辅助头过滤**：`aux_binarize_*`, `aux_thresh_*`, `head.gtc_head.*` 等训练专用参数需跳过
- **蒸馏权重**：去除 `Teacher.` 前缀、`head.sar_head.` 等分支
- **线性层转置**：部分 FC 权重可能需要转置（逐层对比形状判断）
- **numpy 兼容性**：Paddle 3.0 + numpy 1.x 可能需要 monkey-patch

### Phase 5: 验证

#### 5.1 数值一致性
用固定随机种子（`np.random.seed(666)`）生成输入，对比 Paddle 和 PyTorch 输出（sum, mean, max, min）。

#### 5.2 真实图片测试
用真实图片验证端到端推理结果，保留测试图片和结果。

### Phase 6: 文档更新

移植完成后，必须更新以下文档：

#### 6.1 文档更新清单

| 文件 | 更新内容 |
|------|---------|
| `doc/doc_ch/models_list.md` | 添加新模型条目（含训练模型下载链接、转换脚本） |
| `doc/doc_en/models_list_en.md` | 英文版同步 |
| `inference.md` | 添加模型转换命令 + 推理命令 + Python API |
| `inference_en.md` | 英文版同步 |
| `README.md` | "近期更新" 顶部添加新模型条目 |
| `README_en.md` | 英文版同步 |
| `skills/<模型名>_porting_guide.md` | 独立移植指南（重大版本推荐） |

#### 6.2 ptstructure 模块（PP-StructureV3）

如果移植的是 PP-StructureV3 相关模型，更新以下文件：

| 文件 | 更新内容 |
|------|---------|
| `ptstructure/predict_structure.py` | 添加 CLI 参数和模型加载逻辑 |
| `ptstructure/<模块名>/` | 模型代码（自包含，参考 formula/、seal/、doc_preprocess/） |
| `ptstructure/README.md` | 更新功能列表 |
| `converter/ppstructure_<模型名>_converter.py` | 权重转换脚本 |
| `skills/ppstructurev3_porting_guide.md` | 更新移植指南

---

## PaddlePaddle → PyTorch API 对照表

### 基础 API

| PaddlePaddle | PyTorch | 说明 |
|-------------|---------|------|
| `paddle.nn.Conv2D` | `nn.Conv2d` | 参数名 `in_channels`/`out_channels` 相同 |
| `paddle.nn.BatchNorm2D` | `nn.BatchNorm2d` | momentum: Paddle=0.9 → PyTorch=0.1 (1-momentum) |
| `paddle.nn.Linear` | `nn.Linear` | 权重形状相同 `(out, in)` |
| `paddle.nn.LayerNorm` | `nn.LayerNorm` | 接口基本一致 |
| `paddle.nn.GELU` | `nn.GELU` | 默认 approximate='none' |
| `paddle.nn.Hardswish` | `nn.Hardswish` | 行为一致 |
| `paddle.nn.ReLU` | `nn.ReLU` | 行为一致 |
| `paddle.nn.Swish` | `nn.SiLU` | Swish == SiLU |
| `paddle.nn.MaxPool2D` | `nn.MaxPool2d` | 参数名略有差异 |

### 容器和模块

| PaddlePaddle | PyTorch |
|-------------|---------|
| `paddle.nn.Layer` | `nn.Module` |
| `paddle.nn.Sequential` | `nn.Sequential` |
| `paddle.nn.LayerList` | `nn.ModuleList` |
| `self.add_sublayer(name, layer)` | `self.add_module(name, layer)` |

### Tensor 操作

| PaddlePaddle | PyTorch |
|-------------|---------|
| `paddle.concat(x, axis=1)` | `torch.cat(x, dim=1)` |
| `paddle.multiply(x=a, y=b)` | `a * b` 或 `torch.mul(a, b)` |
| `x.mean(axis=[2,3], keepdim=True)` | `x.mean(dim=[2,3], keepdim=True)` |
| `x.flatten(2)` | `x.flatten(2)`（相同） |
| `x.transpose([0, 2, 1])` | `x.permute(0, 2, 1)` |
| `x.reshape([0, H, W, C])` | `x.reshape(-1, H, W, C)` |

### 权重初始化

| PaddlePaddle | PyTorch |
|-------------|---------|
| `Constant(val)(tensor)` | `nn.init.constant_(tensor, val)` |
| `paddle.ParamAttr(initializer=...)` | 用 `nn.init.*` 在 `__init__` 后调用 |
| `param.set_value(val)` | `param.data.copy_(val)` |

### 函数式 API

| PaddlePaddle | PyTorch |
|-------------|---------|
| `F.adaptive_avg_pool2d` | `F.adaptive_avg_pool2d`（相同） |
| `F.avg_pool2d` | `F.avg_pool2d`（相同） |
| `F.upsample(scale_factor=2, mode="nearest")` | `F.interpolate(scale_factor=2, mode="nearest")` |
| `F.pad` | `F.pad`（相同） |
| `paddle.no_grad()` | `torch.no_grad()` |
| `x.stop_gradient = True` | `x.detach()` 或 `x.clone().detach()` |

### 权重键名映射

| Paddle 键名模式 | PyTorch 键名 | 操作 |
|---------------|-------------|------|
| `._mean` | `.running_mean` | 重命名 |
| `._variance` | `.running_var` | 重命名 |
| `fc.weight` (某些情况) | `fc.weight.T` | 可能需要转置 |
| `Student.xxx` | `xxx` | 去除前缀 |
| `Teacher.xxx` | (丢弃) | 蒸馏教师分支 |

---

## 快速检查清单

复制此清单，逐项勾选：

```
[ ] 1. 确定目标模型版本和规格
[ ] 2. 从 PaddleOCR 获取 YAML 配置
[ ] 3. 提取 Architecture 部分
[ ] 4. 检查 Backbone 是否已注册
[ ] 5. 检查 Neck 是否已注册
[ ] 6. 检查 Head 是否已注册
[ ] 7. 检查 Transform 是否已注册
[ ] 8. 如有缺失组件 → 编写 PyTorch 实现 → 注册
[ ] 9. 创建/更新 YAML 配置文件
[ ] 10. 下载 .pdparams 权重
[ ] 11. 创建转换脚本
[ ] 12. 调试：打印两边 state_dict 键名对比
[ ] 13. 实现权重名映射逻辑
[ ] 14. 处理蒸馏/辅助权重过滤
[ ] 15. 处理全连接层转置（逐层对比形状）
[ ] 16. 用随机输入验证数值一致性
[ ] 17. 用真实图像验证推理结果
[ ] 18. 保存测试结果
[ ] 19. 更新 doc/doc_ch/inference.md（模型转换 + 推理命令）
[ ] 20. 更新 doc/doc_en/inference_en.md（英文版同步）
[ ] 21. 更新 README.md（近期更新 + TODO）
[ ] 22. 更新 README_en.md（英文版同步）
[ ] 23. 创建/更新 docs/<版本>_porting_guide.md（移植完整指南）
```

---

## 典型调试流程

### 权重加载 KeyError
```
→ 打印 Paddle state_dict 所有键名
→ 打印 PyTorch state_dict 所有键名
→ 逐层对比，找出差异
→ 添加过滤/重命名/转置逻辑
```

### 推理结果不一致
```
→ 逐层输出中间特征对比
→ 检查激活函数差异（hardswish 实现）
→ 检查 BN momentum 累积
→ 检查 interpolate align_corners
→ 定位到第一个出现差异的层
```

### 形状不匹配
```
→ 检查 Paddle Pooling padding="SAME" 的语义差异
→ 检查 MaxPool vs AvgPool 的参数差异
→ 检查 stride > 1 时的 padding 计算
```

---

## 参考资源

- [PaddleOCR 源码](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddlePaddle API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)
- [PyTorch 文档](https://pytorch.org/docs/stable/)
- 本仓库 WORKFLOW_PORTING_PADDLEOCR_MODELS.md（移植工作流详细说明）
- 本仓库 docs/PP-OCRv6_porting_guide.md（PP-OCRv6 移植案例）
