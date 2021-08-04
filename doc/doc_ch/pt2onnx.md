# PT2ONNX

`.pth` 转 `.onnx`

## 步骤

1. build PyTorch model
2. torch.onnx.export `.onnx` model: 导出静态输入/输出模型，并去除未使用节点
3. 模型是否需要optimize
4. 获得最终`onnx`模型

## References

- [lstm.py](./misc/lstm.py)
- [onnx_inference.py](./misc/onnx_inference.py)
- [onnx_optimizer.py](./misc/onnx_optimizer.py)
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
- [onnxoptimizer](https://github.com/onnx/optimizer)
- [torch.onnx.export](https://pytorch.org/docs/stable/onnx.html?highlight=torch%20onnx%20export#torch.onnx.export)
- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)
- [知乎: onnx simplifier 和 optimizer](https://zhuanlan.zhihu.com/p/350702340)