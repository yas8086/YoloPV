# YoloPV C++ ONNX 推理示例

基于 OpenCV DNN 的 ONNX 模型推理，适配 YoloPV（单类 `Stain`）。

## 前置条件

1. 先导出 ONNX 模型（在项目根目录）：
   ```bash
   python YoloPV.py export --weights runs/detect/train/weights/best.pt --format onnx
   ```
2. 安装 OpenCV（带 DNN 模块）。CUDA 可选，用于 GPU 加速。

## 编译

```bash
cd examples/YOLOv8-CPP-Inference
mkdir build && cd build
cmake ..
make
```

## 运行

```bash
# 默认推理 ../../yolov8/datasets/images/test
./yolopv_infer

# 自定义模型与图像
./yolopv_infer --model path/to/best.onnx --source path/to/image.jpg

# 强制 CPU（无 CUDA 时）
./yolopv_infer --cpu

# 指定输入尺寸
./yolopv_infer --imgsz 1280
```

结果保存到 `build/output/det_*.jpg`。

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `../../runs/detect/train/weights/best.onnx` | ONNX 模型路径 |
| `--source` | `../../yolov8/datasets/images/test` | 图像或目录 |
| `--imgsz` | 640 | 输入尺寸（须与导出时一致或可被模型接受） |
| `--cpu` | false | 强制 CPU 推理 |

## 说明

- 类别定义在 [inference.h](inference.h) 中硬编码为 `Stain`（与 `YoloPV_dataset.yaml` 一致）
- 使用 letterbox 保持长宽比，与训练预处理对齐
- 支持 YOLOv8（4+nc 输出格式）
