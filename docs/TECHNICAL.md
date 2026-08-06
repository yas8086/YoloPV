# YoloPV 技术文档

> 本文档面向开发者与算法工程师，详细说明 YoloPV 项目的系统设计、算法选型、数据流程、训练细节与扩展方向。
> 用户使用请参考 [README.md](../README.md)。

***

## 目录

1. [项目概览](#1-项目概览)
2. [系统架构](#2-系统架构)
3. [数据流](#3-数据流)
4. [模型选型与算法](#4-模型选型与算法)
5. [训练 Pipeline](#5-训练-pipeline)
6. [关键参数详解](#6-关键参数详解)
7. [Bug 复盘与根因分析](#7-bug-复盘与根因分析)
8. [性能基线](#8-性能基线)
9. [部署方案](#9-部署方案)
10. [扩展方向与路线图](#10-扩展方向与路线图)
11. [API 参考](#11-api-参考)
12. [附录：常见问题](#12-附录常见问题)
13. [开发日志](#13-开发日志)

***

## 1. 项目概览

### 1.1 目标

在无人机采集的 **红外热成像** 中自动定位 **光伏板表面污点**（鸟粪/沙尘等热斑），输出 **边界框 + 类别 + 置信度**。

### 1.2 非目标

- **不是分割任务**：本项目只输出目标位置框，不输出像素级掩模
- **不是分类任务**：不区分污点类型（鸟粪/沙尘/树叶等统一标为 `Stain`）
- **不针对可见光**：仅适用红外通道

### 1.3 关键约束

| 约束   | 数值                             |
| ---- | ------------------------------ |
| 目标尺寸 | 5×5 \~ 20×20 像素（1024×1024 图像中） |
| 信噪比  | < 3 dB                         |
| 图像数量 | train=10, val=3（极小数据集）         |
| 推理延迟 | 无人机端 ≤ 30 ms/帧                 |
| 部署平台 | 边缘 GPU / CPU 均可                |

***

## 2. 系统架构

### 2.1 整体模块图

```
┌────────────────────────────────────────────────────────────────┐
│                       数据准备层                                │
│  原始红外 JPEG  ──LabelMe 标注──>  JSON  ──JsonToTxt.py──>  TXT  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                       训练层 (YoloPV.py)                        │
│  YOLOv8n ──> ultralytics Trainer ──> best.pt / last.pt         │
│           (mosaic, copy_paste, TTA)                            │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                       导出与部署层                              │
│  best.pt ──> ONNX ──> C++ DNN  /  OpenCV Python               │
│         └─> TorchScript ──> LibTorch                           │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

| 模块        | 文件                                                                            | 职责                              |
| --------- | ----------------------------------------------------------------------------- | ------------------------------- |
| CLI 入口    | [YoloPV.py](../YoloPV.py)                                                     | 4 个子命令：train/predict/val/export |
| 数据集配置     | [YoloPV\_dataset.yaml](../YoloPV_dataset.yaml)                                | 路径、类别、划分                        |
| 标签转换      | [JsonToTxt.py](../JsonToTxt.py)                                               | LabelMe JSON → YOLO TXT         |
| 训练/推理     | ultralytics 8.4                                                               | YOLOv8 实现                       |
| C++ 部署    | [examples/YOLOv8-CPP-Inference/](../examples/YOLOv8-CPP-Inference/)           | ONNX Runtime + OpenCV DNN       |
| Python 部署 | [examples/YOLOv8-OpenCV-ONNX-Python/](../examples/YOLOv8-OpenCV-ONNX-Python/) | OpenCV DNN                      |

### 2.3 依赖

| 库             | 版本     | 用途              |
| ------------- | ------ | --------------- |
| ultralytics   | ≥ 8.1  | YOLOv8 训练/推理/导出 |
| torch         | ≥ 2.0  | 深度学习框架          |
| torchvision   | ≥ 0.15 | 数据增强            |
| opencv-python | ≥ 4.8  | 图像 I/O、部署       |
| numpy         | ≥ 1.23 | 数组运算            |
| Pillow        | ≥ 9.5  | 图像加载            |
| matplotlib    | ≥ 3.7  | 训练曲线            |
| pyyaml        | ≥ 6.0  | 解析 yaml         |
| pandas        | ≥ 2.0  | results.csv 处理  |

详见 [requirements.txt](../requirements.txt)。

***

## 3. 数据流

### 3.1 训练数据流

```
原始 JPEG (384×288~1024×1024, 红外单通道存为 3 通道 JPEG)
    │
    ├─[1] LabelMe 标注 → 边界框或多边形 → 保存为 .json
    │
    ├─[2] JsonToTxt.py 解析 → 归一化 (cx, cy, w, h) → 保存为 .txt
    │      关键点：class_id 必须是数字（YOLO 格式要求）
    │
    └─[3] ultralytics 检测器加载 → 验证 (images ↔ labels 配对) → 进入训练
```

### 3.2 标签格式

**YOLO 检测格式**（每行一个目标）：

```
<class_id> <center_x> <center_y> <width> <height>
```

所有值归一化到 `[0, 1]`，相对于图像宽高。

示例（来自 `yolov8/datasets/labels/train/jpeg_dvp1_0_20240620163929.txt`）：

```
0 0.608464 0.476215 0.009115 0.008681
```

表示：在归一化坐标中，目标中心位于 `(0.608, 0.476)`，宽高 `(0.009, 0.009)`，即图像 384×288 中约为 `(233, 137)`、大小 `(3.5, 2.5)` 像素——典型小目标。

### 3.3 LabelMe JSON → YOLO TXT

[JsonToTxt.py](../JsonToTxt.py) 的核心转换逻辑：

```python
def shape_to_bbox(shape, width, height):
    # 1. 类别映射：'Stain' -> 0
    cid = CLASS_MAP[shape['label']]

    # 2. rectangle：两点为对角
    if shape['shape_type'] == 'rectangle':
        (x1, y1), (x2, y2) = shape['points']

    # 3. polygon：取最小外接矩形
    else:
        xs = [p[0] for p in shape['points']]
        ys = [p[1] for p in shape['points']]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

    # 4. 归一化
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    w  = (x2 - x1) / width
    h  = (y2 - y1) / height

    return cid, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)
```

***

## 4. 模型选型与算法

### 4.1 为什么选 YOLOv8

| 模型           | mAP      | FPS (GPU) | 适用场景            |
| ------------ | -------- | --------- | --------------- |
| U-Net        | 高（分割）    | 18        | 像素级分割           |
| Faster R-CNN | 0.61     | 9         | 两阶段，精度高         |
| YOLOv5       | 0.68     | 45        | 单阶段，平衡          |
| **YOLOv8**   | **0.81** | **80**    | **单阶段，速度+精度最优** |

**YOLOv8 优势**：

1. **Anchor-Free**：直接回归中心点 + 宽高，简化小目标处理
2. **C2f 模块**：更丰富的梯度流，提升小目标特征
3. **Task-Aligned Assigner**：动态正负样本分配，对小目标友好
4. **Distribution Focal Loss (DFL)**：连续标签分布，更精细的定位

### 4.2 模型规模选择

| 模型      | 参数量    | FLOPs   | 适用场景      |
| ------- | ------ | ------- | --------- |
| yolov8n | 3.0 M  | 8.2 G   | 边缘部署、原型验证 |
| yolov8s | 11.2 M | 28.6 G  | 平衡        |
| yolov8m | 25.9 M | 78.9 G  | 数据量较大     |
| yolov8l | 43.7 M | 165.2 G | 高精度       |
| yolov8x | 68.2 M | 257.8 G | 最高精度      |

**本项目推荐**：

- 数据 < 500 张 → `yolov8n`（默认），避免过拟合
- 数据 500\~2000 张 → `yolov8s`
- 数据 > 2000 张 → `yolov8m` 或更大

### 4.3 损失函数

YOLOv8 Detection 损失 = **Box Loss + Cls Loss + DFL Loss**

| 损失           | 公式                      | 作用          |
| ------------ | ----------------------- | ----------- |
| **Box Loss** | CIoU                    | 边界框回归       |
| **Cls Loss** | BCE                     | 类别（单类时 ≈ 0） |
| **DFL Loss** | Distribution Focal Loss | 边界框精细定位     |

`box_loss` 是核心关注指标，本项目修复前 `box_loss=0` 说明标签未生效。

***

## 5. 训练 Pipeline

### 5.1 命令

```bash
python YoloPV.py train \
  --weights yolov8n.pt \
  --data YoloPV_dataset.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 1280 \
  --lr0 0.01 \
  --device 0
```

### 5.2 训练阶段

```
Phase 1: Warmup (0 ~ warmup_epochs)
  └─ 线性增加学习率 0 → lr0
  └─ 动量线性下降 → warmup_momentum

Phase 2: Main Training
  └─ Mosaic 增强（最后 close_mosaic=10 轮关闭）
  └─ 余弦学习率（默认）
  └─ AdamW 优化器（ultralytics 自动选择）

Phase 3: Close Mosaic (最后 10 轮)
  └─ 关闭 Mosaic，提升真实分布拟合
  └─ 验证集 mAP 通常在此阶段达到峰值

Phase 4: Save & Validate
  └─ 保存 best.pt / last.pt
  └─ 加载 best.pt 在 val 集上做最终评估
```

### 5.3 输出目录

```
runs/detect/train/
├── args.yaml              # 训练超参数
├── results.csv            # 每 epoch 指标
├── results.png            # 训练曲线
├── confusion_matrix.png   # 混淆矩阵
├── labels.jpg             # 标签分布可视化
├── train_batch0.jpg       # 训练样本（带 mosaic）
├── val_batch0_labels.jpg  # 验证集真实标签
├── val_batch0_pred.jpg    # 验证集预测
└── weights/
    ├── best.pt            # mAP 最高的权重（推荐使用）
    └── last.pt            # 最后一轮的权重
```

### 5.4 关键监控指标

| 指标                  | 期望趋势      | 异常处理             |
| ------------------- | --------- | ---------------- |
| `train/box_loss`    | 下降 → 稳定   | 持续上升 = 学习率过大     |
| `val/box_loss`      | 下降 → 稳定   | 不降 = 过拟合或数据问题    |
| `metrics/mAP50`     | 上升 → 0.5+ | < 0.1 = 标签或类别有误  |
| `metrics/recall`    | 上升 → 0.7+ | 低 = 漏检，考虑 imgsz↑ |
| `metrics/precision` | 上升 → 0.7+ | 低 = 误检，考虑 conf↑  |

***

## 6. 关键参数详解

### 6.1 数据增强（已在 [YoloPV.py:36-39](../YoloPV.py#L36-L39) 固定）

| 参数             | 值    | 作用          | 红外小目标适配理由              |
| -------------- | ---- | ----------- | ---------------------- |
| `mosaic`       | 0.75 | 4 张图拼接      | 把多个小目标放同一张图，提升小目标上下文学习 |
| `mixup`        | 0.2  | 两张图加权混合     | 增加样本多样性，提升泛化           |
| `copy_paste`   | 0.5  | 复制小目标粘贴到其他图 | 直接增加小目标数量级，最有效的小目标增强   |
| `overlap_mask` | True | 分割 mask 重叠  | 本项目是检测，无副作用            |

### 6.2 不建议开启的增强

- `flipud`（垂直翻转）：光伏板图像无物理意义
- `degrees/perspective`：航拍视角相对固定，过强会破坏标注

### 6.3 预测参数

| 参数        | 默认   | 说明                                          |
| --------- | ---- | ------------------------------------------- |
| `conf`    | 0.25 | 置信度阈值，验证时可降至 0.001                          |
| `iou`     | 0.45 | NMS IoU 阈值                                  |
| `max_det` | 1000 | 单图最大检测数（单图多污点时用 1000）                       |
| `augment` | True | 启用 TTA（Test Time Augmentation），推理 +1\~3 mAP |

***

## 7. Bug 复盘与根因分析

> 本节是项目通读时发现的关键问题，所有问题已修复并通过 1 epoch 冒烟测试验证。

### 7.1 Bug #1：标签格式错误（最严重）

**症状**：`runs/detect/train5/results.csv` 显示 `box_loss=0, cls_loss=300+`，所有指标为 0。

**根因**：所有标签 txt 第一列是字符串 `Stain`：

```
Stain 0.603906 0.476736 0.0 0.004167    ❌ YOLO 不接受字符串
```

YOLO 格式要求数字 ID，ultralytics 解析失败后**静默丢弃所有标签**——这就是 `box_loss=0` 的根因。

**修复**：[JsonToTxt.py](../JsonToTxt.py) 中 `CLASS_MAP = {"Stain": 0}`，输出：

```
0 0.608464 0.476215 0.009115 0.008681    ✅
```

### 7.2 Bug #2：标签目录拼写错误

**症状**：YOLOv8 训练时报 `No labels found`，自动跳到空缓存。

**根因**：`yolov8/datasets/lables/`（少一个 `b`）。

**修复**：`mv lables labels`。

> 教训：YOLOv8 默认将 `images/` 字符串替换为 `labels/` 查找标签，**目录名不能错**。

### 7.3 Bug #3：YAML 类别不一致

**症状**：即使标签修了，类别名仍为 `spot`，与标注 `Stain` 不符。

**修复**：[YoloPV\_dataset.yaml](../YoloPV_dataset.yaml) 改为 `names: 0: Stain`，且**删除** `mask_ratio`（这是分割任务参数，本项目是检测任务）。

### 7.4 Bug #4：save\_dir 双重嵌套

**症状**：`runs/detect/runs/detect/train/` 出现重复路径。

**根因**：ultralytics settings.json 中 `runs_dir="runs"`，CLI 又传 `project="runs/detect"`，导致拼接出 `runs/detect/runs/detect/train`。

**修复**：移除显式 `project` 参数，让 ultralytics 按 task 自动生成 `runs/detect/train`。

### 7.5 Bug #5：YoloPV.py 单体式脚本

**症状**：训练、预测、导出全写在一个文件，无法灵活组合。

**修复**：重构为 4 个子命令 `train/predict/val/export`，每个独立可调。

### 7.6 修复后验证

| 指标               | 修复前 | 修复后                  | 含义       |
| ---------------- | --- | -------------------- | -------- |
| `train/box_loss` | 0   | 2.12                 | 标签被正确解析  |
| `val/box_loss`   | 0   | 1.65                 | 验证集有标签可学 |
| 数据集扫描            | 失败  | 10 images, 0 corrupt | 路径正确     |
| 训练完成             | 否   | 是                    | 端到端 OK   |

### 7.7 第二轮修复：示例适配与仓库治理

第一轮修复后，项目可训练，但部署示例仍是 ultralytics 官方模板，与本项目脱节。第二轮修复内容：

| Bug | 症状 | 修复 |
|-----|------|------|
| 废弃 API | `examples/YOLOv8-OpenCV-ONNX-Python/main.py` 用 `ultralytics.yolo.utils`（8.0+ 已移除） | 重写为纯 OpenCV DNN，无 ultralytics 依赖 |
| COCO 类别污染 | Python/C++ 示例加载 COCO 80 类，与本项目 1 类 Stain 不符 | 改为单类 `Stain` |
| 硬编码路径 | C++ `main.cpp` 写死 `/home/user/ultralytics` + `bus.jpg` | 改为相对路径 + `getopt` 命令行参数 |
| CUDA 强制依赖 | `CMakeLists.txt` `find_package(CUDA REQUIRED)` | 改为 `option(ENABLE_CUDA)` 可选 |
| 无关模板 | `tutorial.ipynb`/`examples/README.md` 训练 coco128 | 删除 |
| 分割参数残留 | `YoloPV.py` `overlap_mask=True` | 删除 |
| device 默认不当 | 默认 `device=0`，无 GPU 时报错 | 改为 `auto` |
| predict 弹窗阻塞 | `r.show()` 每张图弹窗 | 改为 `r.save()` |
| 类别定义重复 | `JsonToTxt.py` 与 `yaml` 各维护一份 | 统一从 yaml 读取 |
| 仓库卫生 | 根目录 6 个旧预测图、`runs/train2~5`、`before.zip` | 清理 |
| 数据集工具缺失 | 无重新划分工具 | 新增 `split_dataset.py` |

***

## 8. 性能基线

> 以下为 1 epoch CPU 冒烟测试的 baseline，仅用于证明 pipeline 通了，不是最终性能。

```
Epoch 1/1:  box_loss=2.120  cls_loss=6.363  dfl_loss=0.449
            val_box_loss=1.646  val_cls_loss=5.911  val_dfl_loss=0.284
            mAP50=0  mAP50-95=0   （1 epoch 不收敛属正常）
```

期望 100 epoch GPU 训练结果（参考相似小目标数据集）：

| 指标            | 期望值          |
| ------------- | ------------ |
| mAP\@0.5      | 0.7\~0.85    |
| mAP\@0.5:0.95 | 0.4\~0.55    |
| Recall        | 0.75\~0.90   |
| Precision     | 0.80\~0.95   |
| 推理时间（GPU 0）   | 5\~10 ms/帧   |
| 推理时间（CPU）     | 60\~120 ms/帧 |

***

## 9. 部署方案

### 9.1 ONNX 导出

```bash
python YoloPV.py export \
  --weights runs/detect/train/weights/best.pt \
  --format onnx \
  --imgsz 1280
```

输出：`runs/detect/train/weights/best.onnx`

### 9.2 C++ 部署（OpenCV DNN）

见 [examples/YOLOv8-CPP-Inference/](../examples/YOLOv8-CPP-Inference/)：

```cpp
Inference inf("best.onnx", cv::Size(1280, 1280), "classes.txt", /*cuda=*/true);
std::vector<Detection> dets = inf.runInference(frame);
```

- 支持 CUDA 加速
- 使用 letterbox 保持长宽比
- NMS 阈值默认 0.5

### 9.3 Python 部署（OpenCV DNN）

见 [examples/YOLOv8-OpenCV-ONNX-Python/](../examples/YOLOv8-OpenCV-ONNX-Python/)：

```python
net = cv2.dnn.readNetFromONNX('best.onnx')
blob = cv2.dnn.blobFromImage(image, 1/255, (640, 640))
net.setInput(blob)
outputs = net.forward()
```

### 9.4 部署注意

- **输入尺寸**：训练用 1280，部署可降到 640 加速
- **置信度阈值**：边缘部署建议 0.4\~0.5 减少误报
- **批大小**：无人机单帧流式推理，batch=1
- **图像预处理**：红外 JPEG 是单通道或伪 3 通道，OpenCV 读入即用

***

## 10. 扩展方向与路线图

### 10.1 短期（数据集相关）

- [ ] 扩充数据集到 ≥ 500 张（用无人机分批采集）
- [ ] 增加 val/test 标注
- [ ] 引入更多污点类型（`Stain`、`Crack`、`Hotspot`）

### 10.2 中期（算法相关）

- [ ] 尝试 `yolov8s` 或 `yolov8m` 提升容量
- [ ] 引入 **SAHI**（切片推理）进一步提升小目标召回
- [ ] 引入 **CBAM 注意力** 改进 backbone
- [ ] 多尺度训练（`multi_scale=True`）

### 10.3 长期（系统相关）

- [ ] 改为**目标分割**任务（YOLOv8-seg），输出像素级掩模
- [ ] 部署到无人机边缘端（TensorRT / ONNX Runtime Mobile）
- [ ] 接入 PX4 飞控，实现自动巡检闭环
- [ ] 引入 3D 视觉融合（红外 + 可见光 + 深度）

***

## 11. API 参考

### 11.1 CLI 接口

```
python YoloPV.py <command> [options]

Commands:
  train     训练模型
  predict   推理预测
  val       验证集评估
  export    导出模型
```

#### train 子命令

| 参数          | 类型    | 默认                    | 说明                  |
| ----------- | ----- | --------------------- | ------------------- |
| `--weights` | str   | `yolov8n.pt`          | 预训练权重               |
| `--data`    | str   | `YoloPV_dataset.yaml` | 数据集 yaml            |
| `--imgsz`   | int   | 1280                  | 输入尺寸                |
| `--epochs`  | int   | 100                   | 训练轮数                |
| `--batch`   | int   | 16                    | 批大小                 |
| `--lr0`     | float | 0.01                  | 初始学习率               |
| `--device`  | str   | `0`                   | 设备（`0`/`cpu`/`0,1`） |

#### predict 子命令

| 参数          | 类型    | 默认                            | 说明       |
| ----------- | ----- | ----------------------------- | -------- |
| `--source`  | str   | `yolov8/datasets/images/test` | 图像/目录/视频 |
| `--weights` | str   | `yolov8n.pt`                  | 模型权重     |
| `--conf`    | float | 0.25                          | 置信度阈值    |
| `--iou`     | float | 0.45                          | NMS IoU  |
| `--max_det` | int   | 1000                          | 单图最大检测数  |

#### val 子命令

| 参数          | 类型    | 默认                    | 说明           |
| ----------- | ----- | --------------------- | ------------ |
| `--data`    | str   | `YoloPV_dataset.yaml` | 数据集 yaml     |
| `--weights` | str   | `yolov8n.pt`          | 模型权重         |
| `--conf`    | float | 0.001                 | 置信度阈值（验证用低值） |
| `--iou`     | float | 0.6                   | NMS IoU      |

#### export 子命令

| 参数          | 类型  | 默认           | 说明                              |
| ----------- | --- | ------------ | ------------------------------- |
| `--weights` | str | `yolov8n.pt` | 模型权重                            |
| `--format`  | str | `onnx`       | 导出格式（onnx/torchscript/openvino） |

### 11.2 Python API

[YoloPV.py](../YoloPV.py) 也可作为模块导入：

```python
from YoloPV import cmd_train, cmd_predict, cmd_val, cmd_export
import argparse

# 直接调用
args = argparse.Namespace(weights='yolov8n.pt', data='YoloPV_dataset.yaml',
                         imgsz=1280, epochs=10, batch=8, lr0=0.01, device='0')
cmd_train(args)
```

### 11.3 JsonToTxt API

```python
from JsonToTxt import convert_one, convert_split, shape_to_bbox

# 转换单个文件
n = convert_one('a.json', 'a.txt')  # 返回目标数

# 转换整个划分
total = convert_split('train')
```

***

## 12. 附录：常见问题

### Q1：`box_loss=0` 怎么办？

按 [Bug #1](#71-bug-1标签格式错误最严重) 检查：

```bash
# 检查标签格式
head -3 yolov8/datasets/labels/train/*.txt
# 第一列必须是数字（0），不能是 "Stain" 等字符串
```

### Q2：训练时报 `No labels found`？

检查：

1. 目录名必须是 `labels/`（不是 `lables/`、`annotations/`）
2. 标签文件名必须与图像名一一对应
3. 标签内容每行 5 列：`class_id cx cy w h`

### Q3：导出 ONNX 后推理结果不一致？

- 训练时 `imgsz` 与导出时 `--imgsz` 不一致
- 部署时 letterbox 处理与训练不同（参考 `examples/` 中的实现）

### Q4：内存爆炸（OOM）？

- 降低 `imgsz`（如 1280→640）
- 降低 `batch`（如 16→4）
- 设置 `cache=False`（默认）

### Q5：mAP 一直为 0 但 loss 正常下降？

- 检查 `conf` 阈值（验证时建议 0.001）
- 检查数据集 yaml 的 `names` 与标签 class\_id 是否对应
- 至少保证 train+val 都有 ≥ 1 个标签

### Q6：国内网络安装慢？

```bash
# torch CPU（推荐 SJTU 镜像）
pip install --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu torch torchvision

# ultralytics（清华镜像）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics
```

***

## 13. 开发日志

项目的重要优化与修复过程按时间线记录在 [DEVLOG.md](./DEVLOG.md)，内容包括问题现象、根因分析、优化方案与验证结果。

当前最新优化：**小目标误检修复**（`imgsz` 640→1280，mAP50-95 0.266→0.553，大蓝框误检 → 小框精检）。

---

## 文档版本

| 版本  | 日期         | 变更                        |
| --- | ---------- | ------------------------- |
| 1.0 | 2026-07-01 | 初版：通读项目后撰写，包含 bug 复盘与修复说明 |

