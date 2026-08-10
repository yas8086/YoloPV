# 基于 YOLOv8 的光伏板红外小目标检测

> 利用无人机红外图像，自动定位光伏板表面的鸟粪/沙尘等热斑污点。
> 一阶段检测器，端到端训练，支持 ONNX/C++ 部署。

| 任务 | 检测（污点定位，非分割）|
|------|------------------------|
| 类别 | 1 类：`Stain`（污点）|
| 骨干 | YOLOv8n（可换 s/m/l/x）|
| 输入 | 红外 JPEG，1280×1280 |
| 框架 | Ultralytics 8.4 + PyTorch |
| 文档 | [技术文档](docs/TECHNICAL.md) · [开发日志](docs/DEVLOG.md) |

---

## 目录

- [项目背景](#项目背景)
- [技术演进](#技术演进)
- [数据集](#数据集)
- [快速开始](#快速开始)
- [命令行接口](#命令行接口)
- [修复历史（踩坑记录）](#修复历史踩坑记录)
- [开发日志](#开发日志)
- [训练调优建议](#训练调优建议)
- [部署示例](#部署示例)
- [项目结构](#项目结构)
- [许可证与引用](#许可证与引用)

---

## 项目背景

### 1. 行业痛点

随着全球光伏装机容量突破 1TW（数据来源：国际能源署 2023 报告），光伏板表面污垢造成的发电效率损失已成为行业重大挑战：

- 沙尘覆盖可使组件效率下降 18–35%
- 鸟粪等局部污垢可引发热斑效应，组件温度最高达 85℃
- 传统人工巡检效率低下，百兆瓦级电站全面检测需 200+ 工时

![红外热斑示例](https://github.com/yas8086/YoloPV/blob/main/readmeIMG/jpeg_mipi3_0_20250331114219.jpeg)

### 2. 技术挑战

无人机红外检测方案面临多重技术壁垒：

- **小目标特性**：污垢区域在 1024×1024 红外图像中仅占 5×5~20×20 像素
- **低信噪比**：热扩散效应导致目标边缘模糊，SNR < 3dB
- **动态干扰**：无人机航拍引起运动模糊（PSF ≈ 3.5px）
- **多尺度问题**：光伏板与无人机距离变化导致目标尺寸差异达 10 倍

![低信噪比污点](https://github.com/yas8086/YoloPV/blob/main/readmeIMG/jpeg_dvp1_0_20240620164306.jpeg)

## 技术演进

| 方法         | mAP@0.5 | FPS  | 缺陷               |
| :----------- | :------ | :--- | :----------------- |
| U-Net        | 0.52    | 18   | 误检率高           |
| Faster R-CNN | 0.61    | 9    | 小目标漏检         |
| YOLOv5       | 0.68    | 45   | 热斑边缘定位不精确 |
| **YOLOv8**   | **0.81**| **80**| **本方案**        |

> 注：上表为文献中针对该类红外小目标场景的代表性结果，本项目数据集较小（13 张），指标以训练收敛曲线为准。

## 数据集

数据集位于 [yolov8/datasets/](yolov8/datasets)：

| 划分  | 图像数 | 标签目标数 | 类别 |
|-------|--------|------------|------|
| train | 10     | 28         | Stain |
| val   | 3      | 3          | Stain |
| test  | 6      | —（待标注）| —     |

> **注意**：当前数据集规模较小，建议扩充到 ≥ 500 张后再追求高 mAP。

目录结构（遵循 YOLOv8 约定）：

```
yolov8/datasets/
├── images/{train,val,test}/   # 红外 JPEG
└── labels/{train,val}/        # YOLO 格式：class_id cx cy w h （均归一化到 [0,1]）
```

如需从 LabelMe JSON 重新生成标签：

```bash
python JsonToTxt.py                     # 转换 train + val
python JsonToTxt.py --split train        # 仅转换 train
```

详见 [JsonToTxt.py](JsonToTxt.py)。

## 快速开始

```bash
# 1. 克隆 / 进入项目
cd YoloPV

# 2. 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
# 国内网络可加速：
#   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics
#   pip install --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu torch torchvision

# 4. 训练（GPU）
python YoloPV.py train --epochs 100 --device 0

# 5. 预测
python YoloPV.py predict --weights runs/detect/train/weights/best.pt

# 视频 / RTSP 流推理（--source 支持视频文件或 RTSP 地址）
python YoloPV.py predict --weights runs/detect/train/weights/best.pt --source rtsp://user:pass@192.168.1.100:554/stream
python YoloPV.py predict --weights runs/detect/train/weights/best.pt --source video.mp4
```

> 不带 GPU 时 `--device cpu` 也能跑，但速度较慢。冒烟测试建议 `epochs 1~3`、`imgsz 320` 即可。

## 命令行接口

[YoloPV.py](YoloPV.py) 提供四个子命令：

```bash
python YoloPV.py <command> [options]
```

| 子命令  | 作用 | 关键参数 |
|---------|------|---------|
| `train` | 训练模型 | `--data, --epochs, --batch, --lr0, --imgsz, --device, --name` |
| `predict`| 推理预测 | `--source, --weights, --conf, --iou, --max_det, --dedupe, --name` |
| `val`   | 验证集评估 | `--data, --weights, --conf, --iou` |
| `export`| 导出模型 | `--weights, --format`（onnx/torchscript/openvino） |
| `smoke` | 冒烟测试（1 epoch 训练 + 单图预测） | `--data, --source, --name` |

> `--source` 支持图像、目录、视频文件、RTSP 流地址。
> 训练完成后会在输出目录生成 `data_version.yaml`，记录各划分样本数与哈希，用于数据版本追溯与复现。
> `predict` 默认开启重复框去重（`--dedupe 5.0`）：按中心距离合并同一污点上的重复小框，使"框数 = 污点数"更准确。

查看任意子命令帮助：

```bash
python YoloPV.py train --help
```

不带子命令时自动执行 `train → predict → export` 一条龙。

### 数据集工具

| 工具 | 作用 | 用法 |
|------|------|------|
| [JsonToTxt.py](JsonToTxt.py) | LabelMe JSON → YOLO TXT | `python JsonToTxt.py --split all` |
| [split_dataset.py](split_dataset.py) | 重新划分 train/val/test | `python split_dataset.py --ratio 0.8 0.1 0.1` |

### 训练参数默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `yolov8n.pt` | 预训练权重（n/s/m/l/x）|
| `--imgsz` | 1280 | 高分辨率利于小目标 |
| `--epochs` | 100 | 训练轮数 |
| `--batch` | 16 | 批大小 |
| `--lr0` | 0.01 | 初始学习率（ultralytics 会自动调） |
| `--device` | `0` | GPU 0；CPU 用 `--device cpu` |

### 关键增强参数（已在 [YoloPV.py](YoloPV.py) 中固定）

- `mosaic=1.0`：拼接增强小目标上下文（最后 20 轮自动关闭）
- `mixup=0.3`：提升泛化
- `copy_paste=1.0`：最强小目标增强（复制粘贴污点样本）
- 关闭翻转（`fliplr/flipud=0`）、`scale=0.1`、减弱颜色/擦除增强：避免破坏微小污点特征

## 修复历史（踩坑记录）

### 第一轮：核心 Bug 修复（训练失败根因）

| # | 问题 | 后果 | 修复 |
|---|------|------|------|
| 1 | 标签 txt 第一列是字符串 `Stain` | YOLO 解析失败 → `box_loss=0` | [JsonToTxt.py](JsonToTxt.py) 改为输出数字 ID `0` |
| 2 | 标签目录拼写 `lables/` | 找不到标签 | 重命名为 `labels/` |
| 3 | `YoloPV_dataset.yaml` 类别名 `spot` 与标注 `Stain` 不符 | 类别映射错位 | yaml 修正为 `Stain` |
| 4 | yaml 含 `mask_ratio`（分割任务参数） | 误用 | 删除 |
| 5 | `YoloPV.py` 写死训练/预测/导出，且 `save_dir` 双重嵌套 | 不可复用、产物乱 | 重构为 CLI 子命令 |
| 6 | `tests/` 目录是从 ultralytics 仓库复制的样板（用废弃 API） | 跑会报错 | 删除 |
| 7 | 缺 `requirements.txt` 与 `.gitignore` | 不完整 | 新增 |

### 第二轮：示例适配与仓库治理

| # | 问题 | 修复 |
|---|------|------|
| 8 | `examples/YOLOv8-OpenCV-ONNX-Python/main.py` 用废弃 API `ultralytics.yolo` + 加载 COCO 80 类 | 重写为纯 OpenCV DNN + 单类 Stain |
| 9 | `examples/YOLOv8-CPP-Inference/main.cpp` 硬编码 `/home/user/ultralytics` + `bus.jpg` | 改为相对路径 + 命令行参数 |
| 10 | `inference.h` 硬编码 COCO 80 类标签 | 改为单类 `Stain` |
| 11 | `CMakeLists.txt` 强制要求 CUDA | 改为可选（`-DENABLE_CUDA=ON`） |
| 12 | `tutorial.ipynb`/`examples/README.md` 是 ultralytics 官方模板 | 删除 |
| 13 | `YoloPV.py` 含 `overlap_mask=True`（分割参数）、默认 `device=0`（CPU 报错）、`predict` 每张弹窗 | 删除参数、改 `auto`、改 `save` |
| 14 | 类别定义在 `JsonToTxt.py` 与 `yaml` 两处重复 | 统一从 yaml 读取 |
| 15 | 根目录 6 个 `results_jpeg_*.jpeg` + `runs/detect/train2~5` 历史产物 | 清理 |
| 16 | `数据集标注/before.zip` 重复备份 | 删除 |
| 17 | 数据集 10:3 失衡、test 无标签 | 新增 [split_dataset.py](split_dataset.py) 辅助重新划分 |

**修复后端到端验证**（1 epoch CPU 冒烟测试）：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| `train/box_loss` | **0** | **2.12** |
| `val/box_loss`   | **0** | **1.65** |
| 数据集扫描 | 失败 | `10 images, 0 corrupt` |
| 训练完成 | 否 | 是 |

详见 [docs/TECHNICAL.md](docs/TECHNICAL.md) 第 7 节。

## 开发日志

持续记录项目的优化与修复过程，包括问题现象、根因分析、方案与验证结果。

- 最新：**小目标误检修复**（imgsz 640→1280，mAP50-95 0.266→0.553，大蓝框误检 → 小框精检）
- **重复框去重**：新增基于中心距离的后处理，消除同一污点上的重叠框，"框数=污点数"更准确
- **数据集扩充调研**：确认现有原始数据（dvp1/mipi3 两相机）无可扩充的有效污点样本，需采集新数据人工标注

详见 [docs/DEVLOG.md](docs/DEVLOG.md)。

## 训练调优建议

针对红外小目标检测的关键经验：

1. **分辨率优先**：`imgsz=1280` 起步，小目标对下采样非常敏感
2. **数据增强**：保留 `mosaic=1.0`、`copy_paste=1.0`，关闭 `flipud`/`fliplr`（光伏板不会上下/左右颠倒）
3. **预测时 TTA + 去重**：`augment=True` 提升召回；`--dedupe 5.0` 消除同一污点上的重复小框
4. **小数据集**：先冻结骨干 10 epoch 微调，再全量训练；或换 `yolov8s.pt` 提升容量
5. **正负样本不均**：本项目单类，可设 `single_cls=True` 加速

## 部署示例

`examples/` 提供了导出后 ONNX 模型的推理示例：

- [YOLOv8-CPP-Inference/](examples/YOLOv8-CPP-Inference/)：C++ + OpenCV DNN，支持 CUDA
- [YOLOv8-OpenCV-ONNX-Python/](examples/YOLOv8-OpenCV-ONNX-Python/)：Python + OpenCV DNN

先导出模型再使用示例：

```bash
python YoloPV.py export --weights runs/detect/train/weights/best.pt --format onnx
# 产物: runs/detect/train/weights/best.onnx
```

## 项目结构

```
YoloPV/
├── YoloPV.py                       # CLI 主程序：train/predict/val/export/smoke
├── YoloPV_dataset.yaml             # 数据集配置
├── JsonToTxt.py                    # LabelMe JSON -> YOLO TXT 转换
├── requirements.txt                # 依赖
├── yolov8n.pt                      # 预训练权重
├── docs/
│   ├── TECHNICAL.md                # 技术文档（架构/算法/数据流/调优）
│   └── DEVLOG.md                   # 开发日志（优化/修复过程）
├── yolov8/datasets/
│   ├── images/{train,val,test}/
│   └── labels/{train,val}/
├── runs/detect/                    # 训练/预测产物
└── examples/
    ├── YOLOv8-CPP-Inference/       # C++ 部署
    └── YOLOv8-OpenCV-ONNX-Python/  # Python 部署
```

## 许可证与引用

- 本项目代码：仅作学习用途
- 依赖：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（AGPL-3.0）
- 训练数据：来自无人机红外巡检采集，请勿商用
