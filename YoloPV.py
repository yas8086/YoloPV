"""
YoloPV —— 基于 YOLOv8 的光伏板红外小目标（污点）检测。

提供训练、预测、验证、导出四种工作流，通过命令行子命令选择：

    python YoloPV.py train   --epochs 100 --device 0
    python YoloPV.py predict --source yolov8/datasets/images/test --weights runs/detect/train/weights/best.pt
    python YoloPV.py val     --weights runs/detect/train/weights/best.pt
    python YoloPV.py export  --weights runs/detect/train/weights/best.pt --format onnx
    python YoloPV.py smoke   # 冒烟测试：1 epoch 训练 + 单图预测，验证流程无异常

不带子命令时默认执行：训练 -> 预测 -> 导出 ONNX（用于首次跑通流程）。
"""

import argparse
import hashlib
import yaml
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

# 默认参数
DATA_YAML = "YoloPV_dataset.yaml"
WEIGHTS = "yolov8n.pt"          # 预训练权重（训练起点；也可换 yolov8s/m/l/x）
BEST_WEIGHTS = "runs/detect/train/weights/best.pt"  # 已训练好的最优权重
IMGSZ = 1280                    # 高分辨率利于小目标
PREDICT_DIR = "yolov8/datasets/images/test"


def write_data_version(data_yaml: str, out_dir: Path):
    """记录本次训练使用的数据集版本，便于复现与追溯。

    对 images/{train,val,test} 下的图片文件名做哈希，连同各划分样本数
    写入 <out_dir>/data_version.yaml。若多次训练数据集不同，哈希会变化。
    """
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[警告] 无法读取数据配置 {data_yaml}: {e}")
        return

    root = Path(cfg.get("path", "yolov8/datasets"))
    summary = {}
    for split in ("train", "val", "test"):
        img_dir = root / cfg.get(split, f"images/{split}")
        names = sorted(
            p.name for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ) if img_dir.exists() else []
        digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:12]
        summary[split] = {"count": len(names), "hash": digest}

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "data_version.yaml"
    try:
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump({"data_yaml": data_yaml, "splits": summary}, f, allow_unicode=True)
        print(f"数据版本已写入 {target}")
        print(f"  train={summary['train']['count']}张 hash={summary['train']['hash']}")
    except Exception as e:
        print(f"[警告] 写入数据版本失败: {e}")


def cmd_train(args):
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        # 小目标增强策略（已在光伏污点上验证最优）
        single_cls=True,   # 单类污点，更稳定
        mosaic=1.0,        # 增强小目标上下文
        mixup=0.3,         # 提升泛化性
        copy_paste=1.0,    # 最强小目标增强（污点多样性）
        close_mosaic=20,   # 最后20轮关闭 mosaic 稳定收敛
        fliplr=0.0,        # 红外光伏板不需左右翻转
        flipud=0.0,
        scale=0.1,         # 小目标不能缩放太猛
        translate=0.05,
        perspective=0.0,
        degrees=0.0,
        shear=0.0,
        hsv_h=0.0,         # 红外图对颜色敏感，减弱颜色增强
        hsv_s=0.3,
        hsv_v=0.2,
        erasing=0.1,       # 减弱擦除，避免擦掉小目标
        name=args.name,
        exist_ok=True,
    )
    # 训练完成后记录数据集版本，便于复现
    write_data_version(args.data, Path("runs/detect") / args.name)


def dedupe_boxes(xyxy, conf, min_dist=5.0):
    """基于中心距离的贪心去重，消除同一目标上的重复小框。

    NMS 依据 IoU，对几像素的小框去重效果差，导致同一污点输出多个
    几乎重叠的框（中心距离 <2px）。这里按置信度降序贪心：优先保留
    置信度最高的框，丢弃所有中心距离 < min_dist 的框。

    返回去重后保留的索引（按置信度降序）。
    """
    if len(xyxy) == 0:
        return []
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    order = np.argsort(-conf)
    keep = []
    for i in order:
        if all(np.hypot(cx[i] - cx[j], cy[i] - cy[j]) >= min_dist for j in keep):
            keep.append(i)
    return keep


def draw_detections(img, xyxy, conf, keep, color=(0, 0, 255)):
    """在图像上绘制去重后的检测框与置信度。"""
    for i in keep:
        x1, y1, x2, y2 = xyxy[i]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{conf[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(int(y1) - th - 4, 0)
        cv2.rectangle(img, (int(x1), ty), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1) + 2, int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def cmd_predict(args):
    model = YOLO(args.weights)
    results = model.predict(
        args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        augment=True,        # TTA 测试时增强
        max_det=args.max_det,
        save=False,          # 关闭自动保存，改用去重后手绘
        verbose=False,
    )
    save_dir = Path("runs/detect") / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        img = r.orig_img.copy()
        n = 0
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            keep = dedupe_boxes(xyxy, conf, min_dist=args.dedupe)
            draw_detections(img, xyxy, conf, keep)
            n = len(keep)
        out = save_dir / (Path(r.path).stem + ".jpg")
        cv2.imwrite(str(out), img)
        print(f"  [{args.name}] {Path(r.path).name} 去重后 {n} 个目标 -> {out.name}")
    return results


def cmd_val(args):
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        name="val",
        exist_ok=True,
    )
    print(f"\n验证结果：mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    return metrics


def cmd_export(args):
    model = YOLO(args.weights)
    success = model.export(format=args.format, imgsz=args.imgsz)
    print(f"导出 {'成功' if success else '失败'}：{success}")
    return success


def cmd_smoke(args):
    """冒烟测试：快速跑通 训练(1 epoch) -> 预测 全流程，验证无异常。"""
    print("=" * 60)
    print("冒烟测试：1 epoch 快速训练 -> 单图预测")
    print("=" * 60)

    # 1. 快速训练 1 epoch 到临时目录
    train_args = argparse.Namespace(
        weights=args.weights, data=args.data, imgsz=args.imgsz, device=args.device,
        epochs=1, batch=2, lr0=0.001, name=args.name,
    )
    cmd_train(train_args)

    # 2. 用刚训练的最优权重预测单张测试图
    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"[冒烟测试失败] 未找到训练产物 {best}")
    predict_args = argparse.Namespace(
        weights=str(best), source=args.source, imgsz=args.imgsz, device=args.device,
        conf=0.25, iou=0.7, max_det=1000, name=f"{args.name}_predict", dedupe=5.0,
    )
    results = cmd_predict(predict_args)
    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        print(f"  [冒烟] {r.path.split('/')[-1]} 检测到 {n} 个目标")

    print("=" * 60)
    print("冒烟测试通过：训练与预测流程均正常运行")
    print("=" * 60)
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="YoloPV 光伏板红外污点检测工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # 公共参数
    def add_common(sp):
        sp.add_argument("--weights", default=WEIGHTS, help="模型权重")
        sp.add_argument("--imgsz", type=int, default=IMGSZ, help="输入尺寸")
        sp.add_argument("--device", default="auto", help="设备（auto/cpu/0/0,1）")

    # train
    sp = sub.add_parser("train", help="训练模型")
    add_common(sp)
    sp.add_argument("--data", default=DATA_YAML, help="数据集 yaml")
    sp.add_argument("--epochs", type=int, default=100, help="训练轮数")
    sp.add_argument("--batch", type=int, default=16, help="批大小")
    sp.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    sp.add_argument("--name", default="train", help="训练输出目录名（位于 runs/detect/）")
    sp.set_defaults(func=cmd_train)

    # predict
    sp = sub.add_parser("predict", help="推理预测")
    add_common(sp)
    sp.add_argument("--source", default=PREDICT_DIR, help="图像/目录/视频/RTSP 路径")
    sp.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    sp.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值（小目标用较高值避免框被合并）")
    sp.add_argument("--max_det", type=int, default=1000, help="单图最大检测数")
    sp.add_argument("--dedupe", type=float, default=5.0, help="重复框去重中心距离阈值(px)")
    sp.add_argument("--name", default="predict", help="推理输出目录名（位于 runs/detect/）")
    sp.set_defaults(func=cmd_predict)

    # val
    sp = sub.add_parser("val", help="在验证集上评估")
    add_common(sp)
    sp.add_argument("--data", default=DATA_YAML, help="数据集 yaml")
    sp.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    sp.add_argument("--iou", type=float, default=0.6, help="NMS IoU 阈值")
    sp.set_defaults(func=cmd_val)

    # export
    sp = sub.add_parser("export", help="导出模型")
    add_common(sp)
    sp.add_argument("--format", default="onnx", help="导出格式（onnx/torchscript/openvino 等）")
    sp.set_defaults(func=cmd_export)

    # smoke
    sp = sub.add_parser("smoke", help="冒烟测试（1 epoch 训练 + 单图预测）")
    add_common(sp)
    sp.add_argument("--data", default=DATA_YAML, help="数据集 yaml")
    sp.add_argument("--source", default=PREDICT_DIR, help="预测源（取第一张）")
    sp.add_argument("--name", default="smoke", help="冒烟测试输出目录名（位于 runs/detect/）")
    sp.set_defaults(func=cmd_smoke)

    return p.parse_args()


def main():
    args = parse_args()

    if args.command is None:
        # 默认流程：训练 -> 预测 -> 导出（首次跑通用）
        print("=" * 60)
        print("未指定子命令，执行默认流程：train -> predict -> export")
        print("可用子命令：train / predict / val / export")
        print("=" * 60)

        train_args = argparse.Namespace(
            weights=WEIGHTS, data=DATA_YAML, imgsz=IMGSZ, device="auto",
            epochs=100, batch=16, lr0=0.01, name="train",
        )
        cmd_train(train_args)

        best = Path(BEST_WEIGHTS)
        predict_args = argparse.Namespace(
            weights=str(best), source=PREDICT_DIR, imgsz=IMGSZ, device="auto",
            conf=0.25, iou=0.7, max_det=1000, name="predict", dedupe=5.0,
        )
        cmd_predict(predict_args)

        export_args = argparse.Namespace(
            weights=str(best), imgsz=IMGSZ, device="auto", format="onnx",
        )
        cmd_export(export_args)
        return

    args.func(args)


if __name__ == "__main__":
    main()
