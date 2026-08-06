"""
YoloPV —— 基于 YOLOv8 的光伏板红外小目标（污点）检测。

提供训练、预测、验证、导出四种工作流，通过命令行子命令选择：

    python YoloPV.py train   --epochs 100 --device 0
    python YoloPV.py predict --source yolov8/datasets/images/test --weights runs/detect/train/weights/best.pt
    python YoloPV.py val     --weights runs/detect/train/weights/best.pt
    python YoloPV.py export  --weights runs/detect/train/weights/best.pt --format onnx

不带子命令时默认执行：训练 -> 预测 -> 导出 ONNX（用于首次跑通流程）。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# 默认参数
DATA_YAML = "YoloPV_dataset.yaml"
WEIGHTS = "yolov8n.pt"          # 预训练权重（训练起点；也可换 yolov8s/m/l/x）
BEST_WEIGHTS = "runs/detect/train/weights/best.pt"  # 已训练好的最优权重
IMGSZ = 1280                    # 高分辨率利于小目标
PREDICT_DIR = "yolov8/datasets/images/test"


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
        name="train",
        exist_ok=True,
    )


def cmd_predict(args):
    model = YOLO(args.weights)
    results = model.predict(
        args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        augment=True,        # TTA 测试时增强
        max_det=args.max_det,
        save=True,
        name="predict",
        exist_ok=True,
    )
    for r in results:
        r.save()
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
    sp.set_defaults(func=cmd_train)

    # predict
    sp = sub.add_parser("predict", help="推理预测")
    add_common(sp)
    sp.add_argument("--source", default=PREDICT_DIR, help="图像/目录/视频路径")
    sp.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    sp.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值（小目标用较高值避免框被合并）")
    sp.add_argument("--max_det", type=int, default=1000, help="单图最大检测数")
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
            epochs=100, batch=16, lr0=0.01,
        )
        cmd_train(train_args)

        best = Path(BEST_WEIGHTS)
        predict_args = argparse.Namespace(
            weights=str(best), source=PREDICT_DIR, imgsz=IMGSZ, device="auto",
            conf=0.25, iou=0.7, max_det=1000,
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
