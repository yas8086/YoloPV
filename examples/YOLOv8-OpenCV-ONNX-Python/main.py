"""
YoloPV ONNX 推理示例（OpenCV DNN，无 ultralytics 依赖）。

用法：
    python main.py                                    # 默认推理 ../../yolov8/datasets/images/test
    python main.py --model ../../runs/detect/train/weights/best.onnx --source a.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# 本项目类别（须与 YoloPV_dataset.yaml 一致）
CLASSES = ["Stain"]
COLORS = [(0, 0, 255)]  # 红色框

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "runs" / "detect" / "train" / "weights" / "best.onnx"
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "yolov8" / "datasets" / "images" / "test"


def parse_args():
    p = argparse.ArgumentParser(description="YoloPV ONNX 推理示例（OpenCV DNN）")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="ONNX 模型路径")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="图像/目录路径")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    p.add_argument("--save-dir", default="output", help="结果保存目录")
    return p.parse_args()


def letterbox(img, new_shape=640):
    """保持长宽比缩放 + padding 到正方形（与 YOLOv8 训练预处理一致）。"""
    h, w = img.shape[:2]
    r = new_shape / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, left = pad_h // 2, pad_w // 2
    out = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


def draw_box(img, class_id, conf, x, y, x2, y2):
    label = f"{CLASSES[class_id]} {conf:.2f}"
    color = COLORS[class_id % len(COLORS)]
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    cv2.putText(img, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def infer_one(net, img, imgsz, conf_thres, iou_thres):
    """对单张图推理，返回原图上的检测结果列表。"""
    h0, w0 = img.shape[:2]
    blob_img, ratio, (pad_w, pad_h) = letterbox(img, imgsz)
    blob = cv2.dnn.blobFromImage(blob_img, scalefactor=1 / 255.0, size=(imgsz, imgsz), swapRB=False)
    net.setInput(blob)
    out = net.forward()  # shape: (1, 4+nc, N)

    out = out[0].T  # (N, 4+nc)
    boxes, scores, class_ids = [], [], []
    for row in out:
        box = row[:4]          # cx, cy, w, h（在 imgsz 坐标系）
        cls_scores = row[4:]
        max_score = cls_scores.max()
        if max_score < conf_thres:
            continue
        cls_id = int(cls_scores.argmax())
        # 转回原图坐标（去 padding + 去 letterbox 缩放）
        cx, cy, w, h = box
        x1 = (cx - w / 2 - pad_w) / ratio
        y1 = (cy - h / 2 - pad_h) / ratio
        x2 = (cx + w / 2 - pad_w) / ratio
        y2 = (cy + h / 2 - pad_h) / ratio
        x1 = max(0, min(int(x1), w0))
        y1 = max(0, min(int(y1), h0))
        x2 = max(0, min(int(x2), w0))
        y2 = max(0, min(int(y2), h0))
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(max_score))
        class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    return [(class_ids[i], scores[i], boxes[i]) for i in indices.flatten()]


def main():
    args = parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(f"模型不存在：{args.model}（请先运行 python YoloPV.py export）")

    net = cv2.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    src = Path(args.source)
    img_paths = sorted([src] if src.is_file() else list(src.glob("*.jp*g")))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[跳过] 无法读取：{p}")
            continue
        dets = infer_one(net, img, args.imgsz, args.conf, args.iou)
        for cid, conf, box in dets:
            x, y, w, h = [int(v) for v in box]
            draw_box(img, cid, conf, x, y, x + w, y + h)
        out_path = save_dir / f"det_{p.name}"
        cv2.imwrite(str(out_path), img)
        total += len(dets)
        print(f"{p.name}: {len(dets)} 个目标 -> {out_path}")

    print(f"\n完成：{len(img_paths)} 张图，共 {total} 个目标，结果保存到 {save_dir}/")


if __name__ == "__main__":
    main()
