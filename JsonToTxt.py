"""
将 LabelMe 标注的 JSON 文件转换为 YOLO 检测格式的 TXT 标签文件。

YOLO 检测格式（每行一个目标）：
    class_id center_x center_y width height   （均归一化到 [0,1]）

支持两种 shape_type：
    - rectangle：直接由两个点求中心点和宽高
    - polygon  ：取所有顶点的最小外接矩形

用法：
    python JsonToTxt.py                          # 转换 train 和 val
    python JsonToTxt.py --split train            # 只转换 train
    python JsonToTxt.py --json_dir a --txt_dir b # 自定义目录
"""

import argparse
import json
from pathlib import Path

import yaml

# 数据集根目录
DATASET_ROOT = Path(__file__).parent / "yolov8" / "datasets"
# 数据集配置（作为类别定义的唯一真相源）
DATA_YAML = Path(__file__).parent / "YoloPV_dataset.yaml"


def load_class_map():
    """从 YoloPV_dataset.yaml 读取类别定义，反转得到 name->id 映射。"""
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # names 可能是 dict {0: 'Stain'} 或 list ['Stain']
    names = cfg["names"]
    if isinstance(names, list):
        return {n: i for i, n in enumerate(names)}
    return {n: i for i, n in names.items()}


# 兼容历史标注的旧类别名（合并到主映射）
_LEGACY_ALIASES = {"spot": 0}


def parse_args():
    parser = argparse.ArgumentParser(description="LabelMe JSON -> YOLO TXT 转换工具")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="要转换的数据集划分（默认 all，即 train+val+test）",
    )
    parser.add_argument("--json_dir", help="自定义 JSON 输入目录（覆盖 --split）")
    parser.add_argument("--txt_dir", help="自定义 TXT 输出目录（覆盖 --split）")
    return parser.parse_args()


def shape_to_bbox(shape, width, height, class_map):
    """
    将一个 LabelMe shape（rectangle 或 polygon）转换为归一化的
    (class_id, center_x, center_y, box_w, box_h)。
    无法识别的返回 None。
    """
    label = shape.get("label", "")
    if label not in class_map:
        print(f"  [跳过] 未知类别 '{label}'")
        return None

    points = shape.get("points", [])
    if not points:
        return None

    if shape.get("shape_type") == "rectangle":
        # LabelMe rectangle：两点为对角
        (x1, y1), (x2, y2) = points[0], points[1]
    else:
        # polygon：取所有顶点的外接矩形
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

    # 防止越界与零尺寸
    x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = box_w / width
    h = box_h / height

    # 限制到 [0,1]
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return class_map[label], round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def convert_one(json_path, txt_path, class_map):
    """转换单个 JSON 文件到对应 TXT 文件。返回写入的目标数量。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = data["imageWidth"]
    height = data["imageHeight"]
    shapes = data.get("shapes", [])

    lines = []
    for shape in shapes:
        result = shape_to_bbox(shape, width, height, class_map)
        if result is None:
            continue
        cid, cx, cy, w, h = result
        lines.append(f"{cid} {cx} {cy} {w} {h}")

    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return len(lines)


def convert_split(split, class_map):
    """转换一个划分目录（train/val/test）下的所有 JSON。"""
    json_dir = DATASET_ROOT / "labels" / split
    if not json_dir.exists():
        print(f"[警告] 目录不存在: {json_dir}")
        return 0

    json_files = sorted([p for p in json_dir.iterdir() if p.suffix == ".json"])
    total = 0
    for jf in json_files:
        tf = jf.with_suffix(".txt")
        n = convert_one(jf, tf, class_map)
        total += n
        print(f"  {jf.name} -> {tf.name}  ({n} 个目标)")
    return total


def main():
    args = parse_args()
    class_map = load_class_map()
    # 合并历史别名
    class_map.update(_LEGACY_ALIASES)
    print(f"类别映射：{class_map}")

    if args.json_dir and args.txt_dir:
        # 自定义模式
        json_dir = Path(args.json_dir)
        txt_dir = Path(args.txt_dir)
        txt_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for jf in sorted(json_dir.glob("*.json")):
            tf = txt_dir / (jf.stem + ".txt")
            total += convert_one(jf, tf, class_map)
        print(f"完成：共 {total} 个目标，输出到 {txt_dir}")
        return

    splits = ["train", "val"] if args.split == "all" else [args.split]
    grand_total = 0
    for s in splits:
        print(f"\n转换 [{s}] 划分：")
        grand_total += convert_split(s, class_map)
    print(f"\n完成：共写入 {grand_total} 个目标")


if __name__ == "__main__":
    main()
