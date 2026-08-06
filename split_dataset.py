"""
数据集划分工具：将已标注数据按比例重新划分到 train/val/test。

仅处理 labels/ 下有 .txt 标签的样本（图像 + 标签成对移动）。
未标注的图像（如 camera_file/）不会被处理，需先人工标注。

用法：
    python split_dataset.py                          # 默认 0.7/0.2/0.1
    python split_dataset.py --ratio 0.8 0.1 0.1      # 自定义比例
    python split_dataset.py --ratio 0.8 0.2 0 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).parent / "yolov8" / "datasets"
IMG_DIR = DATASET_ROOT / "images"
LBL_DIR = DATASET_ROOT / "labels"


def parse_args():
    p = argparse.ArgumentParser(description="数据集重新划分工具")
    p.add_argument("--ratio", type=float, nargs=3, default=[0.7, 0.2, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="train/val/test 比例（和为 1）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--dry-run", action="store_true", help="只打印不移动")
    return p.parse_args()


def collect_paired_samples():
    """收集所有有标签的样本（跨 train/val/test 汇总）。返回 [(img_path, lbl_path), ...]。"""
    paired = []
    for split in ["train", "val", "test"]:
        img_split = IMG_DIR / split
        lbl_split = LBL_DIR / split
        if not img_split.exists():
            continue
        for img in sorted(img_split.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            # 查找对应标签（同名 .txt）
            lbl = lbl_split / (img.stem + ".txt")
            if lbl.exists():
                paired.append((img, lbl))
            else:
                print(f"[警告] {img.name} 无标签，跳过")
    return paired


def ensure_dirs():
    """确保 images/{train,val,test} 和 labels/{train,val,test} 都存在。"""
    for split in ["train", "val", "test"]:
        (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
        (LBL_DIR / split).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    if abs(sum(args.ratio) - 1.0) > 0.01:
        raise SystemExit(f"比例之和必须为 1，当前 {sum(args.ratio)}")

    paired = collect_paired_samples()
    if not paired:
        raise SystemExit("未找到任何带标签的样本")

    print(f"共找到 {len(paired)} 个带标签样本")
    random.seed(args.seed)
    random.shuffle(paired)

    n = len(paired)
    n_train = int(n * args.ratio[0])
    n_val = int(n * args.ratio[1])
    # 剩余全给 test，避免浮点误差导致丢失
    splits = {
        "train": paired[:n_train],
        "val": paired[n_train:n_train + n_val],
        "test": paired[n_train + n_val:],
    }

    ensure_dirs()
    print(f"\n划分方案（seed={args.seed}）：")
    for name, items in splits.items():
        print(f"  {name}: {len(items)} 张")

    if args.dry_run:
        print("\n[dry-run] 未实际移动文件")
        return

    # 移动文件
    for name, items in splits.items():
        for img, lbl in items:
            # 跳过已在目标目录的
            if img.parent.name == name and lbl.parent.name == name:
                continue
            dst_img = IMG_DIR / name / img.name
            dst_lbl = LBL_DIR / name / lbl.name
            shutil.move(str(img), str(dst_img))
            shutil.move(str(lbl), str(dst_lbl))

    # 清理旧 cache（划分已变）
    for cache in LBL_DIR.glob("*.cache"):
        cache.unlink()

    print(f"\n完成：已按 {args.ratio} 重新划分")


if __name__ == "__main__":
    main()
