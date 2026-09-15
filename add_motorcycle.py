"""
从 COCO val2017 中提取含摩托车(motorcycle)的图片
转换为 YOLO 格式，class_id=5 (motorcycle)
"""

import json
import os
import shutil
from pathlib import Path

# COCO val2017 路径
COCO_DIR = Path(__file__).parent / 'dataset_moto'
COCO_ANN = COCO_DIR / 'annotations' / 'instances_val2017.json'
COCO_IMG = COCO_DIR / 'val2017'

# 输出到现有数据集
DATASET_DIR = Path(__file__).parent / 'dataset'
TRAIN_IMG = DATASET_DIR / 'train' / 'images'
TRAIN_LBL = DATASET_DIR / 'train' / 'labels'
VAL_IMG = DATASET_DIR / 'val' / 'images'
VAL_LBL = DATASET_DIR / 'val' / 'labels'

# COCO motorcycle category_id = 4
MOTO_CAT_ID = 4
# 我们数据集中的类别id (motorcycle = 5)
OUR_MOTO_CLASS = 5

print("加载 COCO 标注...")
with open(COCO_ANN, 'r') as f:
    coco = json.load(f)

# 找出所有含 motorcycle 的图片ID
moto_anns = [a for a in coco['annotations'] if a['category_id'] == MOTO_CAT_ID]
moto_img_ids = set(a['image_id'] for a in moto_anns)
print(f"含摩托车的图片: {len(moto_img_ids)} 张")
print(f"摩托车标注数: {len(moto_anns)} 个")

# 图片ID → 文件名映射
img_info = {img['id']: img for img in coco['images']}

# 按 80/20 分 train/val
img_ids = sorted(moto_img_ids)
split = int(len(img_ids) * 0.8)
train_ids = img_ids[:split]
val_ids = img_ids[split:]

print(f"分配: train={len(train_ids)}, val={len(val_ids)}")

def convert_and_save(ids, img_dir, lbl_dir):
    """转换并保存"""
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_id in ids:
        info = img_info[img_id]
        fname = info['file_name']
        w = info['width']
        h = info['height']
        
        # 该图片的所有 motorcycle 标注
        anns = [a for a in moto_anns if a['image_id'] == img_id]
        
        # YOLO 格式: class_id center_x center_y width height (归一化)
        lines = []
        for ann in anns:
            x, y, bw, bh = ann['bbox']  # COCO: x,y,w,h
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # clip to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            lines.append(f"{OUR_MOTO_CLASS} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        # 复制图片 (加 coco_ 前缀避免跟 UA-DETRAC 重名)
        src = COCO_IMG / fname
        dst_name = f"coco_{fname}"
        dst = img_dir / dst_name
        if not src.exists():
            continue
        shutil.copy2(src, dst)
        
        # 写标注
        lbl_path = lbl_dir / (dst_name.replace('.jpg', '.txt'))
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(lines))
        
        count += 1
    
    return count

print("处理 train...")
n1 = convert_and_save(train_ids, TRAIN_IMG, TRAIN_LBL)
print(f"  train: {n1} 张图片")

print("处理 val...")
n2 = convert_and_save(val_ids, VAL_IMG, VAL_LBL)
print(f"  val: {n2} 张图片")

print(f"\n完成! 共添加 {n1 + n2} 张摩托车图片")
print(f"类别 5 = motorcycle")
