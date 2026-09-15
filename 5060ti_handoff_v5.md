# 5060Ti 交接 Prompt - V5 远距离增强训练

## 你的任务
在现有数据集上做远距离增强 + 修复标注ID + 重新训练。

## 当前 5060Ti 上的目录

```
E:\dev\yjwlYOLO\dev\
├── dataset\
│   ├── train\images\, train\labels\    ← UA-DETRAC + Roboflow 已合并
│   ├── val\images\, val\labels\
│   ├── archive\                        ← Roboflow 原始数据（用完可以删）
│   └── data.yaml                       ← nc=8, 8个类别
├── runs\
│   ├── train\weights\best.pt           ← v1 权重
│   ├── train_v2\weights\               ← v2 权重
│   └── train_v4_merge\weights\         ← v4 权重 (5060KaFine1.pt)
└── train_yolo26_v4.ipynb
```

## 类别说明 (nc=8)

```
ID  类名        来源           说明
0   car         UA-DETRAC      小汽车
1   bus         两个数据集都有  公交车
2   van         UA-DETRAC      面包车
3   truck       UA-DETRAC      卡车
4   others      UA-DETRAC      其他车辆
5   bicycle     Roboflow       自行车
6   motorbike   Roboflow       摩托车/电瓶车
7   person      Roboflow       行人
```

## ⚠️ 重要问题：标注ID冲突

两个数据集的标注文件里类别ID不一样：

```
UA-DETRAC 标注:  class 0=car, 1=bus, 2=van, 3=truck, 4=others  ✅ 正确

Roboflow 标注:   class 0=bicycle, 1=bus, 2=car, 3=motorbike, 4=person  ❌ 需要重映射
应该改成:        class 5=bicycle, 1=bus, 0=car, 6=motorbike, 7=person
```

Roboflow 文件的特征：文件名里包含 `.rf.`，例如：
`aguanambi-1000_png_jpg.rf.0ab6f274892b9b370e6441886b2d7b9d.txt`

## 步骤1：修复 Roboflow 标注ID

新建 `fix_labels.py`，内容如下，然后运行：

```python
"""修复 Roboflow 标注文件的类别ID"""
from pathlib import Path

DATASET = Path('E:/dev/yjwlYOLO/dev/dataset')

# Roboflow原始ID → 正确ID
# bicycle 0→5, bus 1→1(不变), car 2→0, motorbike 3→6, person 4→7
REMAP = {0: 5, 2: 0, 3: 6, 4: 7}

for split in ['train', 'val']:
    lbl_dir = DATASET / split / 'labels'
    count = 0
    for f in lbl_dir.glob('*.txt'):
        if '.rf.' not in f.name:
            continue
        lines = []
        changed = False
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 5:
                    old_id = int(parts[0])
                    if old_id in REMAP:
                        parts[0] = str(REMAP[old_id])
                        changed = True
                lines.append(' '.join(parts))
        if changed:
            with open(f, 'w') as fh:
                fh.write('\n'.join(lines))
            count += 1
    print(f'{split}: 修复了 {count} 个文件')

print('修复完成!')
```

运行：`python fix_labels.py`

## 步骤2：远距离增强

新建 `augment_far.py`，内容如下，然后运行：

```python
"""远距离增强：缩小图片模拟远处小目标，所有类别"""
import cv2
import numpy as np
from pathlib import Path

DATASET = Path('E:/dev/yjwlYOLO/dev/dataset')

# 不同类别用不同缩放比例
# 汽车大目标少缩，摩托/行人/自行车小目标多缩
CLASS_SCALES = {
    0: [0.5, 0.6, 0.7],          # car
    1: [0.5, 0.6, 0.7],          # bus
    2: [0.5, 0.6, 0.7],          # van
    3: [0.5, 0.6, 0.7],          # truck
    4: [0.4, 0.5, 0.6],          # others
    5: [0.3, 0.4, 0.5, 0.6],     # bicycle
    6: [0.3, 0.4, 0.5, 0.6],     # motorbike
    7: [0.3, 0.4, 0.5, 0.6],     # person
}

def augment_far(img_path, lbl_path, out_img_dir, out_lbl_dir, scale):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    x_off = (w - new_w) // 2
    y_off = (h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = small
    lines = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            new_cx = (cx * new_w + x_off) / w
            new_cy = (cy * new_h + y_off) / h
            new_bw = bw * new_w / w
            new_bh = bh * new_h / h
            if new_bw < 0.008 or new_bh < 0.008:
                continue
            lines.append(f'{cls_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}')
    if not lines:
        return False
    stem = img_path.stem + f'_far{int(scale*100)}'
    cv2.imwrite(str(out_img_dir / f'{stem}.jpg'), canvas)
    with open(out_lbl_dir / f'{stem}.txt', 'w') as f:
        f.write('\n'.join(lines))
    return True

def main():
    print('=== 全类别远距离增强 ===')
    for split in ['train', 'val']:
        img_dir = DATASET / split / 'images'
        lbl_dir = DATASET / split / 'labels'
        all_files = []
        for img_path in img_dir.glob('*.jpg'):
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                all_files.append((img_path, lbl_path))
        print(f'{split}: {len(all_files)} 张图片，开始增强...')
        count = 0
        for img_path, lbl_path in all_files:
            with open(lbl_path) as f:
                content = f.read()
            cls_ids = set()
            for line in content.strip().split('\n'):
                parts = line.split()
                if len(parts) == 5:
                    cls_ids.add(int(parts[0]))
            scales = set()
            for cid in cls_ids:
                scales.update(CLASS_SCALES.get(cid, [0.5, 0.6]))
            for scale in scales:
                if augment_far(img_path, lbl_path, img_dir, lbl_dir, scale):
                    count += 1
            if count % 2000 == 0 and count > 0:
                print(f'  已生成 {count} 张...')
        print(f'  完成: {count} 张增强图片')
    train_imgs = len(list((DATASET / 'train' / 'images').iterdir()))
    val_imgs = len(list((DATASET / 'val' / 'images').iterdir()))
    print(f'\n总计: train={train_imgs}, val={val_imgs}')

if __name__ == '__main__':
    main()
```

运行：`python augment_far.py`

⚠️ 这个脚本会比较慢（几十万张图），耐心等。

## 步骤3：确认 data.yaml

`dataset/data.yaml` 内容应该是：

```yaml
path: E:\dev\yjwlYOLO\dev\dataset
train: train/images
val: val/images

nc: 8
names: ['car', 'bus', 'van', 'truck', 'others', 'bicycle', 'motorbike', 'person']
```

## 步骤4：训练

新建 `train_yolo26_v5.ipynb`，3个cell：

**Cell 1 - 训练：**
```python
from ultralytics import YOLO

model = YOLO('E:/dev/yjwlYOLO/dev/runs/train_v2/weights/best.pt')

results = model.train(
    data='E:/dev/yjwlYOLO/dev/dataset/data.yaml',
    epochs=15,
    imgsz=640,
    batch=32,
    device=0,
    workers=0,
    freeze=20,
    patience=5,
    lr0=0.001,
    scale=0.5,
    mosaic=1.0,
    mixup=0.1,
    project='E:/dev/yjwlYOLO/dev/runs',
    name='train_v5_far',
    exist_ok=True,
)
```

**Cell 2 - 评估：**
```python
from ultralytics import YOLO

model = YOLO('E:/dev/yjwlYOLO/dev/runs/train_v5_far/weights/best.pt')
results = model.val(data='E:/dev/yjwlYOLO/dev/dataset/data.yaml', device='cpu')

print(f"mAP50:     {results.box.map50:.4f}")
print(f"mAP50-95:  {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall:    {results.box.mr:.4f}")
```

**Cell 3 - 复制权重：**
```python
import shutil
shutil.copy2(
    'E:/dev/yjwlYOLO/dev/runs/train_v5_far/weights/best.pt',
    'E:/dev/yjwlYOLO/dev/runs/train_v5_far/weights/5060ti_v5.pt'
)
print('Done: 5060ti_v5.pt')
```

## 步骤5：训练完成后

把 `runs\train_v5_far\weights\5060ti_v5.pt` 拷回本机。

## 预期结果

v2 已经会识别 car/bus/van/truck/others，微调主要学 bicycle/motorbike/person。
加了远距离增强后，小目标检测能力应该明显提升。

## 执行顺序

```
1. python fix_labels.py        ← 修复标注 (1分钟)
2. python augment_far.py       ← 远距离增强 (30-60分钟)
3. 检查 data.yaml              ← 确认 nc=8
4. 跑 train_yolo26_v5.ipynb    ← 训练 (1-2小时)
5. 拷回权重
```
