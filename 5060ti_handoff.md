# 5060Ti 训练交接 Prompt

## 你的任务
把两个数据集合并，基于 v2 权重训练一个能识别摩托车的新模型。

## 当前目录情况

```
E:\dev\yjwlYOLO\dev\
├── dataset\
│   ├── train\images\, train\labels\    ← UA-DETRAC 数据集 (135914张)
│   ├── val\images\, val\labels\        ← UA-DETRAC 验证集 (28256张)
│   ├── archive\                        ← Roboflow 交通数据集 (5805张)
│   │   ├── train\images\, train\labels\
│   │   ├── valid\images\, valid\labels\
│   │   ├── test\images\, test\labels\
│   │   └── data.yaml                   ← 类别: bicycle, bus, car, motorbike, person
│   └── data.yaml                       ← 当前: nc=6, car/bus/van/truck/others/motorcycle
├── runs\
│   ├── train\weights\best.pt           ← v1 权重 (4060训练, imgsz=480)
│   ├── train_v2\weights\5060ti1.pt     ← v2 权重 (5060Ti训练, imgsz=640) ← 用这个做预训练
│   └── train_v3_moto\                  ← v3 权重 (效果差，不用)
├── yolo26n.pt                          ← YOLO26 预训练权重
├── gui_app.py                          ← PyQt5 界面 (最终要用新权重)
├── train_yolo26_v2.ipynb               ← v2 训练 notebook
└── train_yolo26_v3.ipynb               ← v3 训练 notebook (效果差)
```

## 数据集合并方案

两个数据集需要合并，统一类别映射：

```
UA-DETRAC 类别:     car=0, bus=1, van=2, truck=3, others=4
Roboflow 类别:      bicycle=0, bus=1, car=2, motorbike=3, person=4

合并后统一类别 (nc=8):
  0: car        (两个数据集都有)
  1: bus        (两个数据集都有)
  2: van        (仅 UA-DETRAC)
  3: truck      (仅 UA-DETRAC)
  4: others     (仅 UA-DETRAC)
  5: bicycle    (仅 Roboflow)
  6: motorbike  (仅 Roboflow)
  7: person     (仅 Roboflow)
```

### 步骤1: 复制 Roboflow 数据到 UA-DETRAC 目录

把 archive 的图片和标注复制到 dataset\train 和 dataset\val 里。

### 步骤2: 重映射 Roboflow 标注的类别ID

Roboflow 标注文件里 class_id 需要改：
```
原: bicycle=0 → 新: 5
原: bus=1     → 新: 1 (不变)
原: car=2     → 新: 0
原: motorbike=3 → 新: 6
原: person=4  → 新: 7
```

### 步骤3: 更新 data.yaml

```yaml
path: E:\dev\yjwlYOLO\dev\dataset
train: train/images
val: val/images

nc: 8
names: ['car', 'bus', 'van', 'truck', 'others', 'bicycle', 'motorbike', 'person']
```

### 步骤4: 训练

```python
from ultralytics import YOLO

model = YOLO('E:/dev/yjwlYOLO/dev/runs/train_v2/weights/5060ti1.pt')

results = model.train(
    data='E:/dev/yjwlYOLO/dev/dataset/data.yaml',
    epochs=15,
    imgsz=640,
    batch=32,
    device=0,
    workers=2,
    freeze=20,        # 冻结前20层，只训练检测头
    patience=5,
    lr0=0.001,        # 小学习率微调
    project='E:/dev/yjwlYOLO/dev/runs',
    name='train_v4_merge',
    exist_ok=True,
)
```

### 步骤5: 评估

```python
model = YOLO('E:/dev/yjwlYOLO/dev/runs/train_v4_merge/weights/best.pt')
results = model.val(data='E:/dev/yjwlYOLO/dev/dataset/data.yaml', device='cpu')

print(f"mAP50:     {results.box.map50:.4f}")
print(f"mAP50-95:  {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall:    {results.box.mr:.4f}")
```

## 预期结果

v2 已经会识别 car/bus/van/truck/others，微调只需要学 bicycle/motorbike/person。
冻结大部分层 + 小学习率 → 不会破坏原有类别 → 训练快。

## 训练完成后

把 best.pt 拷回来放到：
```
E:\dev\yjwlYOLO\dev\runs\train_v4_merge\weights\best.pt
```

然后更新 gui_app.py 的默认模型路径。
