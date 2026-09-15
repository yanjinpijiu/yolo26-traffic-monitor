# 项目规范 - 基于YOLO26的车流量检测系统

## 基本信息
- 课程: 深度学习实践
- 学校: 温州大学
- 学号: 23211880215

## 技术栈
- 模型架构: YOLO26（必须，不可替换为YOLOv8等）
- 框架: Ultralytics 8.4.48
- 训练环境: Windows Anaconda + PyTorch 2.8.0+cu126 + RTX 4060
- 推理设备: CPU（展示实际部署速度）
- 界面: PyQt5
- 跟踪: ByteTrack

## 数据集
- UA-DETRAC 车辆检测数据集
- 4个有效类别: car, bus, van, others
- truck类别验证集无样本
- 训练集: 67957帧, 验证集: 14128帧

## 模型版本记录
| 版本 | 权重 | imgsz | freeze | 备注 |
|------|------|-------|--------|------|
| v1 | yolo26n.pt | 480 | 10 | baseline，冻结主干微调，按验证集指标早停 |
| v2 | yolo26n.pt | 640 | 0 | 提高输入尺寸，全参数训练 |
| v3 | v2 权重 | 640 | 部分冻结 | 新增 motorcycle 类别微调 |

各版本的训练与验证指标记录在 `results/<版本>/results.csv`，横向对比见 README。

## 项目结构
```
E:\dev\yjwlYOLO\dev\
├── PROJECT.md              # 本文件
├── convert_to_yolo.py      # 数据集格式转换
├── train_yolo26.ipynb      # v1训练 + 评估 + 测速
├── train_yolo26_v2.ipynb   # v2优化版训练
├── count_vehicles.py       # 检测+跟踪+计数脚本
├── gui_app.py              # PyQt5界面
├── dataset/                # YOLO格式数据集
├── runs/                   # 训练输出
└── test/                   # 测试文件
```

## 评估API注意事项
- 类别AP值: `metrics.box.ap_class_index`（不是ap_class_indices）
- 遍历方式: 
  ```python
  all_names = list(model.names.values())
  for i, cls_idx in enumerate(metrics.box.ap_class_index):
      print(f'{all_names[cls_idx]}: {metrics.box.ap50[i]:.4f}')
  ```

## Git规范
- 主分支: main
- 大文件排除: *.pt, *.mp4, dataset/, runs/
