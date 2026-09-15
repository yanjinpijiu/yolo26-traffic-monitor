# 基于 YOLO26 的车流监控系统

温州大学《深度学习实践》课程大作业，组长。在公开车辆检测数据集 UA-DETRAC 上训练多类车辆检测模型，
覆盖数据格式转换、迁移学习训练、多版本对比调优的完整流程，并开发可视化界面对视频 / 图像做
实时车辆检测与车流量统计。

## 功能

- **数据集转换**：把 UA-DETRAC 的 XML 标注转成 YOLO txt 格式，划分训练 / 验证集
- **类别扩充**：从 COCO val2017 提取摩托车样本，在原 4 类基础上补充 `motorcycle` 类别
- **远距离增强**：等比缩小图片模拟远处小目标，缓解远距离车辆漏检
- **模型训练**：基于 YOLO26（yolo26n）做迁移学习，先冻结主干微调，再逐步解冻做全参数训练
- **多版本对比**：对比不同输入尺寸（480 / 640）、不同冻结层与训练轮次的组合，配合早停机制筛选最优权重
- **检测与跟踪**：YOLO26 检测 + ByteTrack 跟踪 + 越线计数，支持双向双车道与轨迹存储
- **可视化界面**：PyQt5 界面，支持视频文件与摄像头输入、可拖动计数线、车道分隔线

## 技术栈

| 项目 | 选型 |
|---|---|
| 检测模型 | YOLO26（yolo26n） |
| 框架 | Ultralytics 8.4.48 |
| 训练环境 | Windows + Anaconda + PyTorch 2.8.0+cu126 + RTX 4060 Laptop 8GB |
| 推理设备 | CPU（用于展示实际部署速度） |
| 目标跟踪 | ByteTrack |
| 界面 | PyQt5 |

## 数据集

[UA-DETRAC](https://detrac-db.rit.albany.edu/) 车辆检测数据集。

- 有效类别：`car` / `bus` / `van` / `others`，另扩充 `motorcycle`
- 训练集约 6.8 万帧，验证集约 1.4 万帧
- 注：`truck` 类别在验证集中无样本，未纳入训练

数据集与模型权重体积过大，未包含在本仓库中，需自行下载后按 `convert_to_yolo.py`
顶部的路径配置放置。

## 多版本对比

围绕两个变量做横向对照，每版记录训练与验证指标，配合早停挑最优权重：

| 版本 | 初始化权重 | 输入尺寸 | 冻结层数 | 说明 |
|---|---|---|---|---|
| v1 | yolo26n.pt | 480 | 10 | baseline，冻结主干微调，按验证集指标早停 |
| v2 | yolo26n.pt | 640 | 0 | 提高输入尺寸，全参数训练 |
| v3 | v2 权重 | 640 | 部分冻结 | 新增 motorcycle 类别微调 |

训练过程中的 Loss / PSNR 曲线、PR 与 F1 曲线、混淆矩阵与验证集预测图见 [`results/`](results/)。

## 项目结构

```
dev/
├── PROJECT.md                   项目规范与版本记录
├── convert_to_yolo.py           UA-DETRAC → YOLO 格式转换
├── add_motorcycle.py            从 COCO 提取 motorcycle 类别样本
├── augment_far.py               远距离小目标增强
├── train_yolo26.ipynb           v1 训练 + 评估 + 测速
├── train_yolo26_v2.ipynb        v2 优化版训练
├── train_yolo26_v3.ipynb        v3 摩托车类别微调
├── count_vehicles.py            检测 + 跟踪 + 计数（v1）
├── count_vehicles_v2.py         双向双车道计数 + 轨迹存储（v2）
├── gui_app.py                   PyQt5 可视化界面
├── gen_pptx.js                  答辩 PPT 生成脚本
├── dataset/                     YOLO 格式数据集（未跟踪）
├── runs/                        训练输出（未跟踪）
├── results/                     精选训练结果图（已跟踪）
└── docs/                        实验报告与答辩 PPT
```

## 快速开始

```bash
pip install ultralytics==8.4.48 opencv-python pyqt5
```

1. 下载 UA-DETRAC 数据集，修改 `convert_to_yolo.py` 顶部的 `BASE_DIR` 指向数据集目录，运行生成 YOLO 格式数据
2. 需要摩托车类别时，先跑 `add_motorcycle.py` 补充样本，再用 `augment_far.py` 做远距离增强
3. 按顺序运行 `train_yolo26*.ipynb` 训练模型
4. 启动界面：

```bash
python gui_app.py
```

## 说明

- 脚本中的数据集路径是开发时的本机绝对路径（如 `E:\dev\yjwlYOLO\...`），
  换机器运行需要先改这几处配置
- `runs/` 与 `dataset/` 已在 `.gitignore` 中排除，仓库只保留精选结果图和原始训练指标文件
- 训练环境的交接说明见 [5060ti_handoff.md](5060ti_handoff.md) / [5060ti_handoff_v5.md](5060ti_handoff_v5.md)
