# 弯曲多车道线 + 动态适配 方案设计

## 核心概念

### 车道线 = 一组控制点 + 样条曲线

用户在画面上点几个点，系统自动拟合一条平滑曲线穿过这些点。

```
用户点 4 个点:  ●
               \
                ●
                 \
                  ●
                   \
                    ●

系统拟合样条:  ～～～～～～～～
```

### 多车道 = 多条线

每条线有独立的颜色和名称，可以设为"计数线"（车跨过就计数）或"区域边界"。

```
画面示意:
  ┌──────────────────────┐
  │ ╭─── 车道1(红) ───╮  │
  │ │                  │  │
  │ ╰──────────────────╯  │
  │ ╭─── 车道2(蓝) ───╮  │
  │ │                  │  │
  │ ╰──────────────────╯  │
  │ ╭─── 车道3(绿) ───╮  │
  │ │                  │  │
  │ ╰──────────────────╯  │
  └──────────────────────┘
```

### 动态适配 = 关键帧插值

用户在不同帧位置调整控制点，系统自动插值中间帧。

```
帧 0:    ～～～～ (位置A)
帧 100:  ～～～～ (位置B，往右移了)
帧 200:  ～～～～ (位置C)

系统自动插值帧 1~99、101~199 的位置
```

这样摄像头轻微晃动也能跟上。

---

## 数据结构

```python
class LaneLine:
    """一条车道线"""
    id: int                    # 编号
    name: str                  # "车道1", "车道2"
    color: tuple               # 颜色 (B,G,R)
    keyframes: dict            # {frame_num: [(x,y), (x,y), ...]}
    is_count_line: bool        # 是否为计数线（跨过就计数）
    closed: bool               # 是否闭合（区域边界用）

    # 示例:
    # keyframes = {
    #     0:   [(100,400), (200,300), (400,200), (600,150)],
    #     100: [(102,401), (203,301), (403,201), (603,151)],
    # }
```

## 曲线拟合方法

用 scipy 的 CubicSpline 做样条插值：

```python
from scipy.interpolate import CubicSpline

def fit_curve(points, num_output=100):
    """控制点 → 平滑曲线点列表"""
    if len(points) < 2:
        return points
    # 用累计弦长做参数化
    pts = np.array(points)
    diffs = np.diff(pts, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.concatenate([[0], np.cumsum(dists)])
    t = t / t[-1]  # 归一化

    cs_x = CubicSpline(t, pts[:, 0])
    cs_y = CubicSpline(t, pts[:, 1])
    t_new = np.linspace(0, 1, num_output)
    curve = np.stack([cs_x(t_new), cs_y(t_new)], axis=1).astype(int)
    return [tuple(p) for p in curve]
```

## 关键帧插值

```python
def interpolate_keyframes(keyframes, current_frame):
    """在两个关键帧之间线性插值控制点"""
    frames = sorted(keyframes.keys())
    if current_frame <= frames[0]:
        return keyframes[frames[0]]
    if current_frame >= frames[-1]:
        return keyframes[frames[-1]]

    # 找前后两个关键帧
    for i in range(len(frames) - 1):
        if frames[i] <= current_frame <= frames[i + 1]:
            f0, f1 = frames[i], frames[i + 1]
            pts0 = np.array(keyframes[f0])
            pts1 = np.array(keyframes[f1])
            ratio = (current_frame - f0) / (f1 - f0)
            interpolated = pts0 + ratio * (pts1 - pts0)
            return [tuple(p) for p in interpolated.astype(int)]
    return keyframes[frames[0]]
```

## GUI 交互流程

### 模式切换

```
┌─ 工具栏 ──────────────────────────────┐
│ [预览模式] [编辑车道] [开始检测] [停止] │
└───────────────────────────────────────┘
```

- **预览模式**: 播放视频，不检测。用户看画面了解场景。
- **编辑车道**: 暂停视频，用户在画面上点击添加控制点。
- **开始检测**: 用设定好的车道线开始检测计数。

### 编辑车道操作

```
左键点击:  在当前车道添加控制点
右键点击:  删除最近的控制点
Ctrl+Z:   撤销上一个点

车道面板（右侧）:
┌─ 车道管理 ────────────┐
│ [+ 添加车道] [删除]    │
│                        │
│ ● 车道1 (红) [计数线✓] │
│   控制点: 4个           │
│   关键帧: 2个           │
│ ● 车道2 (蓝) [计数线✓] │
│   控制点: 3个           │
│   关键帧: 1个           │
│                        │
│ [保存关键帧] [清除]     │
└────────────────────────┘
```

### 关键帧操作

```
1. 拖进度条到帧 N
2. 调整控制点位置（拖动或重新点击）
3. 点"保存关键帧" → 记录 frame_N → 当前控制点
4. 拖到帧 M，再调整，再保存
5. 系统自动插值中间帧
```

## 计数逻辑改造

当前：一条水平线，车跨过就计数。

改造后：
- 每条标记为"计数线"的车道线都参与计数
- 判断车辆中心点是否跨过某条曲线
- 跨过哪条线就在对应车道计数

```python
def point_crossed_curve(prev_pos, curr_pos, curve_points):
    """判断点是否从曲线一侧穿越到另一侧"""
    # 简化：用曲线的包围盒 + 逐段检测
    # 实际用点到曲线的距离 + 符号变化判断
    ...
```

---

## 实现步骤

### Step 1: 视频预览模式
- 加一个"预览"按钮，播放视频不检测
- 可暂停、拖进度条

### Step 2: 曲线编辑器
- 画面上点击添加控制点
- scipy.interpolate 拟合平滑曲线
- 右键删除、拖动移动

### Step 3: 多车道管理
- 车道列表面板（添加/删除/选择/重命名）
- 每条线独立颜色
- 标记是否为计数线

### Step 4: 关键帧系统
- 保存当前帧的控制点为关键帧
- 拖到不同帧调整位置再保存
- 播放时自动插值

### Step 5: 计数逻辑适配
- 从单线计数改为多曲线计数
- 车辆中心点与每条计数线做穿越检测

### Step 6: 车道线持久化
- 保存/加载车道配置到 JSON 文件
- 下次打开自动加载

---

## 依赖

需要新增:
- scipy (样条插益)

已有的:
- cv2, numpy, PyQt5 (没问题)
