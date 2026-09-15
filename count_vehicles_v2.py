"""
车流量检测系统 v2
功能: YOLO检测 + ByteTrack跟踪 + 双向双车道计数 + 轨迹存储 + 优化策略
"""

import cv2
import numpy as np
import os
import time
from collections import defaultdict
from ultralytics import YOLO


# ============ 配置参数 ============
MODEL_PATH = 'E:/dev/yjwlYOLO/dev/runs/train/weights/best.pt'
VIDEO_DIR = 'E:/dev/yjwlYOLO/数据集和权重/UA-DETRAC(车辆检测数据集8250车辆)/DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_39051'
OUTPUT_VIDEO = 'E:/dev/yjwlYOLO/dev/output_count_v2.mp4'
DEVICE = 'cpu'

# 计数线位置
COUNT_LINE_RATIO = 0.6

# 优化策略参数
MIN_TRACK_FRAMES = 5       # 轨迹最少帧数
MIN_BOX_AREA = 500         # 最小框面积（像素）
TRAJECTORY_LENGTH = 30     # 轨迹线显示最近N帧

CONF_THRESHOLD = 0.4

COLORS = {
    'car': (0, 255, 0),
    'bus': (255, 0, 0),
    'van': (0, 165, 255),
    'truck': (0, 0, 255),
    'others': (128, 128, 128),
}


# ============ 轨迹存储体系 ============
class TrackManager:
    """管理所有车辆的轨迹数据"""

    def __init__(self):
        self.tracks = {}  # {track_id: {'positions':[], 'classes':[], 'max_area':0}}

    def update(self, track_id, cx, cy, cls_name, box_area):
        """更新轨迹"""
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'positions': [],
                'classes': [],
                'max_area': 0,
            }
        self.tracks[track_id]['positions'].append((cx, cy))
        self.tracks[track_id]['classes'].append(cls_name)
        self.tracks[track_id]['max_area'] = max(
            self.tracks[track_id]['max_area'], box_area
        )

    def get_frame_count(self, track_id):
        """获取轨迹帧数"""
        if track_id not in self.tracks:
            return 0
        return len(self.tracks[track_id]['positions'])

    def get_direction(self, track_id):
        """判断方向: 'up' 或 'down'"""
        if track_id not in self.tracks:
            return None
        positions = self.tracks[track_id]['positions']
        if len(positions) < 2:
            return None
        first_y = positions[0][1]
        last_y = positions[-1][1]
        return 'down' if last_y > first_y else 'up'

    def get_voted_class(self, track_id):
        """历史类别加权投票：取出现次数最多的类别"""
        if track_id not in self.tracks:
            return 'unknown'
        classes = self.tracks[track_id]['classes']
        if not classes:
            return 'unknown'
        # 投票
        vote = defaultdict(int)
        for c in classes:
            vote[c] += 1
        return max(vote, key=vote.get)

    def get_last_n_positions(self, track_id, n=30):
        """获取最近N个位置（用于画轨迹线）"""
        if track_id not in self.tracks:
            return []
        return self.tracks[track_id]['positions'][-n:]

    def get_max_area(self, track_id):
        if track_id not in self.tracks:
            return 0
        return self.tracks[track_id]['max_area']

    def get_start_position(self, track_id):
        if track_id not in self.tracks or not self.tracks[track_id]['positions']:
            return None
        return self.tracks[track_id]['positions'][0]

    def get_last_position(self, track_id):
        if track_id not in self.tracks or not self.tracks[track_id]['positions']:
            return None
        return self.tracks[track_id]['positions'][-1]


# ============ 计数管理 ============
class Counter:
    """双向双车道计数器"""

    def __init__(self):
        self.counted_ids = set()
        # 4个计数: 左上行、左下行、右上行、右下行
        self.counts = {
            'left_up': 0,    # 左车道上行（从上往下）
            'left_down': 0,  # 左车道下行（从下往上）
            'right_up': 0,   # 右车道上行
            'right_down': 0, # 右车道下行
        }

    def check_crossing(self, track_id, prev_y, curr_y, cx, frame_width,
                       count_line_y, track_manager):
        """
        检查是否跨线，如果跨线则计数
        返回: True 如果刚计数了
        """
        if track_id in self.counted_ids:
            return False

        # 轨迹帧数过滤
        if track_manager.get_frame_count(track_id) < MIN_TRACK_FRAMES:
            return False

        # 小目标面积过滤
        if track_manager.get_max_area(track_id) < MIN_BOX_AREA:
            return False

        # 判断是否跨线
        crossed_up = prev_y < count_line_y <= curr_y    # 从上往下
        crossed_down = prev_y > count_line_y >= curr_y  # 从下往上

        if not crossed_up and not crossed_down:
            return False

        # 判断左右车道
        mid_x = frame_width / 2
        is_left = cx < mid_x

        # 判断方向
        if crossed_up:
            if is_left:
                self.counts['left_up'] += 1
            else:
                self.counts['right_up'] += 1
        else:
            if is_left:
                self.counts['left_down'] += 1
            else:
                self.counts['right_down'] += 1

        self.counted_ids.add(track_id)
        return True

    def get_total(self):
        return sum(self.counts.values())

    def get_up_total(self):
        return self.counts['left_up'] + self.counts['right_up']

    def get_down_total(self):
        return self.counts['left_down'] + self.counts['right_down']


# ============ 绘图函数 ============
def draw_ui(frame, count_line_y, counter, fps_val, track_manager, active_ids):
    """画所有UI元素"""
    h, w = frame.shape[:2]

    # 画计数线
    cv2.line(frame, (0, count_line_y), (w, count_line_y), (0, 0, 255), 3)

    # 画车道分隔线（虚线）
    mid_x = w // 2
    for y in range(0, h, 20):
        cv2.line(frame, (mid_x, y), (mid_x, y + 10), (200, 200, 200), 1)

    # 画轨迹线
    for tid in active_ids:
        positions = track_manager.get_last_n_positions(tid, TRAJECTORY_LENGTH)
        if len(positions) < 2:
            continue
        for i in range(1, len(positions)):
            # 颜色渐变：越旧越透明
            alpha = i / len(positions)
            color = (int(255 * alpha), int(255 * (1 - alpha)), 0)
            cv2.line(frame, positions[i - 1], positions[i], color, 2)

    # 统计面板（左上角）
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y0 = 40
    cv2.putText(frame, f'Total: {counter.get_total()}', (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Up:   {counter.get_up_total()}', (20, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
    cv2.putText(frame, f'Down: {counter.get_down_total()}', (20, y0 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
    cv2.putText(frame, f'Left:  {counter.counts["left_up"]}up {counter.counts["left_down"]}dn', (20, y0 + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f'Right: {counter.counts["right_up"]}up {counter.counts["right_down"]}dn', (20, y0 + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f'FPS: {fps_val:.1f}', (20, y0 + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 车道标签
    cv2.putText(frame, 'LEFT', (w // 4 - 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, 'RIGHT', (3 * w // 4 - 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return frame


def draw_boxes(frame, boxes, track_ids, cls_ids, model_names, track_manager):
    """画检测框"""
    for box, tid, cid in zip(boxes, track_ids, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        tid_int = int(tid)
        cls_name = model_names[int(cid)]
        # 用投票后的类别显示
        voted_cls = track_manager.get_voted_class(tid_int)
        color = COLORS.get(voted_cls, COLORS.get(cls_name, (128, 128, 128)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{voted_cls} #{tid_int}'
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


# ============ 主流程 ============
def load_frames(video_dir):
    frames = []
    files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
    for f in files:
        img = cv2.imread(os.path.join(video_dir, f))
        if img is not None:
            frames.append(img)
    return frames


def run():
    print("=" * 60)
    print("车流量检测系统 v2 (双向双车道 + 轨迹 + 优化策略)")
    print("=" * 60)

    # 加载
    print(f"[1/4] 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"[2/4] 加载视频帧: {VIDEO_DIR}")
    frames = load_frames(VIDEO_DIR)
    h, w = frames[0].shape[:2]
    count_line_y = int(h * COUNT_LINE_RATIO)
    print(f"      共 {len(frames)} 帧, {w}x{h}, 计数线 Y={count_line_y}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (w, h))

    # 初始化
    track_manager = TrackManager()
    counter = Counter()
    id_last_y = {}
    total_time = 0

    print(f"[3/4] 开始处理 (设备: {DEVICE})")
    for idx, frame in enumerate(frames):
        start = time.time()

        results = model.track(
            frame, persist=True, tracker='bytetrack.yaml',
            device=DEVICE, conf=CONF_THRESHOLD, verbose=False,
        )
        elapsed = time.time() - start
        total_time += elapsed
        current_fps = 1.0 / elapsed if elapsed > 0 else 0

        active_ids = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cid in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid_int = int(tid)
                cls_name = model.names[int(cid)]
                box_area = (x2 - x1) * (y2 - y1)

                # 更新轨迹
                track_manager.update(tid_int, cx, cy, cls_name, box_area)
                active_ids.append(tid_int)

                # 检查跨线
                if tid_int in id_last_y:
                    counter.check_crossing(
                        tid_int, id_last_y[tid_int], cy, cx, w,
                        count_line_y, track_manager
                    )
                id_last_y[tid_int] = cy

            # 画框
            frame = draw_boxes(frame, boxes, track_ids, cls_ids, model.names, track_manager)

        # 画UI
        frame = draw_ui(frame, count_line_y, counter, current_fps, track_manager, active_ids)
        out.write(frame)

        if (idx + 1) % 100 == 0 or idx == 0:
            avg_fps = (idx + 1) / total_time if total_time > 0 else 0
            print(f"      帧 {idx+1}/{len(frames)} | FPS: {current_fps:.1f} | "
                  f"平均: {avg_fps:.1f} | 总计: {counter.get_total()} "
                  f"(上行:{counter.get_up_total()} 下行:{counter.get_down_total()})")

    out.release()
    avg_fps = len(frames) / total_time

    print(f"[4/4] 完成!")
    print(f"      总帧数: {len(frames)}")
    print(f"      总耗时: {total_time:.1f}秒")
    print(f"      平均FPS: {avg_fps:.1f}")
    print(f"      总车流量: {counter.get_total()}")
    print(f"        上行: {counter.get_up_total()}")
    print(f"        下行: {counter.get_down_total()}")
    print(f"        左上行: {counter.counts['left_up']}")
    print(f"        左下行: {counter.counts['left_down']}")
    print(f"        右上行: {counter.counts['right_up']}")
    print(f"        右下行: {counter.counts['right_down']}")
    print(f"      输出: {OUTPUT_VIDEO}")
    print("=" * 60)


if __name__ == '__main__':
    run()
