"""
车流量检测系统 - PyQt5 简洁版
功能: 视频/摄像头 + YOLO26检测跟踪 + 可拖动计数线 + 车道分隔 + 统计
"""

import sys
import cv2
import csv
import time
import os
import numpy as np
from collections import defaultdict
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QGridLayout,
    QSlider, QTextEdit, QProgressBar, QSpinBox, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen


# ============================================================
# 配置
# ============================================================
DEFAULT_MODEL_PATH = 'E:/dev/yjwlYOLO/dev/runs/5060KaFine1.pt'
DEVICE = 'cpu'
CONF_THRESHOLD = 0.4
COUNT_LINE_Y_RATIO = 0.6       # 计数线初始位置 (画面高度的60%)
MIN_TRACK_FRAMES = 5
MIN_BOX_AREA = 500
TRAJECTORY_LENGTH = 30

COLORS = {
    'car': (0, 255, 0),
    'bus': (255, 0, 0),
    'van': (0, 165, 255),
    'truck': (0, 0, 255),
    'others': (128, 128, 128),
    'bicycle': (255, 255, 0),
    'motorbike': (0, 255, 255),
    'motorcycle': (0, 255, 255),
    'person': (255, 0, 255),
    'pedestrian': (255, 0, 255),
}

# 自动分配颜色兜底（模型输出的类别名不在 COLORS 里时）
_AUTO_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 0),
    (255, 128, 0), (0, 128, 255), (128, 0, 255), (255, 255, 128),
]
def get_color(name):
    if name in COLORS:
        return COLORS[name]
    # 自动分配一个颜色
    idx = hash(name) % len(_AUTO_COLORS)
    COLORS[name] = _AUTO_COLORS[idx]
    return _AUTO_COLORS[idx]


# ============================================================
# 轨迹管理
# ============================================================
class TrackManager:
    def __init__(self):
        self.tracks = {}

    def update(self, tid, cx, cy, cls_name, box_area, aspect_ratio=1.0):
        if tid not in self.tracks:
            self.tracks[tid] = {'positions': [], 'classes': [], 'max_area': 0, 'ratios': []}
        self.tracks[tid]['positions'].append((cx, cy))
        self.tracks[tid]['classes'].append(cls_name)
        self.tracks[tid]['max_area'] = max(self.tracks[tid]['max_area'], box_area)
        self.tracks[tid]['ratios'].append(aspect_ratio)

    def get_frame_count(self, tid):
        return len(self.tracks.get(tid, {}).get('positions', []))

    def get_voted_class(self, tid):
        classes = self.tracks.get(tid, {}).get('classes', [])
        if not classes:
            return 'unknown'
        vote = defaultdict(int)
        for c in classes:
            vote[c] += 1
        return max(vote, key=vote.get)

    def get_positions(self, tid, n=30):
        return self.tracks.get(tid, {}).get('positions', [])[-n:]

    def get_max_area(self, tid):
        return self.tracks.get(tid, {}).get('max_area', 0)

    def get_median_ratio(self, tid):
        ratios = self.tracks.get(tid, {}).get('ratios', [])
        if not ratios:
            return 1.0
        return float(np.median(ratios))


# ============================================================
# 计数器
# ============================================================
class Counter:
    def __init__(self):
        self.counted_ids = set()
        self.up = 0
        self.down = 0
        self.class_counts = defaultdict(int)

    def check(self, tid, prev_y, curr_y, line_y, track_mgr, cls_name):
        if tid in self.counted_ids:
            return False
        if track_mgr.get_frame_count(tid) < MIN_TRACK_FRAMES:
            return False
        if track_mgr.get_max_area(tid) < MIN_BOX_AREA:
            return False
        if track_mgr.get_median_ratio(tid) < 0.3:
            return False

        if prev_y < line_y <= curr_y:
            self.counted_ids.add(tid)
            self.down += 1
            voted = track_mgr.get_voted_class(tid)
            self.class_counts[voted] += 1
            return True
        elif prev_y > line_y >= curr_y:
            self.counted_ids.add(tid)
            self.up += 1
            voted = track_mgr.get_voted_class(tid)
            self.class_counts[voted] += 1
            return True
        return False

    @property
    def total(self):
        return self.up + self.down


# ============================================================
# 视频画布（支持拖动计数线）
# ============================================================
class VideoCanvas(QLabel):
    line_dragged = pyqtSignal(float)  # 发射新的 y_ratio (0~1)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(700, 500)
        self.setStyleSheet('background-color: #1e1e1e; color: white; font-size: 16px;')
        self._video_w = 0
        self._video_h = 0
        self._display_rect = QRect()
        self._dragging = False

    def set_frame(self, pixmap, vw, vh):
        self._video_w = vw
        self._video_h = vh
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._display_rect = QRect(x, y, scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        if self._display_rect.contains(event.pos()):
            self._dragging = True
            self._emit_ratio(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._emit_ratio(event.pos())

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def _emit_ratio(self, pos):
        if self._display_rect.height() == 0:
            return
        dy = pos.y() - self._display_rect.y()
        ratio = max(0.05, min(0.95, dy / self._display_rect.height()))
        self.line_dragged.emit(ratio)


# ============================================================
# 视频处理线程
# ============================================================
class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, dict)
    finished = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)

    def __init__(self, source, model_path, conf, line_y_ratio,
                 cls_conf=None):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.conf = conf
        self.line_y_ratio = line_y_ratio
        self.cls_conf = cls_conf or {}  # {class_name: threshold}
        self.running = False
        self.paused = False
        self.seek_frame = -1
        self.total_frames = 0

    def run(self):
        from ultralytics import YOLO

        self.log_signal.emit(f"加载模型: {self.model_path}")
        model = YOLO(self.model_path)
        self.log_signal.emit(f"模型类别: {model.names}")
        self.log_signal.emit("模型加载完成")

        is_camera = (self.source == 0)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.log_signal.emit("错误: 无法打开视频源")
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else -1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log_signal.emit(f"{'摄像头' if is_camera else f'视频 {self.total_frames} 帧'}, {w}x{h}")

        track_mgr = TrackManager()
        counter = Counter()
        id_last_y = {}
        total_time = 0
        frame_idx = 0

        self.running = True

        while self.running:
            while self.paused and self.running:
                time.sleep(0.05)
            if not self.running:
                break

            if self.seek_frame >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                frame_idx = self.seek_frame
                self.seek_frame = -1

            ret, frame = cap.read()
            if not ret:
                if is_camera:
                    self.log_signal.emit("摄像头读取失败")
                    break
                else:
                    self.log_signal.emit("视频处理完成")
                    break

            start = time.time()
            # 用各类别最低阈值作为全局阈值，确保低置信度类别不被过滤掉
            min_conf = min(self.cls_conf.values()) if self.cls_conf else self.conf
            results = model.track(
                frame, persist=True, tracker='bytetrack.yaml',
                device=DEVICE, conf=min_conf, verbose=False,
            )
            elapsed = time.time() - start
            total_time += elapsed
            fps = 1.0 / elapsed if elapsed > 0 else 0
            frame_idx += 1

            count_line_y = int(h * self.line_y_ratio)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                for box, tid, cid, cv in zip(boxes, track_ids, cls_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    tid_int = int(tid)
                    cls_name = model.names[int(cid)]
                    box_area = (x2 - x1) * (y2 - y1)
                    aspect_ratio = (x2 - x1) / max((y2 - y1), 1)

                    # 按类别阈值过滤
                    cls_threshold = self.cls_conf.get(cls_name, self.conf)
                    if cv < cls_threshold:
                        continue

                    track_mgr.update(tid_int, cx, cy, cls_name, box_area, aspect_ratio)

                    if tid_int in id_last_y:
                        crossed = counter.check(
                            tid_int, id_last_y[tid_int], cy,
                            count_line_y, track_mgr, cls_name,
                        )
                        if crossed:
                            voted = track_mgr.get_voted_class(tid_int)
                            d = '下行' if cy > id_last_y[tid_int] else '上行'
                            self.log_signal.emit(
                                f"计数: {voted} #{tid_int} {d} 总计={counter.total}"
                            )
                    id_last_y[tid_int] = cy

                    # 画框
                    voted = track_mgr.get_voted_class(tid_int)
                    color = COLORS.get(voted, (128, 128, 128))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{voted} #{tid_int} {cv:.2f}',
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # 轨迹线
                    pts = track_mgr.get_positions(tid_int, TRAJECTORY_LENGTH)
                    for i in range(1, len(pts)):
                        a = i / len(pts)
                        c = (int(255 * a), int(255 * (1 - a)), 0)
                        cv2.line(frame, pts[i - 1], pts[i], c, 2)

            # 画计数线
            cv2.line(frame, (0, count_line_y), (w, count_line_y), (0, 0, 255), 3)
            cv2.putText(frame, f'Count Line', (10, count_line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 统计面板
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 175), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f'Total: {counter.total}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f'Up:   {counter.up}', (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)
            cv2.putText(frame, f'Down: {counter.down}', (20, 87),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)

            y = 110
            for cls_name, cnt in counter.class_counts.items():
                c = COLORS.get(cls_name, (128, 128, 128))
                cv2.putText(frame, f'{cls_name}: {cnt}', (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                y += 20

            cv2.putText(frame, f'FPS: {fps:.1f}', (20, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            stats = {
                'count': counter.total, 'up': counter.up, 'down': counter.down,
                'class_counts': dict(counter.class_counts),
                'fps': fps, 'avg_fps': frame_idx / total_time if total_time > 0 else 0,
                'frame': frame_idx, 'total_frames': self.total_frames,
            }
            self.frame_ready.emit(frame, stats)
            if self.total_frames > 0:
                self.progress_signal.emit(frame_idx, self.total_frames)

        cap.release()
        self.running = False
        self.finished.emit(stats)

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def seek(self, n):
        self.seek_frame = n


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('车流量检测系统 - 基于YOLO26')
        self.setMinimumSize(1200, 600)
        self.video_thread = None
        self.current_stats = {}
        self.model_path = DEFAULT_MODEL_PATH
        self.source = None
        self.line_y_ratio = COUNT_LINE_Y_RATIO
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ---- 左侧：视频 ----
        left = QVBoxLayout()
        self.canvas = VideoCanvas()
        self.canvas.line_dragged.connect(self.on_line_dragged)
        left.addWidget(self.canvas)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.mousePressEvent = self._on_progress_click
        left.addWidget(self.progress_bar)

        main_layout.addLayout(left, stretch=3)

        # ---- 右侧：控制（可滚动） ----
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)

        # 输入源
        g = QGroupBox('输入源')
        l = QHBoxLayout(g)
        self.btn_file = QPushButton('选择视频文件')
        self.btn_file.clicked.connect(self.select_video)
        l.addWidget(self.btn_file)
        self.btn_cam = QPushButton('摄像头')
        self.btn_cam.clicked.connect(self.open_camera)
        l.addWidget(self.btn_cam)
        right.addWidget(g)

        # 模型
        g = QGroupBox('模型')
        l = QHBoxLayout(g)
        self.lbl_model = QLabel(os.path.basename(self.model_path))
        self.lbl_model.setWordWrap(True)
        l.addWidget(self.lbl_model, stretch=1)
        btn = QPushButton('更换')
        btn.setMaximumWidth(60)
        btn.clicked.connect(self.select_model)
        l.addWidget(btn)
        right.addWidget(g)

        # 播放控制
        g = QGroupBox('播放控制')
        l = QHBoxLayout(g)
        self.btn_start = QPushButton('开始检测')
        self.btn_start.setMinimumHeight(32)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        l.addWidget(self.btn_start)
        self.btn_pause = QPushButton('暂停')
        self.btn_pause.setMinimumHeight(32)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        l.addWidget(self.btn_pause)
        self.btn_stop = QPushButton('停止')
        self.btn_stop.setMinimumHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        l.addWidget(self.btn_stop)
        right.addWidget(g)

        # 参数
        g = QGroupBox('参数')
        l = QGridLayout(g)
        l.addWidget(QLabel('全局置信度:'), 0, 0)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(40)
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f'{v/100:.2f}'))
        l.addWidget(self.slider_conf, 0, 1)
        self.lbl_conf = QLabel('0.40')
        self.lbl_conf.setMinimumWidth(36)
        l.addWidget(self.lbl_conf, 0, 2)

        l.addWidget(QLabel('计数线位置:'), 1, 0)
        self.slider_line = QSlider(Qt.Horizontal)
        self.slider_line.setRange(10, 90)
        self.slider_line.setValue(int(self.line_y_ratio * 100))
        self.slider_line.valueChanged.connect(self.on_line_slider)
        l.addWidget(self.slider_line, 1, 1)
        self.lbl_line = QLabel(f'{self.line_y_ratio:.2f}')
        self.lbl_line.setMinimumWidth(36)
        l.addWidget(self.lbl_line, 1, 2)
        right.addWidget(g)

        # 各类别独立置信度
        g = QGroupBox('各类别置信度')
        l = QGridLayout(g)
        l.setSpacing(2)
        self.cls_conf_sliders = {}
        self.cls_conf_labels = {}
        # 默认置信度: 好检测的高一些，难检测的低一些
        cls_defaults = {
            'car': 40, 'bus': 40, 'van': 40, 'truck': 40,
            'others': 30, 'bicycle': 20, 'motorbike': 20, 'person': 30,
        }
        cls_names = ['car', 'bus', 'van', 'truck', 'others', 'bicycle', 'motorbike', 'person']
        for i, name in enumerate(cls_names):
            row, col = divmod(i, 2)
            c = COLORS.get(name, (200, 200, 200))
            hex_c = f'#{c[2]:02x}{c[1]:02x}{c[0]:02x}'
            lbl = QLabel(name)
            lbl.setStyleSheet(f'color:{hex_c}; font-weight:bold;')
            l.addWidget(lbl, row, col * 3)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(5, 80)
            slider.setValue(cls_defaults.get(name, 30))
            slider.setFixedWidth(80)
            val_lbl = QLabel(f'{cls_defaults.get(name, 30)/100:.2f}')
            val_lbl.setFixedWidth(30)
            slider.valueChanged.connect(lambda v, lbl=val_lbl: lbl.setText(f'{v/100:.2f}'))
            l.addWidget(slider, row, col * 3 + 1)
            l.addWidget(val_lbl, row, col * 3 + 2)
            self.cls_conf_sliders[name] = slider
            self.cls_conf_labels[name] = val_lbl
        right.addWidget(g)

        # 统计
        g = QGroupBox('实时统计')
        l = QGridLayout(g)
        l.setSpacing(2)
        l.addWidget(QLabel('总车流量:'), 0, 0)
        self.lbl_count = QLabel('0')
        self.lbl_count.setFont(QFont('Arial', 18, QFont.Bold))
        self.lbl_count.setStyleSheet('color: #00ff00;')
        l.addWidget(self.lbl_count, 0, 1)
        l.addWidget(QLabel('上行:'), 1, 0)
        self.lbl_up = QLabel('0')
        self.lbl_up.setStyleSheet('color: #00ccff;')
        l.addWidget(self.lbl_up, 1, 1)
        l.addWidget(QLabel('下行:'), 1, 2)
        self.lbl_down = QLabel('0')
        self.lbl_down.setStyleSheet('color: #ffcc00;')
        l.addWidget(self.lbl_down, 1, 3)
        l.addWidget(QLabel('FPS:'), 2, 0)
        self.lbl_fps = QLabel('0.0')
        l.addWidget(self.lbl_fps, 2, 1)
        l.addWidget(QLabel('帧:'), 2, 2)
        self.lbl_frame = QLabel('0')
        l.addWidget(self.lbl_frame, 2, 3)
        right.addWidget(g)

        # 各类别
        g = QGroupBox('各类别')
        l = QGridLayout(g)
        l.setSpacing(2)
        self.lbl_cls = {}
        cls_names = ['car', 'bus', 'van', 'truck', 'others', 'bicycle', 'motorbike', 'person']
        for i, name in enumerate(cls_names):
            c = COLORS.get(name, (200, 200, 200))
            hex_c = f'#{c[2]:02x}{c[1]:02x}{c[0]:02x}'
            row, col = divmod(i, 2)  # 两列排列
            lbl = QLabel(f'{name}:')
            lbl.setProperty('cssClass', 'color-label')
            lbl.setStyleSheet(f'color:{hex_c} !important; font-weight:bold;')
            l.addWidget(lbl, row, col * 2)
            val = QLabel('0')
            val.setProperty('cssClass', 'color-label')
            val.setStyleSheet(f'color:{hex_c} !important; font-size:12px;')
            l.addWidget(val, row, col * 2 + 1)
            self.lbl_cls[name] = val
        right.addWidget(g)

        # CSV + 日志
        self.btn_csv = QPushButton('导出CSV')
        self.btn_csv.clicked.connect(self.export_csv)
        right.addWidget(self.btn_csv)

        g = QGroupBox('系统日志')
        l = QVBoxLayout(g)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        l.addWidget(self.log_text)
        right.addWidget(g)

        scroll = QScrollArea()
        scroll.setWidget(right_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet('QScrollArea { border: none; background: #2b2b2b; }')
        main_layout.addWidget(scroll, stretch=1)

        self.log('系统启动。请先选择视频或摄像头，然后开始检测。')
        self.log('提示: 拖动视频画面中的红线可调整计数线位置。')

    # ---- 日志 ----
    def log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f'[{ts}] {msg}')
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    # ---- 输入源 ----
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择视频', '', '视频 (*.mp4 *.avi *.mov *.mkv);;所有 (*)')
        if path:
            self.source = path
            self.btn_file.setText(os.path.basename(path))
            self.btn_cam.setText('摄像头')
            self.btn_start.setEnabled(True)
            self.log(f'视频: {os.path.basename(path)}')

    def open_camera(self):
        self.source = 0
        self.btn_cam.setText('摄像头 ✓')
        self.btn_file.setText('选择视频文件')
        self.btn_start.setEnabled(True)
        self.log('摄像头已选择')

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择模型', '', '模型 (*.pt *.onnx);;所有 (*)')
        if path:
            self.model_path = path
            self.lbl_model.setText(os.path.basename(path))
            self.log(f'模型: {os.path.basename(path)}')

    # ---- 计数线 ----
    def on_line_dragged(self, ratio):
        self.line_y_ratio = ratio
        self.slider_line.blockSignals(True)
        self.slider_line.setValue(int(ratio * 100))
        self.slider_line.blockSignals(False)
        self.lbl_line.setText(f'{ratio:.2f}')

    def on_line_slider(self, value):
        self.line_y_ratio = value / 100.0
        self.lbl_line.setText(f'{self.line_y_ratio:.2f}')

    # ---- 播放控制 ----
    def start(self):
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setText('暂停')
        self.btn_file.setEnabled(False)
        self.btn_cam.setEnabled(False)

        for name in self.lbl_cls:
            self.lbl_cls[name].setText('0')
        self.lbl_count.setText('0')
        self.lbl_up.setText('0')
        self.lbl_down.setText('0')
        self.progress_bar.setValue(0)

        conf = self.slider_conf.value() / 100.0
        self.log(f'开始检测 (置信度={conf:.2f}, 计数线={self.line_y_ratio:.2f})')

        # 收集各类别置信度
        cls_conf = {}
        for name, slider in self.cls_conf_sliders.items():
            cls_conf[name] = slider.value() / 100.0

        self.video_thread = VideoThread(self.source, self.model_path, conf, self.line_y_ratio, cls_conf=cls_conf)
        self.video_thread.frame_ready.connect(self.on_frame)
        self.video_thread.finished.connect(self.on_finished)
        self.video_thread.log_signal.connect(self.log)
        self.video_thread.progress_signal.connect(self.on_progress)
        self.video_thread.start()

    def toggle_pause(self):
        if not self.video_thread:
            return
        if self.video_thread.paused:
            self.video_thread.resume()
            self.btn_pause.setText('暂停')
            self.log('继续')
        else:
            self.video_thread.pause()
            self.btn_pause.setText('继续')
            self.log('暂停')

    def stop(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        self._reset()

    def _reset(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText('暂停')
        self.btn_file.setEnabled(True)
        self.btn_cam.setEnabled(True)

    # ---- 帧更新 ----
    def on_frame(self, frame, stats):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.canvas.set_frame(QPixmap.fromImage(qt), w, h)

        self.lbl_count.setText(str(stats['count']))
        self.lbl_up.setText(str(stats['up']))
        self.lbl_down.setText(str(stats['down']))
        self.lbl_fps.setText(f"{stats['fps']:.1f}")
        self.lbl_frame.setText(str(stats['frame']))
        for name, val in stats.get('class_counts', {}).items():
            if name in self.lbl_cls:
                self.lbl_cls[name].setText(str(val))
        self.current_stats = stats

    def on_progress(self, cur, total):
        if total > 0:
            pct = int(cur / total * 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f'{cur}/{total} ({pct}%)')

    def on_finished(self, stats):
        self._reset()
        self.progress_bar.setValue(100)
        self.current_stats = stats
        self.log(f'完成! 共 {stats["count"]} 辆车 (上行:{stats["up"]} 下行:{stats["down"]})')

    def _on_progress_click(self, event):
        if self.video_thread and self.video_thread.isRunning() and self.video_thread.total_frames > 0:
            ratio = event.x() / self.progress_bar.width()
            self.video_thread.seek(int(ratio * self.video_thread.total_frames))

    # ---- CSV ----
    def export_csv(self):
        if not self.current_stats:
            self.log('没有数据可导出')
            return
        path, _ = QFileDialog.getSaveFileName(self, '导出CSV', 'vehicle_stats.csv', 'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['项目', '数值'])
            w.writerow(['时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            w.writerow(['总车流量', self.current_stats.get('count', 0)])
            w.writerow(['上行', self.current_stats.get('up', 0)])
            w.writerow(['下行', self.current_stats.get('down', 0)])
            w.writerow(['平均FPS', f"{self.current_stats.get('avg_fps', 0):.1f}"])
            w.writerow(['帧数', self.current_stats.get('frame', 0)])
            w.writerow([])
            w.writerow(['车型', '数量'])
            for name, cnt in self.current_stats.get('class_counts', {}).items():
                w.writerow([name, cnt])
        self.log(f'CSV: {path}')

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()


# ============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget { background-color: #2b2b2b; color: #ddd; }
        QMainWindow { background-color: #2b2b2b; }
        QGroupBox { color: white; font-weight: bold; border: 1px solid #555;
                    border-radius: 5px; margin-top: 10px; padding-top: 15px;
                    background-color: #333; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QPushButton { background: #3c3f41; color: white; border: 1px solid #555;
                      border-radius: 4px; padding: 8px; }
        QPushButton:hover { background: #4a4d50; }
        QPushButton:pressed { background: #2a2d30; }
        QPushButton:disabled { background: #2b2b2b; color: #666; }
        QSlider::groove:horizontal { border: 1px solid #555; height: 8px;
                                     background: #3c3f41; border-radius: 4px; }
        QSlider::handle:horizontal { background: #00aa00; border: 1px solid #555;
                                     width: 16px; margin: -4px 0; border-radius: 8px; }
        QSlider::sub-page:horizontal { background: #00aa00; border-radius: 4px; }
        QTextEdit { border: 1px solid #555; border-radius: 4px;
                    background: #1e1e1e; color: #ccc; }
        QScrollArea { border: none; background: #2b2b2b; }
        QProgressBar { border: 1px solid #555; border-radius: 3px;
                       background: #2b2b2b; color: white; }
        QProgressBar::chunk { background: #00aa00; }
    """)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
