"""
车流量检测系统 v1
功能: YOLO检测 + ByteTrack跟踪 + 车辆计数
输入: 视频帧图片序列
输出: 带标注的视频 + 车流量统计
"""

import cv2
import numpy as np
import os
import time
from ultralytics import YOLO


# ============ 配置参数 ============
MODEL_PATH = 'E:/dev/yjwlYOLO/dev/runs/train/weights/best.pt'
VIDEO_DIR = 'E:/dev/yjwlYOLO/数据集和权重/UA-DETRAC(车辆检测数据集8250车辆)/DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_39051'
OUTPUT_VIDEO = 'E:/dev/yjwlYOLO/dev/output_count.mp4'
DEVICE = 'cpu'  # 老师要看CPU推理速度

# 计数线位置 (Y坐标，画面高度的百分比)
COUNT_LINE_RATIO = 0.6  # 在画面60%高度处画线
CONF_THRESHOLD = 0.4    # 检测置信度阈值

# ============ 颜色定义 ============
COLORS = {
    'car': (0, 255, 0),     # 绿色
    'bus': (255, 0, 0),     # 蓝色
    'van': (0, 165, 255),   # 橙色
    'truck': (0, 0, 255),   # 红色
    'others': (128, 128, 128),  # 灰色
}


def load_frames(video_dir):
    """加载图片序列"""
    frames = []
    files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
    for f in files:
        img = cv2.imread(os.path.join(video_dir, f))
        if img is not None:
            frames.append(img)
    return frames


def draw_count_line(frame, y, count, fps_val):
    """画计数线和统计信息"""
    h, w = frame.shape[:2]
    # 画计数线
    cv2.line(frame, (0, y), (w, y), (0, 0, 255), 3)
    cv2.putText(frame, 'COUNT LINE', (10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 左上角统计面板
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, f'Vehicle Count: {count}', (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'FPS: {fps_val:.1f}', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Device: {DEVICE}', (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame


def draw_boxes(frame, boxes, track_ids, class_names):
    """画检测框和跟踪ID"""
    for box, track_id, cls_name in zip(boxes, track_ids, class_names):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(cls_name, (128, 128, 128))
        
        # 画框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 标签: 类名 + ID
        label = f'{cls_name} #{int(track_id)}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 画中心点
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, color, -1)
    
    return frame


def run():
    """主流程"""
    print("=" * 50)
    print("车流量检测系统 v1")
    print("=" * 50)
    
    # 1. 加载模型
    print(f"[1/4] 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 2. 加载视频帧
    print(f"[2/4] 加载视频帧: {VIDEO_DIR}")
    frames = load_frames(VIDEO_DIR)
    print(f"      共 {len(frames)} 帧, 分辨率 {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    h, w = frames[0].shape[:2]
    count_line_y = int(h * COUNT_LINE_RATIO)
    
    # 3. 初始化视频输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (w, h))
    
    # 4. 逐帧处理
    print(f"[3/4] 开始处理 (设备: {DEVICE})")
    print(f"      计数线位置: Y={count_line_y} (画面{COUNT_LINE_RATIO*100:.0f}%处)")
    
    counted_ids = set()     # 已计数的跟踪ID
    vehicle_count = 0       # 车辆计数
    id_last_y = {}          # 记录每个ID上一帧的Y坐标
    
    total_time = 0
    
    for idx, frame in enumerate(frames):
        start = time.time()
        
        # YOLO检测 + ByteTrack跟踪
        results = model.track(
            frame,
            persist=True,           # 跨帧跟踪
            tracker='bytetrack.yaml',
            device=DEVICE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        
        elapsed = time.time() - start
        total_time += elapsed
        
        # 解析跟踪结果
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = [model.names[c] for c in cls_ids]
            
            # 检查跨线
            for box, tid in zip(boxes, track_ids):
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                tid = int(tid)
                
                if tid in id_last_y:
                    prev_y = id_last_y[tid]
                    # 从上往下跨线 或 从下往上跨线 都算
                    if (prev_y < count_line_y <= cy) or (prev_y > count_line_y >= cy):
                        if tid not in counted_ids:
                            counted_ids.add(tid)
                            vehicle_count += 1
                
                id_last_y[tid] = cy
            
            # 画框
            frame = draw_boxes(frame, boxes, track_ids, class_names)
        
        # 画计数线和统计
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        frame = draw_count_line(frame, count_line_y, vehicle_count, current_fps)
        
        # 写入输出视频
        out.write(frame)
        
        # 进度
        if (idx + 1) % 100 == 0 or idx == 0:
            avg_fps = (idx + 1) / total_time
            print(f"      帧 {idx+1}/{len(frames)} | 当前FPS: {current_fps:.1f} | 平均FPS: {avg_fps:.1f} | 已计数: {vehicle_count}")
    
    # 5. 完成
    out.release()
    avg_fps = len(frames) / total_time
    
    print(f"[4/4] 处理完成!")
    print(f"      总帧数: {len(frames)}")
    print(f"      总耗时: {total_time:.1f}秒")
    print(f"      平均FPS: {avg_fps:.1f}")
    print(f"      检测到车辆数: {vehicle_count}")
    print(f"      输出视频: {OUTPUT_VIDEO}")
    print("=" * 50)
    
    return vehicle_count, avg_fps


if __name__ == '__main__':
    run()
