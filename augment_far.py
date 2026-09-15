"""
远距离增强：缩小图片模拟远处小目标，所有类别都做
"""

import cv2
import numpy as np
from pathlib import Path

DATASET = Path('E:/dev/yjwlYOLO/dev/dataset')

# 不同类别用不同的缩放比例
# 汽车本来就大，缩少一点；摩托/行人/自行车本来就小，缩多一点
CLASS_SCALES = {
    0: [0.5, 0.6, 0.7],          # car
    1: [0.5, 0.6, 0.7],          # bus
    2: [0.5, 0.6, 0.7],          # van
    3: [0.5, 0.6, 0.7],          # truck
    4: [0.4, 0.5, 0.6],          # others
    5: [0.3, 0.4, 0.5, 0.6],     # bicycle (小目标，多缩几档)
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

        # 找所有有标注的图片
        all_files = []
        for img_path in img_dir.glob('*.jpg'):
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                all_files.append((img_path, lbl_path))

        print(f'{split}: {len(all_files)} 张图片，开始增强...')

        count = 0
        for img_path, lbl_path in all_files:
            # 读标注，确定包含哪些类别
            with open(lbl_path) as f:
                content = f.read()
            cls_ids_in_img = set()
            for line in content.strip().split('\n'):
                parts = line.split()
                if len(parts) == 5:
                    cls_ids_in_img.add(int(parts[0]))

            # 取该图片中所有类别的缩放比例的并集
            scales = set()
            for cid in cls_ids_in_img:
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
