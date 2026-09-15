"""
UA-DETRAC 数据集 → YOLO 格式转换脚本
将 XML 标注转换为 YOLO txt 格式，并划分训练集/验证集
"""
import os
import xml.etree.ElementTree as ET
import shutil
import random

# ============ 配置路径 ============
BASE_DIR = r"E:\dev\yjwlYOLO\数据集和权重\UA-DETRAC(车辆检测数据集8250车辆)"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "DETRAC-train-data", "Insight-MVT_Annotation_Train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "DETRAC-test-data", "Insight-MVT_Annotation_Test")
XML_DIR = os.path.join(BASE_DIR, "DETRAC-Train-Annotations-XML", "DETRAC-Train-Annotations-XML")

# 输出目录
OUTPUT_DIR = r"E:\dev\yjwlYOLO\dev\dataset"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

# ============ 类别映射 ============
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "van": 2,
    "truck": 3,
    "others": 4,
}

# 图片尺寸 (UA-DETRAC 标准)
IMG_W = 960
IMG_H = 540

# 验证集比例
VAL_RATIO = 0.2
random.seed(42)


def convert_box_to_yolo(left, top, width, height):
    """像素坐标 → YOLO 归一化坐标"""
    cx = (left + width / 2) / IMG_W
    cy = (top + height / 2) / IMG_H
    w = width / IMG_W
    h = height / IMG_H
    return cx, cy, w, h


def process_xml(xml_path, img_dir, out_label_dir, out_img_dir):
    """处理单个 XML 文件，生成 YOLO 标签并复制图片"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seq_name = root.get("name")

    src_seq_dir = os.path.join(img_dir, seq_name)
    if not os.path.exists(src_seq_dir):
        print(f"  [跳过] 图片目录不存在: {seq_name}")
        return 0

    count = 0
    for frame in root.findall(".//frame"):
        frame_num = int(frame.get("num"))
        img_name = f"img{frame_num:05d}.jpg"
        img_path = os.path.join(src_seq_dir, img_name)

        if not os.path.exists(img_path):
            continue

        # 生成 YOLO 标签
        labels = []
        for target in frame.findall(".//target"):
            attr = target.find("attribute")
            box = target.find("box")
            if attr is None or box is None:
                continue

            vehicle_type = attr.get("vehicle_type")
            if vehicle_type not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[vehicle_type]
            left = float(box.get("left"))
            top = float(box.get("top"))
            w = float(box.get("width"))
            h = float(box.get("height"))

            cx, cy, nw, nh = convert_box_to_yolo(left, top, w, h)
            # 限制在 [0, 1] 范围内
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # 保存标签文件和图片
        new_name = f"{seq_name}_{img_name}"
        label_path = os.path.join(out_label_dir, new_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        dst_img_path = os.path.join(out_img_dir, new_name)
        shutil.copy2(img_path, dst_img_path)
        count += 1

    return count


def main():
    # 创建输出目录
    for d in [TRAIN_DIR, VAL_DIR]:
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        os.makedirs(os.path.join(d, "labels"), exist_ok=True)

    # 获取所有有标注的视频序列
    xml_files = sorted([f for f in os.listdir(XML_DIR) if f.endswith(".xml")])
    print(f"找到 {len(xml_files)} 个标注文件")

    # 划分训练集/验证集
    random.shuffle(xml_files)
    val_count = int(len(xml_files) * VAL_RATIO)
    val_xmls = xml_files[:val_count]
    train_xmls = xml_files[val_count:]

    print(f"训练集: {len(train_xmls)} 个视频, 验证集: {len(val_xmls)} 个视频")

    total_train = 0
    total_val = 0

    # 处理训练集
    print("\n处理训练集...")
    for xml_name in train_xmls:
        xml_path = os.path.join(XML_DIR, xml_name)
        n = process_xml(xml_path, TRAIN_IMG_DIR,
                        os.path.join(TRAIN_DIR, "labels"),
                        os.path.join(TRAIN_DIR, "images"))
        total_train += n
        print(f"  {xml_name}: {n} 帧")

    # 处理验证集
    print("\n处理验证集...")
    for xml_name in val_xmls:
        xml_path = os.path.join(XML_DIR, xml_name)
        n = process_xml(xml_path, TRAIN_IMG_DIR,
                        os.path.join(VAL_DIR, "labels"),
                        os.path.join(VAL_DIR, "images"))
        total_val += n
        print(f"  {xml_name}: {n} 帧")

    # 生成 data.yaml
    yaml_content = f"""# UA-DETRAC YOLO 数据集配置
path: {OUTPUT_DIR}
train: train/images
val: val/images

nc: {len(CLASS_MAP)}
names: ['car', 'bus', 'van', 'truck', 'others']
"""
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n{'='*50}")
    print(f"转换完成!")
    print(f"  训练集: {total_train} 帧")
    print(f"  验证集: {total_val} 帧")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  配置文件: {yaml_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
