#!/usr/bin/env python3
"""
将 Football Player Segmentation 数据集导入 Label Studio
"""
import json
import os
from pathlib import Path

# 数据集路径
DATASET_DIR = "/Users/wanghao/benchmark/football_player_segmentation"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNO_FILE = os.path.join(DATASET_DIR, "annotations", "instances_default.json")

# 输出 Label Studio 兼容的导入文件
OUTPUT_FILE = "/Users/wanghao/benchmark/labelstudio_import.json"

def convert_to_labelstudio_format():
    """转换 COCO JSON 为 Label Studio 导入格式"""

    with open(ANNO_FILE, 'r') as f:
        coco_data = json.load(f)

    # 创建图像ID到标注的映射
    image_id_to_annos = {}
    for anno in coco_data['annotations']:
        img_id = anno['image_id']
        if img_id not in image_id_to_annos:
            image_id_to_annos[img_id] = []
        image_id_to_annos[img_id].append(anno)

    # 创建 Label Studio 任务列表
    tasks = []
    for img in coco_data['images']:
        img_id = img['id']
        img_filename = img['file_name']
        img_path = os.path.join(IMAGES_DIR, img_filename)

        # Label Studio 任务格式
        task = {
            "data": {
                "image": f"/data/local-files/?d={img_path}"  # 本地文件路径
            },
            "predictions": []  # 预标注（如果有的话）
        }

        # 如果有标注，转换为 predictions（预标注，可以修改）
        if img_id in image_id_to_annos:
            predictions = {
                "model_version": "coco_import",
                "score": 1.0,
                "result": []
            }

            for anno in image_id_to_annos[img_id]:
                # 转换 polygon 为 Label Studio 格式
                if 'segmentation' in anno and isinstance(anno['segmentation'], list):
                    for seg in anno['segmentation']:
                        # 转换为百分比坐标
                        points = []
                        for i in range(0, len(seg), 2):
                            x_pct = (seg[i] / img['width']) * 100
                            y_pct = (seg[i+1] / img['height']) * 100
                            points.append([x_pct, y_pct])

                        predictions["result"].append({
                            "type": "polygonlabels",
                            "value": {
                                "points": points,
                                "polygonlabels": ["person"]
                            },
                            "from_name": "label",
                            "to_name": "image"
                        })

            if predictions["result"]:
                task["predictions"] = [predictions]

        tasks.append(task)

    # 保存为 Label Studio 导入格式
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"✅ 已生成 Label Studio 导入文件: {OUTPUT_FILE}")
    print(f"📊 共 {len(tasks)} 个任务")
    print(f"\n导入步骤:")
    print(f"1. 在 Label Studio 中创建新项目")
    print(f"2. 设置标注配置（选择 Image Segmentation 模板）")
    print(f"3. 点击 Import → Upload Files")
    print(f"4. 上传生成的 {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_to_labelstudio_format()
