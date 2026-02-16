# 自动标注 Pipeline 指南

使用 SAM3 + Label Studio 构建完整的运动员分割数据集

---

## Pipeline 概览

```
原始数据（bbox 标注）
       │
       ▼
  SAM3 自动生成 mask
       │
       ▼
  COCO JSON 输出
       │
       ▼
 Label Studio 可视化检查
       │
       ▼
  人工修正错误标注
       │
       ▼
  导出修正后的 COCO JSON
       │
       ▼
  用于 EdgeSAM 训练
```

---

## 环境准备

### 1. 安装依赖

```bash
# 核心依赖
pip install autodistill-sam3
pip install autodistill
pip install supervision
pip install label-studio
pip install pycocotools

# 设置 Roboflow API Key（用于下载 SAM3 模型）
export ROBOFLOW_API_KEY=你的API密钥
```

获取 API key: https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key

### 2. 验证安装

```bash
python3 -c "from autodistill_sam3 import SegmentAnything3; print('✅ SAM3 installed')"
label-studio --version
```

---

## 步骤 1: 准备原始数据

### 数据组织

将原始图像放入指定文件夹：

```
benchmark/
└── raw_data/
    ├── football/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── basketball/
    │   └── ...
    └── nfl/
        └── ...
```

### 支持的输入格式

1. **仅图像**（推荐用 SAM3 文本 prompt）
2. **图像 + bbox 标注**（用 bbox 作为 prompt，精度更高）
3. **已有部分 mask 标注**（可直接检查和修正）

---

## 步骤 2: SAM3 自动标注

### 方式 A：文本 Prompt（无需任何标注）

创建 `auto_annotate.py`：

```python
from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology

# 定义文本 prompt → 类别映射
base_model = SegmentAnything3(
    ontology=CaptionOntology({
        "football player": "athlete",
        "basketball player": "athlete",
        "player": "athlete"
    })
)

# 批量标注整个文件夹
base_model.label(
    input_folder="./raw_data/football",
    extension=".jpg",
    output_folder="./output/football_labeled"
)
```

运行：

```bash
python auto_annotate.py
```

**输出**：
- `output/football_labeled/train/` - 图像
- `output/football_labeled/valid/` - 图像
- `data.yaml` - 数据集配置
- COCO JSON 标注文件

### 方式 B：使用已有 bbox 标注

如果已有 COCO 格式的 bbox 标注（`annotations.json`）：

```python
from autodistill_sam3 import SegmentAnything3
from autodistill.helpers import load_image
import json
import supervision as sv

# 加载模型
model = SegmentAnything3(
    ontology=CaptionOntology({"player": "athlete"})
)

# 读取 bbox 标注
with open("annotations.json") as f:
    coco_data = json.load(f)

results = []

for img_info in coco_data['images']:
    img_path = f"images/{img_info['file_name']}"
    image = load_image(img_path, return_format="cv2")

    # 获取该图像的所有 bbox
    img_id = img_info['id']
    bboxes = [anno['bbox'] for anno in coco_data['annotations']
              if anno['image_id'] == img_id]

    # 用 bbox 作为 prompt 生成 mask（更精确）
    for bbox in bboxes:
        # bbox 格式转换 [x, y, w, h] -> [x1, y1, x2, y2]
        x, y, w, h = bbox
        box_prompt = [x, y, x+w, y+h]

        detections = model.predict_with_box(img_path, box_prompt)
        results.append(detections)

# 保存为 COCO JSON
# ... (使用 supervision 或 pycocotools 转换)
```

### 关键参数

```python
base_model.label(
    input_folder="./images",
    extension=".jpg",           # 图像格式
    output_folder="./output",
    confidence=0.5              # 置信度阈值（0-1）
)
```

**调优建议**：
- `confidence=0.3` → 检测更多对象（可能有误检）
- `confidence=0.7` → 更严格（可能漏检）
- 默认 `0.5` 是较好的平衡点

---

## 步骤 3: 转换为 Label Studio 格式

创建 `convert_to_labelstudio.py`：

```python
#!/usr/bin/env python3
import json
import os

IMAGES_DIR = "./output/football_labeled/train"
COCO_JSON = "./output/football_labeled/annotations.json"
OUTPUT_JSON = "./labelstudio_import.json"

def convert_coco_to_labelstudio(coco_file, images_dir, output_file):
    """转换 COCO JSON 为 Label Studio 导入格式"""

    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # 构建 image_id → annotations 映射
    anno_map = {}
    for anno in coco_data['annotations']:
        img_id = anno['image_id']
        if img_id not in anno_map:
            anno_map[img_id] = []
        anno_map[img_id].append(anno)

    # 构建 Label Studio 任务
    tasks = []
    for img in coco_data['images']:
        img_id = img['id']
        img_file = img['file_name']
        img_path = os.path.join(images_dir, img_file)

        task = {
            "data": {
                "image": f"/data/local-files/?d={img_path}"
            }
        }

        # 添加预标注（可编辑的标注）
        if img_id in anno_map:
            predictions = {
                "model_version": "sam3_auto",
                "result": []
            }

            for anno in anno_map[img_id]:
                # 转换 polygon 为 Label Studio 格式
                if 'segmentation' in anno:
                    seg = anno['segmentation']

                    # 支持 polygon 和 RLE 两种格式
                    if isinstance(seg, list):  # Polygon
                        for poly in seg:
                            points = []
                            for i in range(0, len(poly), 2):
                                # 转为百分比坐标
                                x_pct = (poly[i] / img['width']) * 100
                                y_pct = (poly[i+1] / img['height']) * 100
                                points.append([x_pct, y_pct])

                            predictions["result"].append({
                                "type": "polygonlabels",
                                "value": {
                                    "points": points,
                                    "polygonlabels": ["athlete"]
                                },
                                "from_name": "label",
                                "to_name": "image"
                            })

                    elif isinstance(seg, dict):  # RLE
                        # RLE 需要先解码为 polygon
                        from pycocotools import mask as mask_utils
                        import numpy as np
                        import cv2

                        rle = seg
                        mask = mask_utils.decode(rle)
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        for contour in contours:
                            points = []
                            for point in contour.squeeze():
                                x_pct = (point[0] / img['width']) * 100
                                y_pct = (point[1] / img['height']) * 100
                                points.append([x_pct, y_pct])

                            predictions["result"].append({
                                "type": "polygonlabels",
                                "value": {
                                    "points": points,
                                    "polygonlabels": ["athlete"]
                                },
                                "from_name": "label",
                                "to_name": "image"
                            })

            if predictions["result"]:
                task["predictions"] = [predictions]

        tasks.append(task)

    # 保存
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"✅ 转换完成: {len(tasks)} 个任务")
    print(f"📁 输出文件: {output_file}")

    return output_file

if __name__ == "__main__":
    convert_coco_to_labelstudio(COCO_JSON, IMAGES_DIR, OUTPUT_JSON)
```

运行：

```bash
python convert_to_labelstudio.py
```

---

## 步骤 4: 导入 Label Studio

### 4.1 启动 Label Studio

```bash
label-studio start --port 8080
```

访问 http://localhost:8080

### 4.2 创建项目

1. 点击 **"Create Project"**
2. 输入项目名称：`Athlete Segmentation`
3. 选择 **"Computer Vision > Image Segmentation"** 模板
4. 点击 **"Save"**

### 4.3 配置标注界面

在 **Labeling Interface** 中使用以下配置：

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2">
    <Label value="athlete" background="red"/>
  </PolygonLabels>
</View>
```

支持的标注类型：
- **PolygonLabels** - 多边形标注（推荐）
- **BrushLabels** - 画笔涂抹
- **RectangleLabels** - 矩形框

### 4.4 导入数据

#### 方式 1：上传 JSON 文件

1. 点击 **"Import"**
2. 选择 **"Upload Files"**
3. 上传生成的 `labelstudio_import.json`

#### 方式 2：批量上传图像

1. 点击 **"Import"**
2. 拖拽整个图像文件夹到浏览器
3. 系统会批量上传所有图片

#### 方式 3：本地存储（推荐，大数据集）

1. **Settings** → **Cloud Storage** → **Add Source Storage**
2. 选择 **"Local files"**
3. 路径：`/Users/wanghao/benchmark/output/football_labeled/train`
4. 点击 **"Sync Storage"**

---

## 步骤 5: 人工检查和修正

### 检查清单

| 问题类型 | 检查内容 | 修正方式 |
|---------|---------|---------|
| **漏检** | 有运动员未标注 | 用 Polygon 工具手动补标 |
| **误检** | 裁判/观众被标为球员 | 删除错误标注 |
| **边界不准** | Mask 边缘模糊/溢出 | 调整 polygon 顶点 |
| **遮挡分离** | 重叠球员未分开 | 拆分为独立 polygon |
| **类别错误** | 错误的标签 | 修改 label |

### 快速操作快捷键

- **删除标注**：选中 polygon 后按 `Delete` 或 `Backspace`
- **添加顶点**：双击 polygon 边缘
- **移动顶点**：拖动顶点
- **完成标注**：按 `Enter` 或点击起点
- **下一张图**：`Ctrl/Cmd + →`
- **上一张图**：`Ctrl/Cmd + ←`
- **提交标注**：`Ctrl/Cmd + Enter`

### 抽样检查策略

**如果数据量大（500+ 张）**：

1. **分层抽样**：
   - 简单场景（无遮挡）：抽 10%
   - 中等场景（部分遮挡）：抽 30%
   - 困难场景（密集/严重遮挡）：抽 50%

2. **质量判断标准**：
   - 90% 以上正确 → 直接使用
   - 70-90% 正确 → 修正明显错误
   - <70% 正确 → 调整 SAM3 参数重新标注

---

## 步骤 6: 导出修正后的标注

### 6.1 导出操作

1. 在项目页面点击 **"Export"**
2. 选择导出格式：**"COCO"**
3. 下载 `project-1-at-2024-xx-xx.json`

### 6.2 验证导出格式

```bash
python -c "
import json
with open('project-1-at-2024-xx-xx.json') as f:
    data = json.load(f)
    print(f'Images: {len(data[\"images\"])}')
    print(f'Annotations: {len(data[\"annotations\"])}')
    print(f'Categories: {data[\"categories\"]}')
"
```

### 6.3 格式转换（如果需要）

Label Studio 导出的 COCO JSON 可能需要小幅调整以完全兼容训练框架：

```python
import json

def clean_labelstudio_export(input_file, output_file):
    """清理 Label Studio 导出的 COCO JSON"""

    with open(input_file, 'r') as f:
        data = json.load(f)

    # 确保有必需字段
    if 'info' not in data:
        data['info'] = {
            "description": "Athlete Segmentation Dataset",
            "version": "1.0",
            "year": 2026
        }

    if 'licenses' not in data:
        data['licenses'] = [{"id": 1, "name": "Unknown", "url": ""}]

    # 清理 annotations
    for anno in data['annotations']:
        # 确保有 area 字段
        if 'area' not in anno:
            if 'segmentation' in anno and isinstance(anno['segmentation'], list):
                from shapely.geometry import Polygon
                poly = Polygon(
                    [(anno['segmentation'][0][i], anno['segmentation'][0][i+1])
                     for i in range(0, len(anno['segmentation'][0]), 2)]
                )
                anno['area'] = poly.area

        # 确保有 iscrowd 字段
        if 'iscrowd' not in anno:
            anno['iscrowd'] = 0

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ 已清理并保存到: {output_file}")

if __name__ == "__main__":
    clean_labelstudio_export(
        "project-1-export.json",
        "cleaned_annotations.json"
    )
```

---

## 步骤 7: 合并多个数据集

如果你标注了多个运动（足球、篮球、NFL）：

```python
import json

def merge_coco_datasets(dataset_files, output_file, unified_category="athlete"):
    """合并多个 COCO JSON 数据集"""

    merged = {
        "info": {"description": "Multi-sport athlete segmentation", "version": "1.0"},
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [{"id": 1, "name": unified_category, "supercategory": "person"}],
        "images": [],
        "annotations": []
    }

    img_id_offset = 0
    anno_id_offset = 0

    for dataset_file in dataset_files:
        with open(dataset_file, 'r') as f:
            data = json.load(f)

        # 重新映射 image IDs
        old_to_new_img_id = {}
        for img in data['images']:
            old_id = img['id']
            new_id = img_id_offset + old_id
            old_to_new_img_id[old_id] = new_id

            img['id'] = new_id
            merged['images'].append(img)

        img_id_offset += len(data['images'])

        # 重新映射 annotation IDs 和统一 category
        for anno in data['annotations']:
            anno['id'] = anno_id_offset + anno['id']
            anno['image_id'] = old_to_new_img_id[anno['image_id']]
            anno['category_id'] = 1  # 统一为 athlete
            merged['annotations'].append(anno)

        anno_id_offset += len(data['annotations'])

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"✅ 合并完成:")
    print(f"   - 图像数: {len(merged['images'])}")
    print(f"   - 标注数: {len(merged['annotations'])}")
    print(f"   - 输出: {output_file}")

if __name__ == "__main__":
    merge_coco_datasets(
        [
            "football_cleaned.json",
            "basketball_cleaned.json",
            "nfl_cleaned.json"
        ],
        "merged_athlete_dataset.json"
    )
```

---

## 步骤 8: 划分训练/验证集

```python
import json
import random

def split_dataset(coco_file, train_ratio=0.8, seed=42):
    """划分 COCO 数据集为 train/val"""

    random.seed(seed)

    with open(coco_file, 'r') as f:
        data = json.load(f)

    # 随机打乱图像
    images = data['images']
    random.shuffle(images)

    # 划分
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}

    # 分离标注
    train_annos = [a for a in data['annotations'] if a['image_id'] in train_img_ids]
    val_annos = [a for a in data['annotations'] if a['image_id'] in val_img_ids]

    # 创建 train.json
    train_data = {
        "info": data['info'],
        "licenses": data['licenses'],
        "categories": data['categories'],
        "images": train_images,
        "annotations": train_annos
    }

    # 创建 val.json
    val_data = {
        "info": data['info'],
        "licenses": data['licenses'],
        "categories": data['categories'],
        "images": val_images,
        "annotations": val_annos
    }

    with open('train.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    with open('val.json', 'w') as f:
        json.dump(val_data, f, indent=2)

    print(f"✅ 划分完成:")
    print(f"   Train: {len(train_images)} 图像, {len(train_annos)} 标注")
    print(f"   Val:   {len(val_images)} 图像, {len(val_annos)} 标注")

if __name__ == "__main__":
    split_dataset("merged_athlete_dataset.json", train_ratio=0.8)
```

---

## 最终数据集结构

```
benchmark/
├── data/
│   └── athletes/
│       ├── train/
│       │   ├── football_001.jpg
│       │   ├── basketball_001.jpg
│       │   ├── nfl_001.jpg
│       │   └── ...
│       ├── val/
│       │   └── ...
│       └── annotations/
│           ├── train.json          # COCO format
│           └── val.json            # COCO format
├── scripts/
│   ├── auto_annotate.py
│   ├── convert_to_labelstudio.py
│   ├── merge_datasets.py
│   └── split_dataset.py
└── AUTO_ANNOTATION_PIPELINE.md
```

---

## 质量评估

在训练前评估数据集质量：

```python
import json
import numpy as np

def analyze_dataset(coco_file):
    """分析数据集统计信息"""

    with open(coco_file, 'r') as f:
        data = json.load(f)

    print("📊 数据集统计:")
    print(f"   - 图像数: {len(data['images'])}")
    print(f"   - 标注数: {len(data['annotations'])}")
    print(f"   - 类别数: {len(data['categories'])}")

    # 每张图平均标注数
    from collections import Counter
    img_anno_count = Counter(a['image_id'] for a in data['annotations'])
    avg_annos = np.mean(list(img_anno_count.values()))
    print(f"   - 平均标注/图: {avg_annos:.1f}")

    # Mask 面积分布
    areas = [a['area'] for a in data['annotations']]
    print(f"   - Mask 面积: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.0f}")

    # 检查缺失字段
    required_fields = ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd']
    for anno in data['annotations']:
        missing = [f for f in required_fields if f not in anno]
        if missing:
            print(f"⚠️  标注 {anno['id']} 缺少字段: {missing}")
            break
    else:
        print("✅ 所有标注字段完整")

if __name__ == "__main__":
    analyze_dataset("train.json")
    analyze_dataset("val.json")
```

---

## 常见问题

### Q1: SAM3 漏检了很多目标怎么办？

**A**: 降低 confidence 阈值，或者换用更明确的 text prompt：
```python
base_model.label(..., confidence=0.3)
# 或
ontology=CaptionOntology({
    "person in sports uniform": "athlete",
    "player wearing jersey": "athlete"
})
```

### Q2: Label Studio 导入后看不到预标注？

**A**: 确保：
1. JSON 格式正确（`predictions` 字段）
2. 坐标是百分比格式（0-100）
3. 标注配置中的 `from_name` 和 `to_name` 匹配

### Q3: 导出的 COCO JSON 格式不对？

**A**: 使用步骤 6.3 的清理脚本，确保所有必需字段存在。

### Q4: 如何处理 RLE 格式的 mask？

**A**: 使用 `pycocotools` 转换：
```python
from pycocotools import mask as mask_utils
binary_mask = mask_utils.decode(rle)
polygon = mask_utils.frPyObjects(segmentation, h, w)
```

---

## 性能优化建议

### 大数据集处理

1. **分批处理**：每次标注 100-200 张
2. **使用本地存储**：避免上传大量图片
3. **GPU 加速**：SAM3 在 GPU 上快 10-100 倍

### 标注效率

1. **快捷键熟练使用**：可提升 50% 效率
2. **置信度过滤**：只修正低置信度标注
3. **协作标注**：Label Studio 支持多人同时标注

---

## 下一步：用于训练

生成的 `train.json` 和 `val.json` 可直接用于：

- **EdgeSAM 微调**：参考 `training/data/coco_dataset.py`
- **Detectron2**：使用 `register_coco_instances`
- **MMDetection**：配置 `COCODataset`
- **YOLO**：转换为 YOLO 格式

EdgeSAM 微调示例配置：

```yaml
DATA:
  DATASET: coco
  DATA_PATH: benchmark/data/athletes/
  BATCH_SIZE: 4

DISTILL:
  FREEZE_IMAGE_ENCODER: True
  FREEZE_PROMPT_ENCODER: True
  FREEZE_MASK_DECODER: False

TRAIN:
  EPOCHS: 10
  BASE_LR: 3.2e-3
```

---

## 参考资源

- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [Autodistill 文档](https://docs.autodistill.com/)
- [Label Studio 文档](https://labelstud.io/guide/)
- [COCO 格式规范](https://cocodataset.org/#format-data)
- [EdgeSAM 训练指南](https://github.com/chongzhou96/EdgeSAM#training)
