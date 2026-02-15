# TV 转播运动员图像分割数据集调研

## 1. 研究目标

通过公开数据集微调 EdgeSAM 模型，实现在 TV 转播画面中对运动员的 instance segmentation。

微调策略：冻结 image encoder 和 prompt encoder，仅微调 mask decoder。

---

## 2. 数据集详细介绍

### 2.1 Football Player Segmentation (Kaggle)

| 项目 | 详情 |
|------|------|
| 运动类型 | 足球 |
| 图像来源 | TV 转播截图 |
| 图像数量 | 512 张 |
| 数据大小 | ~317 MB |
| 标注格式 | COCO JSON (`instances_default.json`) |
| Mask 编码 | Polygon（多边形顶点坐标 `[[x1,y1,x2,y2,...]]`） |
| 分割类型 | Instance Segmentation |
| 类别 | 球员（守门员、后卫、中场、前锋） |
| 数据划分 | 无预设 train/val/test split，需自行划分 |
| 许可证 | CC0 Public Domain |
| TV 适配度 | 高 |

**目录结构：**

```
football-player-segmentation/
├── images/
│   ├── image1.jpg
│   └── ...
└── annotations/
    └── instances_default.json
```

**链接：**
- Kaggle: https://www.kaggle.com/datasets/ihelon/football-player-segmentation
- HuggingFace: https://huggingface.co/datasets/Voxel51/Football-Player-Segmentation

---

### 2.2 DeepSportRadar Instance Segmentation

| 项目 | 详情 |
|------|------|
| 运动类型 | 篮球 |
| 图像来源 | 场馆高位摄像头（接近 TV 转播视角但更固定） |
| 图像数量 | 324 张有标注（总 1,456 张含未标注） |
| 标注实例数 | 2,495 个（trainvaltest） |
| 数据大小 | ~6 GB（含高分辨率原图和相机参数） |
| 标注格式 | COCO JSON |
| Mask 编码 | RLE（Run-Length Encoding, `{"size":[H,W], "counts":"..."}`) |
| 分割类型 | Instance Segmentation |
| 类别 | 1 个类：human（球员 + 教练 + 裁判） |
| 许可证 | CC BY-NC-SA 4.0（非商用） |
| TV 适配度 | 中高 |

**数据划分：**

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 223 | 1,674 |
| Val | 37 | 344 |
| Test | 64 | 477 |
| Challenge | 84 | 0 (无标注) |

**链接：**
- GitHub: https://github.com/DeepSportradar/instance-segmentation-challenge
- Dataset Ninja: https://datasetninja.com/deep-sport-radar
- Kaggle 下载: `kaggle datasets download deepsportradar/basketball-instants-dataset`

---

### 2.3 MADS (Martial Arts, Dancing and Sports)

| 项目 | 详情 |
|------|------|
| 运动类型 | 武术（太极、空手道）、舞蹈（街舞、爵士）、球类（篮球、排球、足球、橄榄球、网球、羽毛球） |
| 图像来源 | 深度传感器单视角（非 TV 画面） |
| 图像数量 | ~28,000 帧 |
| 标注格式 | 二值 bitmap 图像（非 JSON） |
| Mask 编码 | 每帧一张 mask PNG，像素 >0 为人体，0 为背景 |
| 分割类型 | Semantic Segmentation（人/背景二分类，非 instance） |
| 类别 | 前景（人）/ 背景 |
| TV 适配度 | 低 |

**数据划分（作者提供多种比例）：**

| Split | Training | Testing |
|-------|----------|---------|
| 50/50 | 14,414 | 14,414 |
| 60/40 | 17,047 | 11,492 |
| 70/30 | 19,141 | 8,616 |
| 80/20 | 21,473 | 5,742 |

**链接：**
- 论文: https://pmc.ncbi.nlm.nih.gov/articles/PMC8706170/
- 下载: https://drive.google.com/file/d/1Ssob496MJMUy3vAiXkC_ChKbp4gx7OGL/view?usp=sharing

---

### 2.4 SoccerNet v3

| 项目 | 详情 |
|------|------|
| 运动类型 | 足球 |
| 图像来源 | TV 转播视频 |
| 图像数量 | 33,986 张 |
| 标注内容 | 球员 bbox、球场线、球门部件、球衣号码、球员重识别等 |
| 标注格式 | 自定义 JSON + Bounding Box |
| Mask 标注 | **无像素级 mask** |
| 许可证 | 需签 NDA |
| TV 适配度 | 场景匹配但缺少 mask 标注，不适合分割任务 |

**链接：**
- 官网: https://www.soccer-net.org/data
- GitHub: https://github.com/SilvioGiancola/SoccerNet-code

---

## 3. 数据集对比

### 3.1 总览对比

| | Football Player Seg. | DeepSportRadar | MADS | SoccerNet v3 |
|---|---|---|---|---|
| TV 转播画面 | 是 | 接近 | 否 | 是 |
| 有像素级 Mask | 是 (Polygon) | 是 (RLE) | 是 (Bitmap) | 否 |
| Instance Seg | 是 | 是 | 否 | N/A |
| 标注格式 | COCO JSON | COCO JSON | 独立图像文件 | 自定义 JSON |
| 数据体量 | 317 MB / 512 张 | 6 GB / 324 张 | 大 / 28K 帧 | 很大 / 34K 张 |
| 适合 Benchmark | 核心数据 | 跨运动扩展 | 不推荐 | 不适合 |

### 3.2 标注格式对比

| | Football Player Seg. | DeepSportRadar | MADS |
|---|---|---|---|
| 标注文件格式 | 单个 COCO JSON | 单个 COCO JSON | 独立 PNG mask 图像 |
| Mask 编码方式 | Polygon 坐标数组 | RLE 字符串 | Bitmap 像素值 |
| 是否区分个体 | 是（每个球员独立标注） | 是（每个人独立标注） | 否（所有人合并为前景） |
| 与 EdgeSAM 兼容性 | 直接兼容 COCODataset | 兼容 COCODataset（需确认格式） | 不兼容，需大量改造 |

### 3.3 Mask 编码说明

**Polygon（Football Player Seg.）：**
```json
{
  "segmentation": [[x1, y1, x2, y2, x3, y3, ...]],
  "bbox": [x, y, w, h],
  "area": 12345
}
```
多边形顶点按顺序连接围成 mask 区域，直观且通用。

**RLE（DeepSportRadar）：**
```json
{
  "segmentation": {"size": [H, W], "counts": "encoded_string"},
  "bbox": [x, y, w, h],
  "area": 12345
}
```
行程编码压缩存储，空间效率高但不直观，需 pycocotools 解码。

**Bitmap（MADS）：**

每帧独立 PNG 图像，像素值 >0 表示人体区域。无 JSON 元数据，无 instance 区分。

---

## 4. 微调方案

### 4.1 推荐数据组合

**Football Player Seg. + DeepSportRadar**，合计约 770 张有标注图像。

排除 MADS 的理由：
- 深度传感器数据与 TV 转播画面 domain gap 过大
- 仅语义分割（无 instance 区分），与目标任务不匹配

排除 SoccerNet 的理由：
- 无像素级 mask 标注

### 4.2 数据量依据

EdgeSAM 原始训练数据量与性能：

| 数据量 | COCO mIoU |
|--------|-----------|
| 1% SA-1B (~11K 图) | 42.2 |
| 3% SA-1B (~33K 图) | 42.7 |
| 10% SA-1B (~110K 图) | 43.0 |

领域微调（仅调 mask decoder）所需数据远小于全量训练，500~1000 张高质量领域数据即可。

### 4.3 统一标注格式

采用 **COCO JSON + Polygon** 格式，复用 EdgeSAM 已有的 `COCODataset` 数据加载器（`training/data/coco_dataset.py`）。

DeepSportRadar 的 RLE mask 需通过 pycocotools 转换为 Polygon 格式。

统一类别：`1: athlete`。

### 4.4 目标目录结构

```
benchmark/
├── data/
│   └── athletes/
│       ├── trainval/                     # 所有图像
│       │   ├── football_001.jpg
│       │   ├── basketball_001.jpg
│       │   └── ...
│       └── annotations/
│           ├── instances_train2017.json   # 训练集标注
│           └── instances_val2017.json     # 验证集标注
├── scripts/
│   ├── convert_rle_to_polygon.py
│   └── merge_datasets.py
└── dataset_research.md
```

### 4.5 EdgeSAM 微调配置

冻结 encoder 和 prompt encoder，仅训练 mask decoder：

```yaml
DISTILL:
  FREEZE_IMAGE_ENCODER: True
  FREEZE_PROMPT_ENCODER: True
  FREEZE_MASK_DECODER: False
```

可选开启 LoRA（`DISTILL.LORA: True`）进一步减少可训练参数。

---

## 5. 参考链接

- [Football Player Segmentation (Kaggle)](https://www.kaggle.com/datasets/ihelon/football-player-segmentation)
- [Football Player Segmentation (HuggingFace)](https://huggingface.co/datasets/Voxel51/Football-Player-Segmentation)
- [DeepSportRadar GitHub](https://github.com/DeepSportradar/instance-segmentation-challenge)
- [DeepSportRadar Dataset Ninja](https://datasetninja.com/deep-sport-radar)
- [DeepSportRadar 论文](https://dl.acm.org/doi/10.1145/3552437.3555699)
- [MADS 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8706170/)
- [SoccerNet 官网](https://www.soccer-net.org/data)
- [MV-Soccer (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Majeed_MV-Soccer_Motion-Vector_Augmented_Instance_Segmentation_for_Soccer_Player_Tracking_CVPRW_2024_paper.pdf)
- [EdgeSAM 项目](https://github.com/chongzhou96/EdgeSAM)
- [SAM 使用 Football 数据集的博客](https://voxel51.com/blog/computer-vision-sam-for-prediction-kaggle-football-player-segmentation-dataset)
