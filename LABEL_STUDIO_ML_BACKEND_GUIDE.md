# Label Studio ML Backend 完整指南

将机器学习模型集成到 Label Studio 实现自动标注、主动学习和交互式标注

---

## 目录

1. [概述](#1-概述)
2. [快速开始](#2-快速开始)
3. [核心架构](#3-核心架构)
4. [预构建示例](#4-预构建示例)
5. [创建自定义 Backend](#5-创建自定义-backend)
6. [SAM 集成详解](#6-sam-集成详解)
7. [部署指南](#7-部署指南)
8. [最佳实践](#8-最佳实践)

---

## 1. 概述

### 什么是 ML Backend?

**Label Studio ML Backend** 是一个 SDK，让你能够：

✅ **自动标注**：用 AI 模型预标注数据
✅ **交互式标注**：实时响应用户交互（如 SAM 的点击分割）
✅ **主动学习**：训练模型并推荐最需要标注的样本
✅ **批量处理**：高效处理大规模数据集

### 工作原理

```
┌──────────────────┐
│  Label Studio    │
│  (前端界面)       │
└────────┬─────────┘
         │ HTTP REST API
         ↓
┌──────────────────┐
│   Flask Server   │
│  (api.py)        │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Your ML Model   │
│  (继承 Base 类)   │
└──────────────────┘
```

### 仓库结构

```
label-studio-ml-backend/
├── label_studio_ml/              # 核心 SDK
│   ├── api.py                    # REST API 端点
│   ├── model.py                  # LabelStudioMLBase 基类
│   ├── server.py                 # CLI 工具
│   ├── response.py               # 响应格式
│   ├── cache.py                  # 数据持久化
│   └── examples/                 # 25+ 预构建示例
│       ├── segment_anything_2_image/      # SAM 2
│       ├── grounding_sam/                 # Grounding SAM
│       ├── grounding_dino/                # Grounding DINO
│       ├── yolo/                          # YOLOv8+
│       ├── bert_classifier/               # 文本分类
│       └── ...
├── tests/
├── setup.py
└── README.md
```

---

## 2. 快速开始

### 方式 A：使用预构建示例（推荐）

以 **Grounding SAM** 为例（文本 prompt 自动分割）：

#### Step 1: 配置环境

```bash
cd label-studio-ml-backend/label_studio_ml/examples/grounding_sam

# 编辑 docker-compose.yml
vim docker-compose.yml
```

设置：
```yaml
environment:
  - LABEL_STUDIO_HOST=http://host.docker.internal:8080
  - LABEL_STUDIO_API_KEY=你的API密钥  # 从 Label Studio → Account & Settings 获取
  - USE_SAM=true                      # 启用 SAM
```

#### Step 2: 启动 Backend

```bash
docker compose up
```

输出：
```
✓ Started ML backend at http://localhost:9090
✓ Health check: http://localhost:9090/health
```

#### Step 3: 连接到 Label Studio

1. 打开 Label Studio 项目
2. **Settings** → **Machine Learning** → **Add Model**
3. 输入 URL：`http://localhost:9090`
4. 点击 **Validate and Save**

#### Step 4: 配置标注界面

在 **Labeling Interface** 中使用：

```xml
<View>
  <Image name="image" value="$image"/>

  <!-- 文本输入框 -->
  <TextArea name="prompt" toName="image" editable="true"
            rows="2" showSubmitButton="true"/>

  <!-- 输出：Brush Labels (Mask) -->
  <BrushLabels name="label" toName="image">
    <Label value="athlete" background="red"/>
  </BrushLabels>
</View>
```

#### Step 5: 使用

1. 打开标注任务
2. 在文本框输入：`football player`
3. 点击 **Add** 或按 **Enter**
4. ✅ 自动生成所有球员的 mask
5. 手动调整边界 → **Submit**

---

### 方式 B：从头创建自定义 Backend

```bash
# 生成模板
label-studio-ml create my_custom_backend

# 目录结构
my_custom_backend/
├── model.py              # 编辑此文件
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 3. 核心架构

### 3.1 Base 类：LabelStudioMLBase

所有模型继承自 `LabelStudioMLBase`：

```python
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

class MyModel(LabelStudioMLBase):

    def setup(self):
        """初始化：加载模型、设置版本"""
        self.model = load_your_model()
        self.set("model_version", "1.0.0")

    def predict(self, tasks, context=None, **kwargs):
        """核心方法：生成预测"""
        predictions = []
        for task in tasks:
            # 处理任务
            result = self.model.predict(task['data']['image'])
            predictions.append({
                'result': [...],
                'score': 0.95
            })
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """可选：训练/微调模型"""
        if event == 'ANNOTATION_CREATED':
            # 收集新标注，更新模型
            self.model.train(data)
```

### 3.2 API 端点

ML Backend 暴露 4 个主要端点：

| 端点 | 方法 | 用途 | 调用时机 |
|------|------|------|---------|
| `/predict` | POST | 生成预测 | 用户请求自动标注时 |
| `/setup` | POST | 初始化模型 | 首次连接或配置更改 |
| `/webhook` | POST | 处理事件 | 标注创建/更新时 |
| `/health` | GET | 健康检查 | Label Studio 定期检查 |

### 3.3 数据流

#### Predict 流程

```
Label Studio
    ↓ POST /predict
{
    'tasks': [
        {
            'id': 1,
            'data': {
                'image': 'http://localhost:8080/data/upload/1.jpg'
            }
        }
    ],
    'context': {  # 可选：交互式输入
        'result': [...]
    }
}
    ↓
Your Model.predict()
    ↓ ModelResponse
{
    'predictions': [
        {
            'result': [
                {
                    'from_name': 'label',
                    'to_name': 'image',
                    'type': 'brushlabels',
                    'value': {
                        'format': 'rle',
                        'rle': [...]
                    }
                }
            ],
            'score': 0.95,
            'model_version': '1.0.0'
        }
    ]
}
    ↓
Label Studio 显示预测
```

### 3.4 标注类型与输出格式

#### Bounding Box (RectangleLabels)

```python
{
    'from_name': 'label',
    'to_name': 'image',
    'type': 'rectanglelabels',
    'value': {
        'x': 10.5,        # % of width
        'y': 20.3,        # % of height
        'width': 30.2,    # % of width
        'height': 40.1,   # % of height
        'rectanglelabels': ['athlete']
    },
    'score': 0.95
}
```

#### Polygon (PolygonLabels)

```python
{
    'type': 'polygonlabels',
    'value': {
        'points': [
            [10.5, 20.3],  # % coordinates
            [30.2, 40.1],
            [50.6, 60.8]
        ],
        'polygonlabels': ['athlete']
    },
    'score': 0.95
}
```

#### Brush/Mask (BrushLabels)

```python
from label_studio_sdk.converter import brush

# 将 numpy mask 转为 RLE
rle = brush.mask2rle(np_mask)

{
    'type': 'brushlabels',
    'value': {
        'format': 'rle',
        'rle': rle,  # Run-Length Encoding
        'brushlabels': ['athlete']
    },
    'score': 0.95
}
```

---

## 4. 预构建示例

### 4.1 SAM 2 Image（交互式分割）

**用途**：点击/框选 → 实时生成 mask

**模型**：Meta SAM 2.1

**输入**：
- KeyPointLabels（正/负点击）
- RectangleLabels（框选）

**输出**：BrushLabels（RLE mask）

**配置**：

```yaml
# docker-compose.yml
environment:
  - DEVICE=cuda
  - MODEL_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
  - MODEL_CHECKPOINT=sam2.1_hiera_large.pt
```

**标注界面**：

```xml
<View>
  <Image name="image" value="$image"/>

  <!-- 输入：点击 -->
  <KeyPointLabels name="kp" toName="image">
    <Label value="positive" background="green"/>
    <Label value="negative" background="red"/>
  </KeyPointLabels>

  <!-- 输入：框选 -->
  <RectangleLabels name="rect" toName="image">
    <Label value="box"/>
  </RectangleLabels>

  <!-- 输出：Mask -->
  <BrushLabels name="label" toName="image">
    <Label value="athlete" background="red"/>
  </BrushLabels>
</View>
```

**工作流**：
1. 用户点击对象 → SAM 自动生成 mask
2. 用户添加负点击 → 精细化边界
3. Submit 保存最终 mask

---

### 4.2 Grounding SAM（文本 Prompt）

**用途**：输入文字 → 自动检测并分割

**模型**：Grounding DINO + SAM

**输入**：TextArea（文本描述）

**输出**：BrushLabels（mask）

**特色**：
- ✅ 零样本检测（无需训练）
- ✅ 自然语言 prompt
- ✅ 批量处理

**环境变量**：

```bash
USE_SAM=true                      # 启用 SAM
USE_MOBILE_SAM=false              # 使用完整 SAM（更准确）
BOX_THRESHOLD=0.3                 # 检测阈值
TEXT_THRESHOLD=0.25               # 文本匹配阈值
```

**示例 Prompt**：
```
"football player"
"goalkeeper wearing gloves"
"person in red jersey"
```

---

### 4.3 YOLO（多任务）

**用途**：检测、分割、分类、追踪

**支持任务**：

| 任务 | Label Studio 控件 | 输出 |
|------|-------------------|------|
| 目标检测 | RectangleLabels | Bounding boxes |
| 实例分割 | PolygonLabels | Polygon vertices |
| 图像分类 | Choices | 类别标签 |
| 姿态估计 | KeyPointLabels | 关节点坐标 |
| 视频追踪 | VideoRectangle | 跨帧追踪框 |

**配置**：

```xml
<RectangleLabels
    name="label"
    toName="image"
    model_path="yolov8m.pt"              <!-- 自定义模型 -->
    model_score_threshold="0.3"           <!-- 置信度阈值 -->
>
    <Label value="athlete"
           predicted_values="person,player"/>  <!-- 类别映射 -->
</RectangleLabels>
```

**自定义模型**：

```bash
# 挂载自定义权重
docker-compose.yml:
  volumes:
    - ./my_model.pt:/app/models/my_model.pt

# 在标注配置中引用
model_path="/app/models/my_model.pt"
```

---

### 4.4 其他有用示例

| 示例 | 用途 | 技术栈 |
|------|------|--------|
| `grounding_dino` | 文本 → 检测框 | Grounding DINO |
| `bert_classifier` | 文本分类 | Hugging Face BERT |
| `huggingface_ner` | 命名实体识别 | Transformers |
| `easyocr` | 光学字符识别 | EasyOCR |
| `mmdetection-3` | 高级目标检测 | OpenMMLab |
| `segment_anything_2_video` | 视频分割 | SAM 2 Video |

---

## 5. 创建自定义 Backend

### 5.1 生成模板

```bash
label-studio-ml create athlete_segmentation
cd athlete_segmentation
```

### 5.2 实现核心方法

编辑 `model.py`：

```python
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
import numpy as np

class AthleteSegmentation(LabelStudioMLBase):

    def setup(self):
        """初始化模型"""
        # 加载你的模型
        from your_model import load_model
        self.model = load_model('path/to/weights.pth')

        # 设置版本
        self.set("model_version", "1.0.0")

        # 缓存配置
        self.set("confidence_threshold", 0.5)

    def predict(self, tasks, context=None, **kwargs):
        """生成预测"""
        # Step 1: 解析标注配置
        from_name, to_name, value = self.get_first_tag_occurence(
            control_type='BrushLabels',
            object_type='Image'
        )

        if not from_name:
            return ModelResponse(predictions=[])

        # Step 2: 处理每个任务
        predictions = []
        for task in tasks:
            # 获取图像 URL
            image_url = task['data'][value]

            # 下载并缓存图像
            image_path = self.get_local_path(
                url=image_url,
                task_id=task.get('id')
            )

            # 加载图像
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            # Step 3: 运行模型推理
            masks, scores = self.model.predict(image)

            # Step 4: 格式化输出
            results = []
            for mask, score in zip(masks, scores):
                if score < self.get("confidence_threshold"):
                    continue

                # 转换 mask 为 RLE
                rle = brush.mask2rle(mask.astype(np.uint8))

                results.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'brushlabels',
                    'value': {
                        'format': 'rle',
                        'rle': rle,
                        'brushlabels': ['athlete']
                    },
                    'score': float(score),
                    'original_width': width,
                    'original_height': height,
                    'image_rotation': 0
                })

            predictions.append({
                'result': results,
                'score': float(np.mean([r['score'] for r in results])) if results else 0.0,
                'model_version': self.get('model_version')
            })

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """可选：训练模型"""
        if event == 'ANNOTATION_CREATED':
            # 收集新标注
            annotations = data.get('annotation', {}).get('result', [])

            # 提取训练数据
            # ...

            # 更新模型
            # self.model.fine_tune(training_data)

            # 更新版本
            self.bump_model_version()
```

### 5.3 配置 Docker

编辑 `requirements.txt`：

```txt
# Base requirements
label-studio-ml>=1.0.9

# Your model dependencies
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
```

编辑 `Dockerfile`：

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    git wget libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model code
COPY . /app

# Download model weights (optional)
RUN python -c "from your_model import download_weights; download_weights()"

CMD ["python", "_wsgi.py"]
```

### 5.4 测试

```bash
# 启动
docker compose up

# 健康检查
curl http://localhost:9090/health

# 测试预测（需要先连接到 Label Studio）
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "id": 1,
      "data": {"image": "https://example.com/image.jpg"}
    }]
  }'
```

---

## 6. SAM 集成详解

### 6.1 SAM 2 集成架构

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Backend(LabelStudioMLBase):
    def setup(self):
        # 构建 SAM 2 模型
        self.sam2_model = build_sam2(
            config_file='configs/sam2.1/sam2.1_hiera_l.yaml',
            ckpt_path='checkpoints/sam2.1_hiera_large.pt',
            device='cuda'
        )
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def predict(self, tasks, context=None, **kwargs):
        # 检查是否有交互输入
        if not context or not context.get('result'):
            return ModelResponse(predictions=[])

        # 解析用户输入
        point_coords = []
        point_labels = []
        input_box = None

        for ctx in context['result']:
            if ctx['type'] == 'keypointlabels':
                # 提取点击坐标
                x = ctx['value']['x'] * image_width / 100
                y = ctx['value']['y'] * image_height / 100
                point_coords.append([x, y])

                # 正/负点标签
                is_positive = ctx.get('is_positive', 0)
                point_labels.append(1 if is_positive else 0)

            elif ctx['type'] == 'rectanglelabels':
                # 提取 bbox
                x = ctx['value']['x'] * image_width / 100
                y = ctx['value']['y'] * image_height / 100
                w = ctx['value']['width'] * image_width / 100
                h = ctx['value']['height'] * image_height / 100
                input_box = np.array([x, y, x+w, y+h])

        # 设置图像
        self.predictor.set_image(image_array)

        # 运行预测
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(point_coords) if point_coords else None,
            point_labels=np.array(point_labels) if point_labels else None,
            box=input_box,
            multimask_output=True  # 返回 3 个候选 mask
        )

        # 选择最佳 mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]

        # 转换为 RLE
        rle = brush.mask2rle(best_mask.astype(np.uint8))

        return ModelResponse(predictions=[...])
```

### 6.2 Grounding SAM 工作流

```
用户输入文本 "football player"
         ↓
    Grounding DINO
    检测所有匹配对象
         ↓
    返回 bounding boxes
         ↓
    传递给 SAM
         ↓
    Box → Mask 转换
         ↓
    返回所有 masks
```

**代码示例**：

```python
from groundingdino.util.inference import load_model, predict

def predict_with_text_prompt(self, image, text_prompt):
    # Step 1: 文本 → 检测框
    boxes, logits, phrases = predict(
        model=self.groundingdino_model,
        image=image,
        caption=text_prompt,
        box_threshold=0.3,
        text_threshold=0.25
    )

    # Step 2: 检测框 → Masks
    if self.use_sam:
        self.sam_predictor.set_image(image)

        # 转换坐标格式
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes, image.shape[:2]
        ).to('cuda')

        # 批量预测
        masks, iou_predictions, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )

        return masks, logits
    else:
        return boxes, logits
```

### 6.3 添加 SAM 3 支持（未来）

当 SAM 3 发布后，修改步骤：

1. **更新 checkpoint 下载**：
```python
# download_ckpts.sh
wget https://example.com/sam3_checkpoint.pth
```

2. **更新导入**：
```python
from sam3.build_sam import build_sam3
from sam3.sam3_predictor import SAM3Predictor
```

3. **更新 Docker 环境变量**：
```yaml
MODEL_VERSION=sam3
MODEL_CHECKPOINT=sam3_checkpoint.pth
```

---

## 7. 部署指南

### 7.1 本地开发

```bash
# 克隆示例
cd label_studio_ml/examples/grounding_sam

# 编辑配置
vim docker-compose.yml

# 启动
docker compose up

# 查看日志
docker compose logs -f
```

### 7.2 生产部署（Docker）

**docker-compose.yml**：

```yaml
version: "3.8"
services:
  ml-backend:
    container_name: athlete-segmentation
    build:
      context: .
      dockerfile: Dockerfile

    environment:
      # Label Studio 连接
      - LABEL_STUDIO_HOST=${LABEL_STUDIO_HOST}
      - LABEL_STUDIO_API_KEY=${LABEL_STUDIO_API_KEY}

      # 服务器配置
      - LOG_LEVEL=INFO
      - WORKERS=4
      - THREADS=8
      - PORT=9090

      # GPU 配置
      - DEVICE=cuda
      - NVIDIA_VISIBLE_DEVICES=all

    # GPU 支持
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    ports:
      - "9090:9090"

    volumes:
      - ./data:/data
      - ./models:/app/models

    restart: unless-stopped

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**环境变量文件（.env）**：

```bash
LABEL_STUDIO_HOST=http://your-labelstudio.com
LABEL_STUDIO_API_KEY=your_api_key_here
```

### 7.3 云部署（GCP Cloud Run）

```bash
# 使用内置部署命令
label-studio-ml deploy \
  --project-id your-gcp-project \
  --region us-central1 \
  --model-dir ./athlete_segmentation
```

### 7.4 Kubernetes 部署

**deployment.yaml**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-backend
  template:
    metadata:
      labels:
        app: ml-backend
    spec:
      containers:
      - name: ml-backend
        image: your-registry/ml-backend:latest
        ports:
        - containerPort: 9090
        env:
        - name: LABEL_STUDIO_HOST
          value: "http://label-studio-service:8080"
        - name: LABEL_STUDIO_API_KEY
          valueFrom:
            secretKeyRef:
              name: ml-backend-secrets
              key: api-key
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-backend-service
spec:
  selector:
    app: ml-backend
  ports:
  - port: 9090
    targetPort: 9090
  type: LoadBalancer
```

---

## 8. 最佳实践

### 8.1 性能优化

**1. 批量预测**

```python
def predict(self, tasks, context=None, **kwargs):
    # 批量加载图像
    images = [self.load_image(task) for task in tasks]

    # 批量推理
    with torch.no_grad():
        results = self.model.batch_predict(images)

    # 格式化输出
    return self.format_predictions(results)
```

**2. 模型缓存**

```python
def setup(self):
    # 检查缓存
    if self.has('model_weights'):
        weights = self.get('model_weights')
        self.model.load_state_dict(weights)
    else:
        # 首次加载
        self.model = load_model()
        self.set('model_weights', self.model.state_dict())
```

**3. GPU 内存管理**

```python
import gc
import torch

def predict(self, tasks, context=None, **kwargs):
    try:
        results = self.model.predict(...)
    finally:
        # 清理 GPU 内存
        torch.cuda.empty_cache()
        gc.collect()

    return results
```

### 8.2 错误处理

```python
from label_studio_ml.exceptions import ValidationError

def predict(self, tasks, context=None, **kwargs):
    try:
        # 验证输入
        if not tasks:
            raise ValidationError("No tasks provided")

        # 生成预测
        predictions = []
        for task in tasks:
            try:
                result = self.predict_single(task)
                predictions.append(result)
            except Exception as e:
                # 记录错误但继续处理其他任务
                print(f"Error processing task {task.get('id')}: {e}")
                predictions.append({
                    'result': [],
                    'score': 0.0,
                    'error': str(e)
                })

        return ModelResponse(predictions=predictions)

    except Exception as e:
        # 记录详细错误
        import traceback
        print(f"Prediction error: {traceback.format_exc()}")
        raise
```

### 8.3 日志记录

```python
import logging

logger = logging.getLogger(__name__)

class MyModel(LabelStudioMLBase):
    def predict(self, tasks, context=None, **kwargs):
        logger.info(f"Received {len(tasks)} tasks for prediction")
        logger.debug(f"Context: {context}")

        predictions = []
        for i, task in enumerate(tasks):
            logger.info(f"Processing task {i+1}/{len(tasks)}")
            # ...

        logger.info(f"Generated {len(predictions)} predictions")
        return ModelResponse(predictions=predictions)
```

### 8.4 版本管理

```python
def fit(self, event, data, **kwargs):
    """训练后自动更新版本"""
    if event == 'ANNOTATION_CREATED':
        # 训练模型
        self.train_model(data)

        # 更新版本
        old_version = str(self.model_version)
        self.bump_model_version()  # 1.0.0 → 1.1.0
        new_version = str(self.model_version)

        logger.info(f"Model updated: {old_version} → {new_version}")

        # 保存新权重
        self.set('model_weights', self.model.state_dict())
        self.set('last_trained', datetime.now().isoformat())
```

### 8.5 安全最佳实践

**1. API Key 保护**

```bash
# 使用环境变量，不要硬编码
export LABEL_STUDIO_API_KEY=xxx

# Docker secrets
docker secret create ls_api_key api_key.txt
```

**2. 认证**

```yaml
# docker-compose.yml
environment:
  - BASIC_AUTH_USER=admin
  - BASIC_AUTH_PASS=secure_password
```

**3. 输入验证**

```python
def predict(self, tasks, context=None, **kwargs):
    # 验证任务结构
    for task in tasks:
        if 'data' not in task:
            raise ValidationError("Task missing 'data' field")

        # 验证 URL
        if not self.is_valid_url(task['data'].get('image')):
            raise ValidationError("Invalid image URL")
```

---

## 9. 常见问题

### Q1: 如何调试预测结果？

```python
# 启用调试输出
import os
os.environ['DEBUG_PLOT'] = 'true'

def predict(self, tasks, context=None, **kwargs):
    # 保存可视化结果
    import cv2
    for i, (image, mask) in enumerate(zip(images, masks)):
        overlay = self.visualize_mask(image, mask)
        cv2.imwrite(f'/tmp/debug_{i}.jpg', overlay)

    return predictions
```

### Q2: 预测超时怎么办？

在 Label Studio 中增加超时：

```bash
# Label Studio 环境变量
ML_TIMEOUT_PREDICT=300  # 5 分钟
```

或优化模型：

```python
# 使用更小的模型
USE_MOBILE_SAM=true

# 减少输出
multimask_output=False
```

### Q3: 如何处理大图像？

```python
def predict(self, tasks, context=None, **kwargs):
    MAX_SIZE = 1024

    for task in tasks:
        image = self.load_image(task)

        # Resize 大图
        if max(image.shape[:2]) > MAX_SIZE:
            scale = MAX_SIZE / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # 预测...
```

### Q4: 多个模型如何共存？

在同一项目中启动多个 ML Backend：

```bash
# Backend 1: SAM
docker compose -f sam/docker-compose.yml up -d
# 运行在 http://localhost:9090

# Backend 2: YOLO
docker compose -f yolo/docker-compose.yml up -d
# 运行在 http://localhost:9091
```

在 Label Studio 中添加两个模型连接。

---

## 10. 完整示例：运动员分割 Pipeline

### 目标

用 Grounding SAM 自动标注运动员 → Label Studio 人工修正 → 导出训练数据

### 步骤

**1. 启动 Grounding SAM Backend**

```bash
cd label-studio-ml-backend/label_studio_ml/examples/grounding_sam

# 编辑配置
cat > docker-compose.yml << 'EOF'
version: "3.8"
services:
  ml-backend:
    build: .
    environment:
      - LABEL_STUDIO_HOST=http://host.docker.internal:8080
      - LABEL_STUDIO_API_KEY=${LS_API_KEY}
      - USE_SAM=true
      - BOX_THRESHOLD=0.3
    ports:
      - "9090:9090"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF

# 启动
export LS_API_KEY=你的API密钥
docker compose up -d
```

**2. Label Studio 配置**

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>

  <Style>
    .prompt { margin: 10px 0; }
  </Style>

  <View className="prompt">
    <Header value="Text Prompt:"/>
    <TextArea name="prompt" toName="image"
              editable="true" rows="2"
              showSubmitButton="true"
              placeholder="e.g., football player, athlete"/>
  </View>

  <BrushLabels name="label" toName="image" strokeWidth="2">
    <Label value="athlete" background="#FF0000"/>
    <Label value="referee" background="#0000FF"/>
  </BrushLabels>
</View>
```

**3. 批量标注**

1. 上传 512 张图像到 Label Studio
2. Data Manager → 全选任务
3. Actions → **Add Text Prompt for GroundingDINO**
4. 输入：`football player`
5. Submit → 自动为所有任务生成预测

**4. 人工审核**

1. 开启 Quick View 模式
2. 逐张检查 mask 质量
3. 修正错误边界
4. Submit

**5. 导出**

```bash
# 导出 COCO 格式
点击 Export → COCO

# 或使用 API
curl -X GET \
  "http://localhost:8080/api/projects/1/export?exportType=COCO" \
  -H "Authorization: Token ${LS_API_KEY}" \
  -o athlete_dataset.json
```

**6. 清理和验证**

```python
import json

# 加载导出数据
with open('athlete_dataset.json') as f:
    data = json.load(f)

print(f"Images: {len(data['images'])}")
print(f"Annotations: {len(data['annotations'])}")

# 验证 mask 格式
for anno in data['annotations'][:5]:
    print(f"ID: {anno['id']}, Area: {anno['area']}, Format: {type(anno['segmentation'])}")
```

---

## 11. 参考资源

- [Label Studio ML Backend GitHub](https://github.com/HumanSignal/label-studio-ml-backend)
- [Label Studio 官方文档](https://labelstud.io/guide/ml.html)
- [SAM 2 论文](https://ai.meta.com/sam2/)
- [Grounding DINO 论文](https://github.com/IDEA-Research/GroundingDINO)
- [YOLO 官方文档](https://docs.ultralytics.com/)
- [Docker 部署指南](https://docs.docker.com/compose/)

---

## 总结

Label Studio ML Backend SDK 提供了强大而灵活的框架来集成任何机器学习模型。关键要点：

✅ **预构建示例丰富**：SAM、YOLO、DINO 等开箱即用
✅ **扩展简单**：继承基类，实现 `predict()` 即可
✅ **交互式支持**：支持点击、框选等实时交互
✅ **生产就绪**：Docker 部署、GPU 支持、健康检查
✅ **与 Label Studio 无缝集成**：标准 REST API

通过本指南，你应该能够：
1. 快速启动预构建的 SAM/YOLO backend
2. 创建自定义 ML backend
3. 部署到生产环境
4. 构建完整的自动标注 Pipeline
