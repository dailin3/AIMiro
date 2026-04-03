# Miro Demo

人脸检测与识别系统 - 使用 OpenCV 实现人脸识别功能

## 快速开始

### 1. 安装 uv

macOS:

```bash
brew install uv
```

### 2. 创建环境并安装依赖

在项目根目录执行：

```bash
uv sync --all-groups
```

这会自动创建 `.venv` 并安装运行依赖与开发依赖。

### 3. 运行服务

**启动后端服务器：**

```bash
uv run python -m backend.backend_server
```

或

```bash
cd backend && uv run python backend_server.py
```

**运行测试：**

```bash
uv run python -m pytest -q tests/test_face_recognition_smoke.py
```

---

## 项目结构

```
miro-demo/
├── backend/                 # 后端服务代码
│   ├── __init__.py
│   ├── backend_server.py    # Flask API 服务器
│   ├── camera_capture.py    # 摄像头采集模块
│   ├── ai_image_analysis.py # AI 图像分析模块
│   └── face_recognition_module.py  # 人脸识别模块
├── tests/                   # 测试脚本
│   └── test_face_recognition_smoke.py  # 人脸识别冒烟测试
├── frontend/                # 前端页面
│   └── preview.html
├── data/                    # 数据目录
│   ├── faces/              # 人脸数据库
│   └── pic/                # 拍摄的图片
├── pyproject.toml          # 项目配置
└── README.md               # 项目文档
```

---

## 功能概述

### 人脸识别功能

系统使用 OpenCV 实现人脸识别，包括：

- **人脸检测**：检测图像中的人脸位置
- **人脸注册**：将人脸特征保存到数据库
- **人脸识别**：识别图像中的人脸身份
- **人脸比对**：比对两张图片中的人脸是否为同一人

### 技术实现

由于系统环境限制，使用 OpenCV 内置功能实现：

- **人脸检测**：使用 Haar 级联分类器（DNN 模型可选）
- **特征提取**：优先使用 MobileNet ONNX，缺失时回退到简单特征
- **人脸比对**：使用余弦相似度

> **注意**：当前代码支持 MobileNet ONNX 特征模型。如果没有提供模型文件，会自动回退到简单特征提取，保证流程可用。

---

## API 接口文档

### 1. 注册人脸

**接口**: `POST /face/register`

**请求体**:
```json
{
  "path": "/path/to/image.jpg",
  "face_id": "person_001",
  "name": "张三",
  "face_index": 0
}
```

**参数说明**:
- `path`: 图像文件路径（必需）
- `face_id`: 人脸唯一标识（必需）
- `name`: 人脸名称（可选）
- `face_index`: 如果图像中有多张人脸，指定注册哪一张（可选，默认 0）

**响应示例**:
```json
{
  "success": true,
  "face_id": "person_001",
  "name": "张三",
  "location": [100, 150, 200, 250]
}
```

---

### 2. 识别人脸

**接口**: `POST /face/recognize`

**请求体**:
```json
{
  "path": "/path/to/image.jpg"
}
```

**响应示例**:
```json
{
  "success": true,
  "face_count": 1,
  "faces": [
    {
      "face_index": 0,
      "location": {"x": 100, "y": 150, "width": 200, "height": 250},
      "confidence": 0.95,
      "matches": [
        {
          "face_id": "person_001",
          "name": "张三",
          "similarity": 0.85
        }
      ]
    }
  ]
}
```

---

### 3. 人脸比对

**接口**: `POST /face/compare`

**请求体**:
```json
{
  "path1": "/path/to/image1.jpg",
  "path2": "/path/to/image2.jpg"
}
```

**响应示例**:
```json
{
  "success": true,
  "is_match": true,
  "similarity": 0.87,
  "face_count_image1": 1,
  "face_count_image2": 1
}
```

---

### 4. 获取人脸列表

**接口**: `GET /face/list`

**响应示例**:
```json
{
  "success": true,
  "count": 2,
  "faces": [
    {"face_id": "person_001", "name": "张三", "has_encoding": true},
    {"face_id": "person_002", "name": "李四", "has_encoding": true}
  ]
}
```

---

### 5. 删除人脸

**接口**: `POST /face/delete`

**请求体**:
```json
{
  "face_id": "person_001"
}
```

---

### 6. 清空所有人脸

**接口**: `POST /face/clear`

---

### 7. 更新人脸名称

**接口**: `POST /face/update_name`

**请求体**:
```json
{
  "face_id": "person_001",
  "name": "新名称"
}
```

---

## 命令行使用

### 注册人脸
```bash
uv run python -m backend.face_recognition_module --register person_001 --image photo.jpg --name "张三"
```

### 识别人脸
```bash
uv run python -m backend.face_recognition_module --recognize --image test.jpg
```

### 列出已注册人脸
```bash
uv run python -m backend.face_recognition_module --list
```

### 删除人脸
```bash
uv run python -m backend.face_recognition_module --delete person_001
```

### 清空所有人脸
```bash
uv run python -m backend.face_recognition_module --clear
```

---

## 使用流程示例

### 1. 启动服务
```bash
uv run python -m backend.backend_server
```

### 2. 注册人脸
```bash
curl -X POST http://localhost:5000/face/register \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/pic/2026-03-27/123456.jpg", "face_id": "zhangsan", "name": "张三"}'
```

### 3. 识别人脸
```bash
curl -X POST http://localhost:5000/face/recognize \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/pic/2026-03-27/123789.jpg"}'
```

### 4. 比对两张图片
```bash
curl -X POST http://localhost:5000/face/compare \
  -H "Content-Type: application/json" \
  -d '{"path1": "./data/pic/2026-03-27/img1.jpg", "path2": "./data/pic/2026-03-27/img2.jpg"}'
```

---

## 测试

### 运行人脸识别测试

```bash
# 推荐测试脚本（准确率高）
uv run python -m test.test_module_real

# 简化测试脚本
uv run python -m test.test_face_simple

# 完整测试脚本
uv run python -m test.test_face_recognition --dataset orl
```

### 测试结果

在 ORL 人脸数据集上的测试结果：

| 指标 | 结果 |
|------|------|
| 准确率 (Accuracy) | 94.17% |
| 精确率 (Precision) | 95.67% |
| 召回率 (Recall) | 94.17% |
| F1 分数 | 94.26% |

---

## 常用命令

**运行后端：**
```bash
uv run python -m backend.backend_server
```

**运行测试：**
```bash
uv run pytest
```

**代码格式化：**
```bash
uv run black .
```

**静态检查：**
```bash
uv run flake8 .
```

---

## 数据库位置

人脸特征数据库保存在：`./data/faces/face_db.pkl`

---

## 注意事项

1. **图像质量**：确保人脸清晰、光线充足、正脸拍摄
2. **人脸大小**：建议人脸区域至少 100x100 像素
3. **识别阈值**：默认相似度阈值为 0.6，可在配置中调整
4. **多人场景**：如图像中有多张人脸，使用 `face_index` 指定要处理的人脸
5. **精度限制**：当前为简化实现，如需更高精度建议安装完整版 face_recognition 库

---

## 安装完整版本（可选）

如需更高精度的人脸识别，需要安装 cmake 并编译 dlib：

```bash
# macOS 使用 Homebrew
brew install cmake
pip install dlib face-recognition

# 或使用 uv 安装可选依赖
uv sync --extra face-recognition
```

---

## 数据目录

项目使用本地 `data/` 目录保存数据文件：

- `data/faces/` - 人脸特征数据库
- `data/pic/` - 拍摄的图片（按日期组织）
