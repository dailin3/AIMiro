# miro-demo

使用 `uv` 在本地运行，不再依赖 Docker 容器。

## 1. 安装 uv

macOS:

```bash
brew install uv
```

## 2. 创建环境并安装依赖

在项目根目录执行：

```bash
uv sync --all-groups
```

这会自动创建 `.venv` 并安装运行依赖与开发依赖。

## 3. 常用命令

运行后端（当前为模板骨架）：

```bash
uv run python backend_server.py
```

运行摄像头模块（当前为模板骨架）：

```bash
uv run python camera_capture.py
```

运行图像分析模块（当前为模板骨架）：

```bash
uv run python ai_image_analysis.py
```

运行测试：

```bash
uv run pytest
```

代码格式化：

```bash
uv run black .
```

静态检查：

```bash
uv run flake8 .
```

## 4. 数据目录

项目使用本地 `data/` 目录保存数据文件。
