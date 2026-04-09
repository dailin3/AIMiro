FROM python:3.13-bookworm

WORKDIR /app

# 安装系统依赖（ARMv6 兼容）
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml ./
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 安装 Python 依赖（跳过 opencv 和 numpy，用系统包）
RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    requests \
    python-dotenv

# 创建数据目录
RUN mkdir -p /app/data/temp /app/data/pic /app/data/faces

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["python", "-m", "backend.core"]
