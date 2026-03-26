FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装常用开发工具
RUN pip install --upgrade pip && \
    pip install \
    black \
    flake8 \
    pytest \
    pytest-cov \
    ipython \
    python-dotenv

# 创建非 root 用户
RUN useradd -m -u 1000 developer && chown -R developer:developer /app
USER developer

# 默认命令
CMD ["python"]
