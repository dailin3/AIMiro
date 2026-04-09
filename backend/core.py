#!/usr/bin/env python3
"""
核心调度器模块

负责常驻摄像头管理、HTTP 触发接口、临时文件管理。
设计用于树莓派等嵌入式设备，支持快速连续拍照和远程触发。

功能：
- 常驻摄像头进程（后台线程持续获取最新帧到内存）
- HTTP API 触发拍照（单次/批量）
- Temp 文件生命周期管理
- 状态监控与健康检查
"""

import os
import sys
import time
import signal
import shutil
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from flask import Flask, jsonify, request
import cv2
import numpy as np

# 从现有模块导入
from backend.camera_capture import CameraConfig, CameraCapture

# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 状态枚举
# ============================================================================

class CameraState(str, Enum):
    """摄像头状态机"""
    IDLE = "idle"               # 初始状态，未初始化
    READY = "ready"             # 摄像头就绪，正在采集帧
    CAPTURING = "capturing"     # 正在拍照（保存文件）
    ERROR = "error"             # 错误状态（摄像头不可用）


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class CoreConfig:
    """核心调度器配置"""

    # 摄像头配置
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_width: int = int(os.getenv("CAMERA_WIDTH", "1920"))
    camera_height: int = int(os.getenv("CAMERA_HEIGHT", "1080"))
    camera_fps: float = float(os.getenv("CAMERA_FPS", "30.0"))

    # 目录配置
    data_dir: str = os.getenv("DATA_DIR", "./data")
    temp_dir: str = os.getenv("TEMP_DIR", "./data/temp")

    # Temp 文件管理
    temp_max_age_hours: int = int(os.getenv("TEMP_MAX_AGE_HOURS", "24"))
    temp_max_count: int = int(os.getenv("TEMP_MAX_COUNT", "1000"))

    # HTTP 服务配置
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "8080"))

    # 系统配置
    init_timeout_seconds: int = int(os.getenv("INIT_TIMEOUT_SECONDS", "5"))
    min_disk_space_mb: int = int(os.getenv("MIN_DISK_SPACE_MB", "100"))
    frame_capture_interval: float = float(os.getenv("FRAME_CAPTURE_INTERVAL", "0.033"))  # ~30fps

    def validate(self) -> bool:
        """验证配置是否有效"""
        if self.camera_width <= 0 or self.camera_height <= 0:
            logger.error("无效的摄像头分辨率")
            return False
        if self.camera_fps <= 0:
            logger.error("无效的帧率")
            return False
        if self.server_port < 1 or self.server_port > 65535:
            logger.error("无效的端口号")
            return False
        if self.init_timeout_seconds <= 0:
            logger.error("无效的初始化超时时间")
            return False
        return True


# ============================================================================
# 核心调度器
# ============================================================================

class CoreScheduler:
    """
    核心调度器

    职责：
    - 管理常驻摄像头（后台线程持续采集帧到内存）
    - 提供拍照接口（从内存获取最新帧，保存到 temp/）
    - 状态管理与线程安全
    - Temp 文件清理
    """

    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config or CoreConfig()

        # 状态管理
        self._state = CameraState.IDLE
        self._state_lock = threading.Lock()

        # 帧管理
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_timestamp: Optional[float] = None  # 最新帧的时间戳

        # 摄像头对象
        self._camera: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False

        # 拍照序列号（用于文件名）
        self._capture_counter = 0
        self._counter_lock = threading.Lock()

        # 启动时间
        self._start_time: Optional[float] = None

        # 确保目录存在
        self._ensure_directories()

    # ------------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------------

    def start(self) -> bool:
        """
        启动调度器

        流程：
        1. 检查磁盘空间
        2. 初始化摄像头
        3. 启动后台帧采集线程

        Returns:
            启动是否成功
        """
        logger.info("=" * 60)
        logger.info("核心调度器启动中...")
        logger.info(f"配置: 摄像头索引={self.config.camera_index}, "
                    f"分辨率={self.config.camera_width}x{self.config.camera_height}, "
                    f"帧率={self.config.camera_fps}")

        # 1. 检查磁盘空间
        if not self._check_disk_space():
            logger.warning("磁盘空间不足，但继续启动")

        # 2. 初始化摄像头（带超时）
        if not self._init_camera_with_timeout():
            self._set_state(CameraState.ERROR)
            logger.error("摄像头初始化失败")
            return False

        # 3. 启动后台帧采集线程
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._frame_capture_loop,
            name="FrameCaptureThread",
            daemon=True
        )
        self._capture_thread.start()
        logger.info("后台帧采集线程已启动")

        self._set_state(CameraState.READY)
        self._start_time = time.time()
        logger.info("核心调度器启动完成，状态: READY")
        return True

    def stop(self) -> None:
        """停止调度器（优雅关闭）"""
        logger.info("核心调度器关闭中...")
        self._running = False

        # 等待采集线程退出
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
            if self._capture_thread.is_alive():
                logger.warning("帧采集线程未能在 5 秒内退出")

        # 释放摄像头
        if self._camera is not None:
            self._camera.release()
            self._camera = None
            logger.info("摄像头资源已释放")

        self._set_state(CameraState.IDLE)
        logger.info("核心调度器已关闭")

    def graceful_shutdown(self, signum=None, frame=None) -> None:
        """处理系统信号，优雅关闭"""
        sig_name = signal.Signals(signum).name if signum else "UNKNOWN"
        logger.info(f"收到信号 {sig_name}，准备优雅关闭...")
        self.stop()
        sys.exit(0)

    # ------------------------------------------------------------------------
    # 摄像头初始化
    # ------------------------------------------------------------------------

    def _init_camera_with_timeout(self) -> bool:
        """带超时的摄像头初始化"""
        logger.info(f"初始化摄像头（超时 {self.config.init_timeout_seconds} 秒）...")

        result = [False]  # 用列表包装以便在闭包中修改
        init_done = threading.Event()

        def _init():
            try:
                camera_config = CameraConfig(
                    camera_index=self.config.camera_index,
                    width=self.config.camera_width,
                    height=self.config.camera_height,
                    fps=self.config.camera_fps
                )
                capture = CameraCapture(config=camera_config, data_dir=self.config.data_dir)
                success = capture.initialize()
                if success and capture.cap is not None:
                    self._camera = capture.cap
                    result[0] = True
            except Exception as e:
                logger.error(f"摄像头初始化异常: {e}")
            finally:
                init_done.set()

        init_thread = threading.Thread(target=_init, name="CameraInitThread", daemon=True)
        init_thread.start()

        # 等待初始化完成或超时
        if init_done.wait(timeout=self.config.init_timeout_seconds):
            return result[0]
        else:
            logger.error(f"摄像头初始化超时（{self.config.init_timeout_seconds} 秒）")
            return False

    # ------------------------------------------------------------------------
    # 后台帧采集循环
    # ------------------------------------------------------------------------

    def _frame_capture_loop(self) -> None:
        """
        后台线程：持续从摄像头获取最新帧并保存到内存

        运行在独立线程中，约 30fps 获取帧。
        """
        logger.info("帧采集循环开始")
        consecutive_errors = 0
        max_consecutive_errors = 10  # 连续错误次数上限

        # 摄像头预热：等待 2 秒让摄像头稳定
        logger.info("摄像头预热中（2 秒）...")
        time.sleep(2)
        # 清空缓冲区
        if self._camera:
            for _ in range(5):
                self._camera.read()
        logger.info("摄像头预热完成")

        while self._running:
            try:
                if self._camera is None or not self._camera.isOpened():
                    logger.error("摄像头未打开")
                    self._set_state(CameraState.ERROR)
                    time.sleep(1)
                    continue

                ret, frame = self._camera.read()
                if ret and frame is not None:
                    # 更新内存中的最新帧
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                        self._frame_timestamp = time.time()
                    consecutive_errors = 0  # 重置错误计数
                else:
                    consecutive_errors += 1
                    if consecutive_errors <= 3:
                        logger.debug(f"读取帧失败 ({consecutive_errors}/{max_consecutive_errors})")
                    elif consecutive_errors == 4:
                        logger.warning(f"连续读取帧失败，可能摄像头有问题 ({consecutive_errors}/{max_consecutive_errors})")
                    else:
                        logger.warning(f"读取帧失败 ({consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("连续读取失败次数过多，标记为 ERROR 状态")
                        self._set_state(CameraState.ERROR)

                # 控制帧率
                time.sleep(self.config.frame_capture_interval)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"帧采集异常: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    self._set_state(CameraState.ERROR)
                time.sleep(1)

        logger.info("帧采集循环结束")

    # ------------------------------------------------------------------------
    # 拍照功能
    # ------------------------------------------------------------------------

    def capture(self) -> Optional[Dict[str, Any]]:
        """
        拍照并保存到 temp/ 目录

        从内存中获取最新帧（无延迟），支持快速连续调用。

        Returns:
            拍照结果字典，包含：
            - success: 是否成功
            - file_path: 保存的文件路径
            - timestamp: 拍照时间戳
            - error: 错误信息（如果失败）
        """
        # 状态检查
        current_state = self.get_state()
        if current_state != CameraState.READY:
            error_msg = f"摄像头状态为 {current_state}，无法拍照"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}

        # 切换到 CAPTURING 状态
        self._set_state(CameraState.CAPTURING)

        try:
            # 从内存获取最新帧
            with self._frame_lock:
                if self._latest_frame is None:
                    return {"success": False, "error": "内存中暂无帧数据，请稍后重试"}
                frame = self._latest_frame.copy()

            # 生成文件名
            timestamp = datetime.now()
            sequence = self._get_next_sequence()
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{sequence:04d}.jpg"

            # 确保 temp 目录存在
            temp_path = Path(self.config.temp_dir)
            temp_path.mkdir(parents=True, exist_ok=True)

            # 保存文件
            file_path = temp_path / filename
            success = cv2.imwrite(str(file_path), frame)

            if not success:
                return {"success": False, "error": f"保存文件失败: {file_path}"}

            logger.info(f"拍照成功: {file_path}")

            return {
                "success": True,
                "file_path": str(file_path),
                "filename": filename,
                "timestamp": timestamp.isoformat(),
                "sequence": sequence
            }

        except Exception as e:
            logger.error(f"拍照异常: {e}")
            return {"success": False, "error": str(e)}

        finally:
            # 恢复到 READY 状态
            self._set_state(CameraState.READY)

    def capture_burst(self, count: int = 3, interval: float = 0.5) -> Dict[str, Any]:
        """
        连续拍照

        Args:
            count: 拍照数量
            interval: 间隔时间（秒）

        Returns:
            批量拍照结果
        """
        if count <= 0:
            return {"success": False, "error": "拍照数量必须大于 0"}
        if count > 50:
            return {"success": False, "error": "单次批量拍照数量不能超过 50 张"}

        logger.info(f"开始批量拍照: count={count}, interval={interval}s")
        results = []
        success_count = 0

        for i in range(count):
            result = self.capture()
            results.append(result)
            if result.get("success"):
                success_count += 1

            # 间隔（最后一张不需要等待）
            if i < count - 1 and interval > 0:
                time.sleep(interval)

        logger.info(f"批量拍照完成: 成功 {success_count}/{count}")

        return {
            "success": success_count > 0,
            "total": count,
            "success_count": success_count,
            "failed_count": count - success_count,
            "results": results
        }

    def _get_next_sequence(self) -> int:
        """获取下一个拍照序列号（线程安全）"""
        with self._counter_lock:
            self._capture_counter += 1
            return self._capture_counter

    # ------------------------------------------------------------------------
    # Temp 文件管理
    # ------------------------------------------------------------------------

    def cleanup_temp_files(self, max_age_hours: Optional[int] = None) -> Dict[str, Any]:
        """
        清理 temp 目录中的旧文件

        Args:
            max_age_hours: 文件最大保留时间（小时），使用配置默认值如果为 None

        Returns:
            清理结果
        """
        hours = max_age_hours or self.config.temp_max_age_hours
        temp_path = Path(self.config.temp_dir)

        if not temp_path.exists():
            return {"success": True, "deleted_count": 0, "message": "Temp 目录不存在"}

        deleted_count = 0
        total_size = 0
        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        total_size += file_size
                        logger.info(f"已清理过期文件: {file_path.name} "
                                    f"(修改时间: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')})")

            freed_mb = total_size / (1024 * 1024)
            message = f"已清理 {deleted_count} 个文件，释放空间 {freed_mb:.2f} MB"
            logger.info(message)

            return {
                "success": True,
                "deleted_count": deleted_count,
                "freed_space_mb": round(freed_mb, 2),
                "message": message
            }

        except Exception as e:
            logger.error(f"清理 temp 文件异常: {e}")
            return {"success": False, "error": str(e)}

    def enforce_temp_count_limit(self) -> Dict[str, Any]:
        """
        强制限制 temp 文件数量（删除最旧的文件）

        Returns:
            清理结果
        """
        temp_path = Path(self.config.temp_dir)

        if not temp_path.exists():
            return {"success": True, "deleted_count": 0}

        try:
            files = sorted(
                temp_path.iterdir(),
                key=lambda f: f.stat().st_mtime
            )

            deleted_count = 0
            while len(files) > self.config.temp_max_count:
                oldest_file = files.pop(0)
                oldest_file.unlink()
                deleted_count += 1
                logger.info(f"已删除超出数量限制的文件: {oldest_file.name}")

            return {
                "success": True,
                "deleted_count": deleted_count,
                "remaining_count": len(files)
            }

        except Exception as e:
            logger.error(f"执行数量限制异常: {e}")
            return {"success": False, "error": str(e)}

    def get_temp_info(self) -> Dict[str, Any]:
        """获取 temp 目录信息"""
        temp_path = Path(self.config.temp_dir)

        if not temp_path.exists():
            return {"exists": False, "file_count": 0, "total_size_mb": 0}

        files = list(temp_path.iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return {
            "exists": True,
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_age_hours": self.config.temp_max_age_hours,
            "max_count": self.config.temp_max_count
        }

    # ------------------------------------------------------------------------
    # 状态与工具方法
    # ------------------------------------------------------------------------

    def get_state(self) -> CameraState:
        """获取当前状态（线程安全）"""
        with self._state_lock:
            return self._state

    def _set_state(self, state: CameraState) -> None:
        """设置状态（线程安全）"""
        with self._state_lock:
            self._state = state
            logger.debug(f"状态变更: {state}")

    def _check_disk_space(self) -> bool:
        """检查磁盘空间"""
        try:
            stat = shutil.disk_usage(self.config.data_dir)
            free_mb = stat.free / (1024 * 1024)

            if free_mb < self.config.min_disk_space_mb:
                logger.warning(f"磁盘空间不足: {free_mb:.2f} MB < {self.config.min_disk_space_mb} MB")
                return False

            logger.info(f"磁盘可用空间: {free_mb:.2f} MB")
            return True

        except Exception as e:
            logger.error(f"检查磁盘空间异常: {e}")
            return False

    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"数据目录: {Path(self.config.data_dir).absolute()}")
        logger.info(f"Temp 目录: {Path(self.config.temp_dir).absolute()}")

    def get_status(self) -> Dict[str, Any]:
        """获取调度器完整状态"""
        uptime = time.time() - self._start_time if self._start_time else 0

        with self._frame_lock:
            frame_age = time.time() - self._frame_timestamp if self._frame_timestamp else None

        return {
            "state": self.get_state().value,
            "uptime_seconds": round(uptime, 1),
            "camera_index": self.config.camera_index,
            "resolution": f"{self.config.camera_width}x{self.config.camera_height}",
            "frame_age_seconds": round(frame_age, 2) if frame_age is not None else None,
            "capture_counter": self._capture_counter,
            "temp_info": self.get_temp_info(),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# Flask HTTP 接口
# ============================================================================

def create_app(scheduler: CoreScheduler) -> Flask:
    """
    创建 Flask 应用

    路由：
    - POST /trigger         单次拍照
    - POST /trigger/burst   批量拍照（参数：count, interval）
    - GET  /status          状态查询
    - GET  /health          健康检查
    - DELETE /temp/cleanup  清理 temp 目录
    """
    app = Flask(__name__)

    # 禁用默认日志（使用自定义日志）
    app.logger.disabled = True
    import flask
    flask.logging.default_handler.setFormatter(logging.Formatter('%(message)s'))

    @app.route("/trigger", methods=["POST"])
    def trigger():
        """单次拍照"""
        try:
            result = scheduler.capture()
            if result.get("success"):
                return jsonify(result), 200
            else:
                return jsonify(result), 503  # Service Unavailable
        except Exception as e:
            logger.error(f"/trigger 异常: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/trigger/burst", methods=["POST"])
    def trigger_burst():
        """批量拍照"""
        try:
            data = request.get_json(silent=True) or {}
            count = int(data.get("count", 3))
            interval = float(data.get("interval", 0.5))

            result = scheduler.capture_burst(count=count, interval=interval)
            status_code = 200 if result.get("success") else 503
            return jsonify(result), status_code

        except ValueError as e:
            return jsonify({"success": False, "error": f"参数错误: {e}"}), 400
        except Exception as e:
            logger.error(f"/trigger/burst 异常: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/status", methods=["GET"])
    def status():
        """状态查询"""
        try:
            return jsonify(scheduler.get_status()), 200
        except Exception as e:
            logger.error(f"/status 异常: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health():
        """健康检查"""
        state = scheduler.get_state()
        is_healthy = state == CameraState.READY

        response = {
            "status": "healthy" if is_healthy else "unhealthy",
            "state": state.value,
            "timestamp": datetime.now().isoformat()
        }

        status_code = 200 if is_healthy else 503
        return jsonify(response), status_code

    @app.route("/temp/cleanup", methods=["DELETE", "POST"])
    def temp_cleanup():
        """清理 temp 目录"""
        try:
            data = request.get_json(silent=True) or {}
            max_age_hours = data.get("max_age_hours")

            # 先清理过期文件
            result = scheduler.cleanup_temp_files(max_age_hours=max_age_hours)

            # 再执行数量限制
            count_result = scheduler.enforce_temp_count_limit()
            result["count_limit_enforced"] = count_result

            status_code = 200 if result.get("success") else 500
            return jsonify(result), status_code

        except Exception as e:
            logger.error(f"/temp/cleanup 异常: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/temp/info", methods=["GET"])
    def temp_info():
        """获取 temp 目录信息"""
        try:
            return jsonify(scheduler.get_temp_info()), 200
        except Exception as e:
            logger.error(f"/temp/info 异常: {e}")
            return jsonify({"error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not Found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"error": "Method Not Allowed"}), 405

    return app


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：启动核心调度器和 HTTP 服务"""

    # 注册信号处理器
    signal.signal(signal.SIGINT, lambda s, f: scheduler.graceful_shutdown(s))
    signal.signal(signal.SIGTERM, lambda s, f: scheduler.graceful_shutdown(s))

    # 加载配置
    config = CoreConfig()
    if not config.validate():
        logger.error("配置验证失败，退出")
        sys.exit(1)

    # 创建调度器
    global scheduler
    scheduler = CoreScheduler(config=config)

    # 启动调度器
    if not scheduler.start():
        logger.error("调度器启动失败，退出")
        sys.exit(1)

    # 创建并启动 Flask 服务
    app = create_app(scheduler)

    logger.info("=" * 60)
    logger.info(f"HTTP 服务启动中: http://{config.server_host}:{config.server_port}")
    logger.info("可用接口:")
    logger.info("  POST   /trigger          - 单次拍照")
    logger.info("  POST   /trigger/burst    - 批量拍照")
    logger.info("  GET    /status           - 状态查询")
    logger.info("  GET    /health           - 健康检查")
    logger.info("  DELETE /temp/cleanup     - 清理 temp 文件")
    logger.info("  GET    /temp/info        - Temp 目录信息")
    logger.info("=" * 60)

    try:
        app.run(
            host=config.server_host,
            port=config.server_port,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        scheduler.stop()


# 全局调度器实例（用于信号处理）
scheduler: Optional[CoreScheduler] = None


if __name__ == "__main__":
    main()
