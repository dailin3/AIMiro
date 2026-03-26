#!/usr/bin/env python3
"""
摄像头采集与储存模块

负责从摄像头采集图像/视频流，并储存到指定目录。
提供窗口预览和网页预览功能。
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import logging
import threading
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraConfig:
    """摄像头配置类"""
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        codec: str = "MJPG"
    ):
        """
        初始化摄像头配置
        
        Args:
            camera_index: 摄像头设备索引，0 表示默认摄像头
            width: 分辨率宽度
            height: 分辨率高度
            fps: 帧率
            codec: 编码格式
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec


class CameraCapture:
    """摄像头采集类"""
    
    def __init__(self, config: Optional[CameraConfig] = None, data_dir: str = "./data"):
        """
        初始化摄像头采集器
        
        Args:
            config: 摄像头配置，使用默认配置如果为 None
            data_dir: 图像保存目录
        """
        self.config = config or CameraConfig()
        self.data_dir = Path(data_dir)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
        
        # 自动创建数据目录
        self._create_data_dir()
    
    def _create_data_dir(self) -> None:
        """创建数据目录"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"数据目录已创建：{self.data_dir.absolute()}")
    
    def initialize(self) -> bool:
        """
        初始化摄像头
        
        Returns:
            初始化是否成功
        """
        try:
            # 打开摄像头
            self.cap = cv2.VideoCapture(self.config.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 (索引：{self.config.camera_index})")
                return False
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            
            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # 设置编码格式
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            
            # 验证配置是否应用成功（实际值可能与设置值不同）
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"摄像头初始化成功")
            logger.info(f"实际分辨率：{actual_width}x{actual_height}")
            logger.info(f"实际帧率：{actual_fps} FPS")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"初始化摄像头时出错：{e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获单帧图像
        
        Returns:
            图像帧（BGR 格式的 numpy 数组），失败返回 None
        """
        if not self.is_initialized or self.cap is None:
            logger.error("摄像头未初始化")
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning("无法读取图像帧")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"捕获图像帧时出错：{e}")
            return None
    
    def capture_frames(self, count: int, interval: float = 0.0) -> list:
        """
        捕获多帧图像
        
        Args:
            count: 要捕获的帧数
            interval: 帧间间隔（秒）
            
        Returns:
            图像帧列表
        """
        frames = []
        
        for i in range(count):
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame)
            
            if interval > 0:
                import time
                time.sleep(interval)
        
        logger.info(f"成功捕获 {len(frames)}/{count} 帧图像")
        return frames
    
    def save_image(
        self,
        image: np.ndarray,
        filename: Optional[str] = None,
        format: str = "jpg"
    ) -> Optional[Path]:
        """
        保存图像到文件
        
        Args:
            image: 要保存的图像
            filename: 文件名，如果为 None 则使用时间戳生成
            format: 图像格式（jpg, png 等）
            
        Returns:
            保存的文件路径，失败返回 None
        """
        if image is None:
            logger.error("图像为空，无法保存")
            return None

        try:
            now = datetime.now()
            
            # 按日期创建子文件夹：data/pic/2026-03-26/
            date_folder = self.data_dir / "pic" / now.strftime("%Y-%m-%d")
            date_folder.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名：时分秒.png
            if filename is None:
                filename = now.strftime("%H%M%S") + f".{format}"
            
            # 确保扩展名正确
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"
            
            # 完整路径
            file_path = date_folder / filename
            
            # 保存图像
            success = cv2.imwrite(str(file_path), image)
            
            if success:
                logger.info(f"图像已保存：{file_path}")
                return file_path
            else:
                logger.error(f"保存图像失败：{file_path}")
                return None
                
        except Exception as e:
            logger.error(f"保存图像时出错：{e}")
            return None
    
    def capture_and_save(
        self,
        filename: Optional[str] = None,
        format: str = "jpg"
    ) -> Optional[Path]:
        """
        捕获当前帧并保存
        
        Args:
            filename: 文件名
            format: 图像格式
            
        Returns:
            保存的文件路径，失败返回 None
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        
        return self.save_image(frame, filename, format)
    
    def get_video_stream(self) -> Optional[cv2.VideoCapture]:
        """
        获取视频流对象
        
        Returns:
            VideoCapture 对象，如果未初始化返回 None
        """
        if not self.is_initialized or self.cap is None:
            logger.error("摄像头未初始化")
            return None
        return self.cap
    
    def check_disk_space(self, min_space_mb: int = 100) -> bool:
        """
        检查磁盘空间
        
        Args:
            min_space_mb: 最小可用空间（MB）
            
        Returns:
            空间是否充足
        """
        import shutil
        try:
            stat = shutil.disk_usage(self.data_dir)
            free_mb = stat.free / (1024 * 1024)
            
            if free_mb < min_space_mb:
                logger.warning(f"磁盘空间不足：{free_mb:.2f} MB < {min_space_mb} MB")
                return False
            
            logger.info(f"磁盘可用空间：{free_mb:.2f} MB")
            return True
            
        except Exception as e:
            logger.error(f"检查磁盘空间时出错：{e}")
            return False
    
    def cleanup_old_files(
        self,
        max_files: int = 100,
        max_age_days: int = 7
    ) -> int:
        """
        清理旧文件
        
        Args:
            max_files: 最大文件数量
            max_age_days: 文件最大保存天数
            
        Returns:
            删除的文件数量
        """
        deleted_count = 0
        now = datetime.now()
        
        try:
            # 获取所有文件并按时间排序
            files = sorted(
                self.data_dir.glob("*"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # 删除超过数量限制的文件
            if len(files) > max_files:
                for file_path in files[max_files:]:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"已删除旧文件：{file_path}")
            
            # 删除超过时间的文件
            for file_path in files[:max_files]:
                file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.days > max_age_days:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"已删除过期文件：{file_path}")
            
            logger.info(f"共删除 {deleted_count} 个文件")
            
        except Exception as e:
            logger.error(f"清理旧文件时出错：{e}")
        
        return deleted_count
    
    def release(self) -> None:
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.is_initialized = False
            logger.info("摄像头资源已释放")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()
    
    def preview_window(
        self,
        window_name: str = "Camera Preview",
        wait_key: int = 1,
        enable_capture: bool = True
    ) -> bool:
        """
        使用 OpenCV 窗口预览摄像头画面

        Args:
            window_name: 窗口标题
            wait_key: 等待按键的时间（毫秒），1 表示实时
            enable_capture: 是否启用按键拍照（空格键）

        Returns:
            是否成功退出（按 q 或 ESC 退出）
        """
        if not self.is_initialized or self.cap is None:
            logger.error("摄像头未初始化")
            return False

        logger.info(f"打开预览窗口 '{window_name}'，按 'q' 或 ESC 退出，按空格键拍照")

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            capture_count = 0

            while True:
                frame = self.capture_frame()

                if frame is None:
                    logger.warning("无法获取图像帧")
                    continue

                # 添加提示信息
                display_frame = frame.copy()
                tips = "Q:Quit | SPACE:Capture"
                cv2.putText(display_frame, tips, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(window_name, display_frame)

                # 检测按键
                key = cv2.waitKey(wait_key) & 0xFF
                if key == ord('q') or key == 27:  # q 或 ESC
                    break
                elif key == ord(' ') and enable_capture:  # 空格键拍照
                    file_path = self.capture_and_save()
                    if file_path:
                        capture_count += 1
                        logger.info(f"已拍照 {capture_count} 张，保存到：{file_path}")
                        # 显示拍照提示
                        cv2.putText(display_frame, "CAPTURED!", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.imshow(window_name, display_frame)
                        cv2.waitKey(500)  # 短暂显示提示

            cv2.destroyWindow(window_name)
            logger.info(f"预览窗口已关闭，共拍照 {capture_count} 张")
            return True

        except Exception as e:
            logger.error(f"预览窗口出错：{e}")
            cv2.destroyWindow(window_name)
            return False
    
    def preview_web(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        quality: int = 80
    ) -> None:
        """
        启动网页预览服务器

        Args:
            host: 服务器地址
            port: 服务器端口
            quality: JPEG 质量 (1-100)
        """
        try:
            from flask import Flask, Response, jsonify, request
        except ImportError:
            logger.error("Flask 未安装，无法启动网页预览")
            return

        app = Flask(__name__)
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._capture_count = 0

        def generate_frames():
            """后台线程：持续获取最新帧"""
            while self._streaming:
                frame = self.capture_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._current_frame = frame
                time.sleep(0.03)  # 约 30fps

        @app.route('/')
        def index():
            """返回预览页面"""
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>摄像头预览</title>
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        background: #1a1a1a;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    }
                    .container {
                        text-align: center;
                    }
                    h1 {
                        color: #fff;
                        margin-bottom: 20px;
                    }
                    .video-container {
                        position: relative;
                        display: inline-block;
                    }
                    img {
                        max-width: 100%;
                        border-radius: 8px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    }
                    .capture-btn {
                        margin-top: 20px;
                        padding: 15px 40px;
                        font-size: 18px;
                        background: #007bff;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        transition: background 0.2s;
                    }
                    .capture-btn:hover {
                        background: #0056b3;
                    }
                    .capture-btn:active {
                        transform: scale(0.98);
                    }
                    .tips {
                        color: #888;
                        margin-top: 15px;
                        font-size: 14px;
                    }
                    .status {
                        color: #28a745;
                        margin-top: 10px;
                        font-size: 16px;
                        min-height: 24px;
                    }
                    .count {
                        color: #ffc107;
                        margin-top: 10px;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📷 摄像头实时预览</h1>
                    <div class="video-container">
                        <img src="/video_feed" alt="摄像头画面">
                    </div>
                    <br>
                    <button class="capture-btn" onclick="capturePhoto()">📸 拍照</button>
                    <p class="status" id="status"></p>
                    <p class="count" id="count"></p>
                    <p class="tips">按 Ctrl+C 停止预览 | 空格键也可拍照</p>
                </div>
                <script>
                    function capturePhoto() {
                        fetch('/capture', {method: 'POST'})
                            .then(r => r.json())
                            .then(data => {
                                document.getElementById('status').textContent = 
                                    data.success ? '✓ 已保存：' + data.filename : '✗ 失败：' + data.error;
                                if (data.success) {
                                    document.getElementById('count').textContent = 
                                        '已拍照 ' + data.count + ' 张';
                                }
                            })
                            .catch(err => {
                                document.getElementById('status').textContent = '✗ 请求失败';
                            });
                    }
                    // 空格键拍照
                    document.addEventListener('keydown', function(e) {
                        if (e.code === 'Space') {
                            e.preventDefault();
                            capturePhoto();
                        }
                    });
                </script>
            </body>
            </html>
            '''

        @app.route('/capture', methods=['POST'])
        def capture():
            """拍照接口"""
            try:
                with self._frame_lock:
                    if self._current_frame is None:
                        return jsonify({'success': False, 'error': '无图像'})
                    
                    # 保存当前帧
                    file_path = self.capture_and_save()
                    self._capture_count += 1
                
                if file_path:
                    return jsonify({
                        'success': True,
                        'filename': file_path.name,
                        'count': self._capture_count
                    })
                else:
                    return jsonify({'success': False, 'error': '保存失败'})
                    
            except Exception as e:
                logger.error(f"拍照失败：{e}")
                return jsonify({'success': False, 'error': str(e)})

        @app.route('/video_feed')
        def video_feed():
            """返回视频流"""
            def generate():
                while self._streaming:
                    with self._frame_lock:
                        if self._current_frame is not None:
                            ret, buffer = cv2.imencode(
                                '.jpg',
                                self._current_frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                            )
                            if ret:
                                frame_bytes = buffer.tobytes()
                                yield b'--frame\r\n' + \
                                      b'Content-Type: image/jpeg\r\n\r\n' + \
                                      frame_bytes + b'\r\n'
                    time.sleep(0.03)

            return Response(
                generate(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        logger.info(f"网页预览已启动：http://{host}:{port}")
        logger.info("按 Ctrl+C 停止预览")
        
        self._streaming = True
        
        # 启动帧获取线程
        frame_thread = threading.Thread(target=generate_frames, daemon=True)
        frame_thread.start()
        
        # 运行 Flask 服务器
        try:
            app.run(host=host, port=port, threaded=True)
        except Exception as e:
            logger.error(f"服务器出错：{e}")
        finally:
            self._streaming = False


def main():
    """主函数 - 演示摄像头采集功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description="摄像头采集与预览工具")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头设备索引 (默认：0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="分辨率宽度 (默认：1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="分辨率高度 (默认：1080)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="输出目录 (默认：./data)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="window",
        choices=["window", "web", "capture"],
        help="模式：window=窗口预览，web=网页预览，capture=拍照 (默认：window)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="网页预览端口 (默认：5001)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="捕获图像数量 (默认：1，仅在 capture 模式有效)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="图像格式 (默认：jpg)"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = CameraConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height
    )
    
    # 创建采集器
    capture = CameraCapture(config=config, data_dir=args.output)
    
    try:
        # 初始化摄像头
        if not capture.initialize():
            logger.error("摄像头初始化失败")
            return 1
        
        if args.mode == "window":
            # 窗口预览模式
            capture.preview_window()
            
        elif args.mode == "web":
            # 网页预览模式
            capture.preview_web(port=args.port)
            
        elif args.mode == "capture":
            # 拍照模式
            if not capture.check_disk_space():
                logger.warning("磁盘空间可能不足")
            
            saved_files = []
            for i in range(args.count):
                logger.info(f"正在捕获第 {i + 1}/{args.count} 张图像...")
                file_path = capture.capture_and_save(format=args.format)
                if file_path:
                    saved_files.append(file_path)
                
                if args.count > 1 and i < args.count - 1:
                    time.sleep(0.5)
            
            logger.info(f"成功保存 {len(saved_files)}/{args.count} 张图像")
            for f in saved_files:
                logger.info(f"  - {f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断")
        return 0
    finally:
        capture.release()


if __name__ == "__main__":
    exit(main())
