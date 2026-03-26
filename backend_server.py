#!/usr/bin/env python3
"""
后端服务器模块

提供 RESTful API 服务，协调摄像头采集与 AI 分析功能。
"""

import os
import sys
import json
import time
import signal
import base64
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# 导入相机模块
from camera_capture import CameraCapture, CameraConfig

# 导入 AI 分析模块
from ai_image_analysis import AIImageAnalyzer, AnalysisConfig, load_env_file

# 加载 .env 配置
load_env_file()


@dataclass
class ServerConfig:
    """服务器配置类"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    data_dir: str = "./data"
    cors_origins: List[str] = None

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


class CameraManager:
    """摄像头管理器"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.capture: Optional[CameraCapture] = None
        self.is_running = False
        self._current_frame: Optional[Any] = None
        self._frame_lock = threading.Lock()
        self._streaming = False
        self._capture_count = 0

    def start(self, camera_config: Optional[CameraConfig] = None) -> bool:
        """启动摄像头"""
        if self.is_running:
            return True

        try:
            self.capture = CameraCapture(
                config=camera_config or CameraConfig(),
                data_dir=str(self.data_dir)
            )

            if not self.capture.initialize():
                return False

            self.is_running = True
            self._streaming = True

            # 启动后台帧获取线程
            frame_thread = threading.Thread(target=self._generate_frames, daemon=True)
            frame_thread.start()

            return True
        except Exception as e:
            print(f"启动摄像头失败: {e}")
            return False

    def stop(self) -> None:
        """停止摄像头"""
        self._streaming = False
        self.is_running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def _generate_frames(self) -> None:
        """后台线程：持续获取最新帧"""
        while self._streaming and self.capture:
            frame = self.capture.capture_frame()
            if frame is not None:
                with self._frame_lock:
                    self._current_frame = frame
            time.sleep(0.033)  # 约 30fps

    def get_current_frame(self) -> Optional[Any]:
        """获取当前帧"""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def capture_photo(self, auto_start: bool = True) -> Optional[Path]:
        """拍照

        Args:
            auto_start: 如果摄像头未运行，是否自动启动

        Returns:
            保存的文件路径，失败返回 None
        """
        # 如果未运行且允许自动启动，则启动摄像头
        if not self.is_running:
            if auto_start:
                if not self.start():
                    return None
            else:
                return None

        # 等待获取第一帧
        for _ in range(30):  # 最多等待 1 秒
            frame = self.get_current_frame()
            if frame is not None:
                break
            time.sleep(0.033)

        frame = self.get_current_frame()
        if frame is None:
            return None

        file_path = self.capture.save_image(frame)
        if file_path:
            self._capture_count += 1
        return file_path

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "is_running": self.is_running,
            "capture_count": self._capture_count,
            "has_frame": self._current_frame is not None
        }


class AIService:
    """AI 分析服务"""

    def __init__(self):
        self.analyzer: Optional[AIImageAnalyzer] = None
        self.results: Dict[str, Dict] = {}
        self._init_analyzer()

    def _init_analyzer(self) -> None:
        """初始化分析器"""
        try:
            config = AnalysisConfig()
            self.analyzer = AIImageAnalyzer(config)
        except Exception as e:
            print(f"初始化 AI 分析器失败: {e}")

    def analyze_image(self, image_path: Path, prompt: Optional[str] = None) -> Dict[str, Any]:
        """分析单张图像"""
        if self.analyzer is None:
            return {"success": False, "error": "AI 分析器未初始化"}

        if not image_path.exists():
            return {"success": False, "error": "图像文件不存在"}

        result = self.analyzer.analyze(image_path, prompt=prompt)

        # 保存结果到内存
        if result.get("success"):
            result_id = f"{image_path.stem}_{int(time.time())}"
            self.results[result_id] = {
                "id": result_id,
                "image_path": str(image_path),
                "analysis": result.get("analysis"),
                "model": result.get("model"),
                "usage": result.get("usage"),
                "timestamp": datetime.now().isoformat()
            }
            result["result_id"] = result_id

        return result

    def get_result(self, result_id: str) -> Optional[Dict]:
        """获取分析结果"""
        return self.results.get(result_id)

    def get_history(self, limit: int = 50) -> List[Dict]:
        """获取分析历史"""
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        return sorted_results[:limit]


class BackendServer:
    """后端服务器"""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.app = Flask(__name__)
        CORS(self.app, origins=self.config.cors_origins)

        # 初始化服务
        self.camera = CameraManager(data_dir=self.config.data_dir)
        self.ai_service = AIService()

        # 注册路由
        self._register_routes()

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理"""
        print(f"\n收到信号 {signum}，正在关闭服务器...")
        self.camera.stop()
        sys.exit(0)

    def _register_routes(self) -> None:
        """注册路由"""

        # ===== 健康检查 =====
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查接口"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "camera_status": self.camera.get_status(),
                "ai_service_ready": self.ai_service.analyzer is not None
            })

        # ===== 摄像头控制 =====
        @self.app.route('/camera/start', methods=['POST'])
        def camera_start():
            """启动摄像头"""
            if self.camera.is_running:
                return jsonify({"success": True, "message": "摄像头已在运行"})

            data = request.get_json() or {}
            config = CameraConfig(
                camera_index=data.get('camera_index', 0),
                width=data.get('width', 1920),
                height=data.get('height', 1080)
            )

            if self.camera.start(config):
                return jsonify({"success": True, "message": "摄像头启动成功"})
            else:
                return jsonify({"success": False, "error": "摄像头启动失败"}), 500

        @self.app.route('/camera/stop', methods=['POST'])
        def camera_stop():
            """停止摄像头"""
            self.camera.stop()
            return jsonify({"success": True, "message": "摄像头已停止"})

        @self.app.route('/camera/status', methods=['GET'])
        def camera_status():
            """获取摄像头状态"""
            return jsonify(self.camera.get_status())

        @self.app.route('/camera/capture', methods=['POST'])
        def camera_capture():
            """拍照"""
            file_path = self.camera.capture_photo()
            if file_path:
                return jsonify({
                    "success": True,
                    "filename": file_path.name,
                    "path": str(file_path),
                    "url": f"/files/{file_path.parent.name}/{file_path.name}"
                })
            else:
                return jsonify({"success": False, "error": "拍照失败"}), 500

        @self.app.route('/camera/list', methods=['GET'])
        def camera_list():
            """获取已存储的图像列表"""
            pic_dir = Path(self.config.data_dir) / "pic"
            images = []

            if pic_dir.exists():
                for date_dir in sorted(pic_dir.iterdir(), reverse=True):
                    if date_dir.is_dir():
                        for img_file in sorted(date_dir.iterdir(), reverse=True):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                images.append({
                                    "filename": img_file.name,
                                    "date": date_dir.name,
                                    "path": str(img_file),
                                    "url": f"/files/{date_dir.name}/{img_file.name}",
                                    "created": datetime.fromtimestamp(img_file.stat().st_mtime).isoformat()
                                })

            return jsonify({
                "count": len(images),
                "images": images[:100]  # 限制返回数量
            })

        # ===== 视频流 =====
        @self.app.route('/video_feed')
        def video_feed():
            """视频流"""
            def generate():
                import cv2
                while True:
                    frame = self.camera.get_current_frame()
                    if frame is not None:
                        ret, buffer = cv2.imencode(
                            '.jpg', frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        )
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield b'--frame\r\n' + \
                                  b'Content-Type: image/jpeg\r\n\r\n' + \
                                  frame_bytes + b'\r\n'
                    time.sleep(0.033)

            return Response(
                generate(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        # ===== 网页预览 =====
        @self.app.route('/preview')
        def preview_page():
            """预览页面"""
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>摄像头预览</title>
                <meta charset="utf-8">
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
                        background: #000;
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    img {
                        max-width: 100%;
                        max-height: 70vh;
                        display: block;
                    }
                    .btn-group {
                        margin-top: 20px;
                        display: flex;
                        gap: 15px;
                        justify-content: center;
                    }
                    .btn {
                        padding: 15px 40px;
                        font-size: 18px;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        transition: all 0.2s;
                    }
                    .btn-capture {
                        background: #007bff;
                        color: white;
                    }
                    .btn-capture:hover {
                        background: #0056b3;
                    }
                    .btn-analyze {
                        background: #28a745;
                        color: white;
                    }
                    .btn-analyze:hover {
                        background: #1e7e34;
                    }
                    .btn:active {
                        transform: scale(0.98);
                    }
                    .status {
                        color: #28a745;
                        margin-top: 15px;
                        font-size: 16px;
                        min-height: 24px;
                    }
                    .result {
                        margin-top: 20px;
                        padding: 20px;
                        background: #2a2a2a;
                        border-radius: 8px;
                        max-width: 800px;
                        color: #ddd;
                        text-align: left;
                        white-space: pre-wrap;
                        max-height: 300px;
                        overflow-y: auto;
                    }
                    .tips {
                        color: #888;
                        margin-top: 15px;
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
                    <div class="btn-group">
                        <button class="btn btn-capture" onclick="capturePhoto()">📸 拍照</button>
                        <button class="btn btn-analyze" onclick="captureAndAnalyze()">🔬 拍照并分析</button>
                    </div>
                    <p class="status" id="status"></p>
                    <div class="result" id="result" style="display: none;"></div>
                    <p class="tips">按空格键拍照 | 图片保存在 data/pic/ 目录</p>
                </div>
                <script>
                    function capturePhoto() {
                        document.getElementById('status').textContent = '正在拍照...';
                        fetch('/camera/capture', {method: 'POST'})
                            .then(r => r.json())
                            .then(data => {
                                if (data.success) {
                                    document.getElementById('status').innerHTML =
                                        '✓ 已保存：<a href="' + data.url + '" target="_blank" style="color: #28a745;">' +
                                        data.filename + '</a>';
                                } else {
                                    document.getElementById('status').textContent = '✗ 失败：' + data.error;
                                }
                            })
                            .catch(err => {
                                document.getElementById('status').textContent = '✗ 请求失败';
                            });
                    }

                    function captureAndAnalyze() {
                        document.getElementById('status').textContent = '正在拍照并分析...';
                        document.getElementById('result').style.display = 'none';

                        fetch('/camera/capture', {method: 'POST'})
                            .then(r => r.json())
                            .then(data => {
                                if (data.success) {
                                    document.getElementById('status').textContent = '正在 AI 分析...';
                                    // 调用分析接口
                                    fetch('/analyze/image', {
                                        method: 'POST',
                                        headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify({path: data.path})
                                    })
                                    .then(r => r.json())
                                    .then(analysis => {
                                        if (analysis.success) {
                                            document.getElementById('status').innerHTML =
                                                '✓ 分析完成：<a href="' + data.url + '" target="_blank" style="color: #28a745;">' +
                                                data.filename + '</a>';
                                            document.getElementById('result').textContent = analysis.analysis;
                                            document.getElementById('result').style.display = 'block';
                                        } else {
                                            document.getElementById('status').textContent = '✗ 分析失败：' + analysis.error;
                                        }
                                    });
                                } else {
                                    document.getElementById('status').textContent = '✗ 拍照失败：' + data.error;
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

        # ===== AI 分析 =====
        @self.app.route('/analyze/image', methods=['POST'])
        def analyze_image():
            """分析单张图像"""
            data = request.get_json() or {}
            image_path = data.get('path')
            prompt = data.get('prompt')  # 可选自定义提示词

            if not image_path:
                return jsonify({"success": False, "error": "未提供图像路径"}), 400

            result = self.ai_service.analyze_image(Path(image_path), prompt)
            return jsonify(result)

        @self.app.route('/analyze/results/<result_id>', methods=['GET'])
        def get_result(result_id):
            """获取分析结果"""
            result = self.ai_service.get_result(result_id)
            if result:
                return jsonify({"success": True, "result": result})
            else:
                return jsonify({"success": False, "error": "结果不存在"}), 404

        @self.app.route('/analyze/history', methods=['GET'])
        def analyze_history():
            """获取分析历史"""
            limit = request.args.get('limit', 50, type=int)
            history = self.ai_service.get_history(limit)
            return jsonify({"success": True, "count": len(history), "history": history})

        # ===== 文件服务 =====
        @self.app.route('/files/<date>/<filename>')
        def serve_image(date, filename):
            """提供图像文件"""
            pic_dir = Path(self.config.data_dir) / "pic" / date
            if pic_dir.exists():
                return send_from_directory(pic_dir, filename)
            return jsonify({"error": "文件不存在"}), 404

        @self.app.route('/files/<date>/<filename>', methods=['DELETE'])
        def delete_image(date, filename):
            """删除图像文件"""
            file_path = Path(self.config.data_dir) / "pic" / date / filename
            if file_path.exists():
                file_path.unlink()
                return jsonify({"success": True, "message": "文件已删除"})
            return jsonify({"success": False, "error": "文件不存在"}), 404

        # ===== 首页 =====
        @self.app.route('/')
        def index():
            """API 首页"""
            return jsonify({
                "name": "AI 图像分析后端服务",
                "version": "1.0.0",
                "endpoints": {
                    "健康检查": "/health",
                    "摄像头控制": {
                        "启动": "POST /camera/start",
                        "停止": "POST /camera/stop",
                        "状态": "GET /camera/status",
                        "拍照": "POST /camera/capture",
                        "列表": "GET /camera/list"
                    },
                    "视频预览": {
                        "预览页面": "/preview",
                        "视频流": "/video_feed"
                    },
                    "AI 分析": {
                        "分析图像": "POST /analyze/image",
                        "获取结果": "GET /analyze/results/<id>",
                        "历史记录": "GET /analyze/history"
                    },
                    "文件服务": {
                        "获取文件": "GET /files/<date>/<filename>",
                        "删除文件": "DELETE /files/<date>/<filename>"
                    }
                }
            })

    def run(self) -> None:
        """运行服务器"""
        print(f"=" * 50)
        print(f"🚀 AI 图像分析后端服务启动")
        print(f"=" * 50)
        print(f"📡 API 地址: http://{self.config.host}:{self.config.port}")
        print(f"📷 预览页面: http://{self.config.host}:{self.config.port}/preview")
        print(f"💾 数据目录: {self.config.data_dir}")
        print(f"=" * 50)
        print(f"按 Ctrl+C 停止服务器")
        print()

        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            threaded=True
        )


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="AI 图像分析后端服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--data-dir", default="./data", help="数据目录")

    args = parser.parse_args()

    config = ServerConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        data_dir=args.data_dir
    )

    server = BackendServer(config)
    server.run()


if __name__ == "__main__":
    main()
