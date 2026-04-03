#!/usr/bin/env python3
"""
后端服务器模块

提供 RESTful API 服务，协调摄像头采集与 AI 分析功能。
"""

import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# 导入相机模块
from backend.camera_capture import CameraCapture, CameraConfig

# 导入 AI 分析模块
from backend.ai_image_analysis import AIImageAnalyzer, AnalysisConfig, load_env_file

# 导入人脸识别模块
from backend.face_recognition_module import FaceRecognizerService, FaceRecognitionConfig

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


class FaceService:
    """人脸识别服务"""

    def __init__(self):
        self.recognizer: Optional[FaceRecognizerService] = None
        self._init_recognizer()

    def _init_recognizer(self) -> None:
        """初始化识别器"""
        try:
            config = FaceRecognitionConfig()
            self.recognizer = FaceRecognizerService(config)
        except Exception as e:
            print(f"初始化人脸识别器失败：{e}")

    def register_face(self, image_path: Path, face_id: str, name: Optional[str] = None, 
                      face_index: int = 0) -> Dict[str, Any]:
        """注册人脸"""
        if self.recognizer is None:
            return {"success": False, "error": "人脸识别器未初始化"}

        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            return {"success": False, "error": "图像文件读取失败"}

        result = self.recognizer.register_face(image, face_id, name, face_index)
        if result:
            return {
                "success": True,
                "face_id": result.face_id,
                "name": result.name,
                "location": result.location
            }
        else:
            return {"success": False, "error": "人脸注册失败"}

    def recognize_faces(self, image_path: Path) -> Dict[str, Any]:
        """识别人脸"""
        if self.recognizer is None:
            return {"success": False, "error": "人脸识别器未初始化"}

        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            return {"success": False, "error": "图像文件读取失败"}

        results = self.recognizer.recognize_faces(image)
        return {
            "success": True,
            "face_count": len(results),
            "faces": results
        }

    def compare_faces(self, image_path1: Path, image_path2: Path) -> Dict[str, Any]:
        """比对两张图片中的人脸"""
        if self.recognizer is None:
            return {"success": False, "error": "人脸识别器未初始化"}

        import cv2
        image1 = cv2.imread(str(image_path1))
        image2 = cv2.imread(str(image_path2))
        
        if image1 is None or image2 is None:
            return {"success": False, "error": "图像文件读取失败"}

        # 提取两张图片的人脸特征
        rgb_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        rgb_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        encodings1 = self.recognizer.extract_face_encodings(rgb_image1)
        encodings2 = self.recognizer.extract_face_encodings(rgb_image2)

        if len(encodings1) == 0 or len(encodings2) == 0:
            return {"success": False, "error": "未在图片中检测到人脸"}

        # 比对第一张人脸（如果有更多人脸，可以扩展逻辑）
        is_match, similarity = self.recognizer.compare_faces(encodings1[0], encodings2[0])
        
        return {
            "success": True,
            "is_match": is_match,
            "similarity": round(similarity, 4),
            "face_count_image1": len(encodings1),
            "face_count_image2": len(encodings2)
        }

    def get_face_list(self) -> List[Dict]:
        """获取人脸列表"""
        if self.recognizer is None:
            return []
        return self.recognizer.get_face_list()

    def delete_face(self, face_id: str) -> bool:
        """删除人脸"""
        if self.recognizer is None:
            return False
        return self.recognizer.delete_face(face_id)

    def clear_all_faces(self) -> bool:
        """清空所有人脸"""
        if self.recognizer is None:
            return False
        return self.recognizer.clear_all_faces()

    def update_face_name(self, face_id: str, new_name: str) -> bool:
        """更新人脸名称"""
        if self.recognizer is None:
            return False
        return self.recognizer.update_face_name(face_id, new_name)


class BackendServer:
    """后端服务器"""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.app = Flask(__name__)
        CORS(self.app, origins=self.config.cors_origins)

        # 初始化服务
        self.camera = CameraManager(data_dir=self.config.data_dir)
        self.ai_service = AIService()
        self.face_service = FaceService()

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
            try:
                data = request.get_json(silent=True) or {}
            except Exception:
                data = {}

            camera_index = data.get('camera_index', 0)
            print(f"[摄像头启动] 请求参数：{data}, 当前运行状态：{self.camera.is_running}")

            # 如果摄像头已在运行，检查是否需要切换摄像头
            if self.camera.is_running:
                # 如果索引相同，直接返回
                if data.get('camera_index') is None:
                    return jsonify({"success": True, "message": "摄像头已在运行"})
                # 如果索引不同，先停止再重启
                print(f"[摄像头启动] 检测到索引变化，停止当前摄像头...")
                self.camera.stop()

            config = CameraConfig(
                camera_index=camera_index,
                width=data.get('width', 1920),
                height=data.get('height', 1080)
            )

            print(f"[摄像头启动] 使用配置：index={camera_index}, {config.width}x{config.height}")
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

        # ===== 摄像头设备列表 =====
        @self.app.route('/camera/devices', methods=['GET'])
        def camera_devices():
            """获取可用摄像头设备列表"""
            import cv2
            devices = []

            # 尝试打开前 10 个可能的摄像头索引
            for index in range(10):
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    # 读取摄像头名称（部分系统支持）
                    ret, frame = cap.read()
                    if ret:
                        devices.append({
                            "index": index,
                            "name": f"摄像头 {index}",
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        })
                    cap.release()
                    # 等待摄像头资源释放
                    import time
                    time.sleep(0.3)
                else:
                    # 如果索引 0 都打不开，后面的大概率也没有
                    if index == 0:
                        break

            return jsonify({
                "count": len(devices),
                "devices": devices
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
            """预览页面 - 提供独立前端页面"""
            frontend_dir = Path(__file__).parent / "frontend"
            return send_from_directory(frontend_dir, "preview.html")

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

        # ===== 人脸识别 =====
        @self.app.route('/face/register', methods=['POST'])
        def face_register():
            """注册人脸"""
            data = request.get_json() or {}
            image_path = data.get('path')
            face_id = data.get('face_id')
            name = data.get('name')
            face_index = data.get('face_index', 0)

            if not image_path:
                return jsonify({"success": False, "error": "未提供图像路径"}), 400
            if not face_id:
                return jsonify({"success": False, "error": "未提供人脸 ID"}), 400

            result = self.face_service.register_face(Path(image_path), face_id, name, face_index)
            return jsonify(result)

        @self.app.route('/face/recognize', methods=['POST'])
        def face_recognize():
            """识别人脸"""
            data = request.get_json() or {}
            image_path = data.get('path')

            if not image_path:
                return jsonify({"success": False, "error": "未提供图像路径"}), 400

            result = self.face_service.recognize_faces(Path(image_path))
            return jsonify(result)

        @self.app.route('/face/compare', methods=['POST'])
        def face_compare():
            """比对两张图片中的人脸"""
            data = request.get_json() or {}
            image_path1 = data.get('path1')
            image_path2 = data.get('path2')

            if not image_path1 or not image_path2:
                return jsonify({"success": False, "error": "未提供图像路径"}), 400

            result = self.face_service.compare_faces(Path(image_path1), Path(image_path2))
            return jsonify(result)

        @self.app.route('/face/list', methods=['GET'])
        def face_list():
            """获取已注册人脸列表"""
            faces = self.face_service.get_face_list()
            return jsonify({
                "success": True,
                "count": len(faces),
                "faces": faces
            })

        @self.app.route('/face/delete', methods=['POST'])
        def face_delete():
            """删除人脸"""
            data = request.get_json() or {}
            face_id = data.get('face_id')

            if not face_id:
                return jsonify({"success": False, "error": "未提供人脸 ID"}), 400

            if self.face_service.delete_face(face_id):
                return jsonify({"success": True, "message": f"已删除人脸：{face_id}"})
            else:
                return jsonify({"success": False, "error": "人脸不存在或删除失败"}), 404

        @self.app.route('/face/clear', methods=['POST'])
        def face_clear():
            """清空所有人脸"""
            if self.face_service.clear_all_faces():
                return jsonify({"success": True, "message": "已清空所有人脸"})
            else:
                return jsonify({"success": False, "error": "清空失败"}), 500

        @self.app.route('/face/update_name', methods=['POST'])
        def face_update_name():
            """更新人脸名称"""
            data = request.get_json() or {}
            face_id = data.get('face_id')
            new_name = data.get('name')

            if not face_id or not new_name:
                return jsonify({"success": False, "error": "未提供人脸 ID 或新名称"}), 400

            if self.face_service.update_face_name(face_id, new_name):
                return jsonify({"success": True, "message": f"已更新人脸名称：{new_name}"})
            else:
                return jsonify({"success": False, "error": "人脸不存在或更新失败"}), 404

        @self.app.route('/face/capture_register', methods=['POST'])
        def face_capture_register():
            """拍照并注册人脸"""
            data = request.get_json(silent=True) or {}
            face_id = data.get('face_id')
            name = data.get('name')
            face_index = data.get('face_index', 0)

            if not face_id:
                return jsonify({"success": False, "error": "未提供人脸 ID"}), 400

            # 拍照
            file_path = self.camera.capture_photo()
            if not file_path:
                return jsonify({"success": False, "error": "拍照失败"}), 500

            # 注册人脸
            result = self.face_service.register_face(file_path, face_id, name, face_index)
            if result.get("success"):
                result["image_url"] = f"/files/{file_path.parent.name}/{file_path.name}"
            return jsonify(result)

        @self.app.route('/face/capture_recognize', methods=['POST'])
        def face_capture_recognize():
            """拍照并识别人脸"""
            # 拍照
            file_path = self.camera.capture_photo()
            if not file_path:
                return jsonify({"success": False, "error": "拍照失败"}), 500

            # 识别人脸
            result = self.face_service.recognize_faces(file_path)
            if result.get("success"):
                result["image_url"] = f"/files/{file_path.parent.name}/{file_path.name}"
            return jsonify(result)

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
                        "列表": "GET /camera/list",
                        "设备列表": "GET /camera/devices"
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
                    "人脸识别": {
                        "注册人脸": "POST /face/register",
                        "识别人脸": "POST /face/recognize",
                        "人脸比对": "POST /face/compare",
                        "人脸列表": "GET /face/list",
                        "删除人脸": "POST /face/delete",
                        "清空人脸": "POST /face/clear",
                        "更新名称": "POST /face/update_name",
                        "拍照注册": "POST /face/capture_register",
                        "拍照识别": "POST /face/capture_recognize"
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
