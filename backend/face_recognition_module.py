#!/usr/bin/env python3
"""
人脸识别模块（OpenCV 版本）

提供人脸检测、特征提取、注册与识别能力。

特征提取策略：
1) 优先使用 MobileNet ONNX（如果模型文件可用）
2) 否则回退到简单灰度特征（保证流程可运行）
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import cv2

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FaceRect = Tuple[int, int, int, int]
FaceDetection = Tuple[int, int, int, int, float]


@dataclass
class FaceInfo:
    """人脸信息类"""
    face_id: str
    name: Optional[str] = None
    encoding: Optional[List[float]] = None  # 特征向量
    image_path: Optional[str] = None
    location: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FaceRecognitionConfig:
    """人脸识别配置类"""
    data_dir: str = "./data/faces"
    detection_model: str = "dnn"  # 检测模型：dnn (深度学习) 或 haar (Haar 级联)
    detection_model_path: Optional[str] = None  # DNN 人脸检测模型路径（如 YuNet ONNX）
    recognition_model: str = "mobilenet"  # 识别模型：mobilenet（ONNX 特征模型）
    mobilenet_model_path: Optional[str] = "./models/mobilenetv2-7.onnx"  # MobileNet ONNX 模型路径
    confidence_threshold: float = 0.5  # 检测置信度阈值
    recognition_threshold: float = 0.6  # 识别阈值，越小越严格


class FaceDetector:
    """人脸检测器"""
    
    def __init__(self, model_type: str = "dnn", model_path: Optional[str] = None):
        self.model_type = model_type
        self.model_path = model_path
        self.detector = None
        self._init_detector()
    
    def _init_detector(self):
        """初始化检测器"""
        if self.model_type == "dnn":
            candidate_paths = []
            if self.model_path:
                candidate_paths.append(Path(self.model_path))

            workspace_root = Path(__file__).resolve().parent.parent
            candidate_paths.extend(
                [
                    workspace_root / "models" / "face_detection_yunet_2023mar.onnx",
                    workspace_root / "models" / "yunet.onnx",
                ]
            )

            model_file = next((p for p in candidate_paths if p.exists()), None)
            if model_file is None:
                logger.info("未找到 DNN 人脸检测模型，回退到 Haar 检测")
                self.model_type = "haar"
                self._init_detector()
                return

            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=str(model_file),
                    config="",
                    input_size=(320, 320),
                )
                logger.info(f"使用 DNN 人脸检测器：{model_file}")
            except Exception as e:
                logger.warning(f"DNN 检测器初始化失败：{e}, 回退到 Haar 检测")
                self.model_type = "haar"
                self._init_detector()
        else:
            # 使用 Haar 级联分类器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                logger.warning("无法加载 Haar 级联分类器")
            else:
                logger.info("使用 Haar 级联人脸检测器")
    
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        检测人脸
        
        Args:
            image: BGR 格式图像
            
        Returns:
            人脸列表 [(x, y, w, h, confidence), ...]
        """
        if self.model_type == "dnn":
            return self._detect_dnn(image)
        else:
            return self._detect_haar(image)
    
    def _detect_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """DNN 检测"""
        try:
            if self.detector is None:
                return self._detect_haar(image)

            h, w = image.shape[:2]
            self.detector.setInputSize((w, h))

            detect_result = self.detector.detect(image)
            results = self._parse_dnn_result(detect_result)

            logger.info(f"DNN 检测到 {len(results)} 张人脸")
            return results
        except Exception as e:
            logger.warning(f"DNN 检测失败：{e}, 回退到 Haar 检测")
            return self._detect_haar(image)

    @staticmethod
    def _parse_dnn_result(detect_result: Any) -> List[FaceDetection]:
        """兼容不同 OpenCV 版本的 FaceDetectorYN.detect 返回格式。"""
        if detect_result is None:
            return []

        # 常见返回: (retval, faces)
        if isinstance(detect_result, tuple):
            if len(detect_result) < 2 or detect_result[1] is None:
                return []
            faces = detect_result[1]
        else:
            faces = detect_result

        results: List[FaceDetection] = []
        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[4]) if len(face) > 4 else 0.0
            results.append((x, y, w, h, confidence))
        return results
    
    def _detect_haar(self, image: np.ndarray) -> List[FaceDetection]:
        """Haar 检测"""
        image = self._to_uint8(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        results = []
        for (x, y, w, h) in faces:
            results.append((x, y, w, h, 0.8))  # Haar 没有置信度，假设 0.8

        logger.info(f"Haar 检测到 {len(results)} 张人脸")
        return results

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        return np.clip(image * 255, 0, 255).astype(np.uint8)


class FaceRecognizer:
    """人脸识别器（使用 OpenCV DNN）"""

    def __init__(self, model_type: str = "mobilenet", model_path: Optional[str] = None):
        self.model_type = model_type
        self.model_path = model_path
        self.net = None
        self.input_size = (224, 224)
        self._init_model()
    
    def _init_model(self):
        """初始化识别模型"""
        try:
            if self.model_type == "mobilenet":
                candidate_paths = []
                if self.model_path:
                    candidate_paths.append(Path(self.model_path))
                workspace_root = Path(__file__).resolve().parent.parent
                candidate_paths.extend(
                    [
                        workspace_root / "models" / "mobilenetv2-7.onnx",
                        workspace_root / "models" / "mobilenetv2.onnx",
                        workspace_root / "models" / "mobilenet_v2.onnx",
                        workspace_root / "models" / "mobilenet.onnx",
                        workspace_root / "data" / "models" / "mobilenetv2.onnx",
                        workspace_root / "data" / "models" / "mobilenet_v2.onnx",
                        workspace_root / "data" / "models" / "mobilenet.onnx",
                    ]
                )

                models_dir = workspace_root / "models"
                if models_dir.exists():
                    candidate_paths.extend(sorted(models_dir.glob("*.onnx")))

                model_file = next((p for p in candidate_paths if p.exists()), None)

                if model_file is not None:
                    self.net = cv2.dnn.readNetFromONNX(str(model_file))
                    logger.info(f"使用 MobileNet ONNX 特征模型：{model_file}")
                else:
                    self.net = None
                    logger.warning(
                        "未提供可用的 MobileNet ONNX 模型，回退到简单特征提取器。"
                        "请将模型放到 ./models 并配置 mobilenet_model_path。"
                    )
            else:
                self.net = None
                logger.info(f"不支持的识别模型 {self.model_type}，回退到简单特征提取器")
        except Exception as e:
            logger.warning(f"初始化识别模型失败：{e}")
            self.net = None
    
    def extract_features(self, image: np.ndarray, face_rect: FaceRect) -> Optional[np.ndarray]:
        """
        提取人脸特征
        
        Args:
            image: BGR 格式图像
            face_rect: 人脸矩形 (x, y, w, h)
            
        Returns:
            特征向量
        """
        x, y, w, h = self._clip_face_rect(image, face_rect)
        if w <= 0 or h <= 0:
            return None

        face_roi = image[y:y+h, x:x+w]
        if face_roi.size == 0:
            return None
        
        # 优先使用 MobileNet 特征，失败时回退到简单特征
        if self.net is not None and self.model_type == "mobilenet":
            feature = self._extract_mobilenet_features(face_roi)
            if feature is not None:
                return feature

        face_roi = cv2.resize(face_roi, (112, 112))
        face_roi = self._to_uint8(face_roi)

        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.equalizeHist(gray_face)

        feature = self._extract_simple_features(gray_face)

        return feature

    @staticmethod
    def _clip_face_rect(image: np.ndarray, face_rect: FaceRect) -> FaceRect:
        x, y, w, h = face_rect
        h_img, w_img = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        return x, y, w, h

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        return np.clip(image * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _normalize_feature(feature: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(feature)
        if norm > 0:
            return feature / norm
        return feature

    def _extract_mobilenet_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """使用 MobileNet ONNX 模型提取特征向量。"""
        try:
            face_image = self._to_uint8(face_image)

            resized = cv2.resize(face_image, self.input_size)
            blob = cv2.dnn.blobFromImage(
                resized,
                scalefactor=1.0 / 127.5,
                size=self.input_size,
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
                crop=False,
            )
            self.net.setInput(blob)
            output = self.net.forward()

            feature = output.reshape(-1).astype(np.float32)
            feature = self._normalize_feature(feature)
            return feature
        except Exception as e:
            logger.warning(f"MobileNet 特征提取失败，回退到简单特征：{e}")
            return None
    
    def _extract_simple_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        提取特征（直接像素 + 归一化）

        对于对齐良好的人脸，原始像素是最有效的特征
        """
        size = 64
        face_image = cv2.resize(face_image, (size, size))
        face_image = self._to_uint8(face_image)
        
        face_image = cv2.GaussianBlur(face_image, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_image = clahe.apply(face_image)

        face_float = face_image.astype(np.float64) / 255.0

        face_small = cv2.resize(face_float, (32, 32), interpolation=cv2.INTER_AREA)

        feature = face_small.flatten()
        return self._normalize_feature(feature)


class FaceDatabase:
    """人脸特征数据库"""

    def __init__(self, db_path: str = "./data/faces/face_db.pkl"):
        self.db_path = Path(db_path)
        self.faces: Dict[str, FaceInfo] = {}
        self._load()

    def _load(self) -> bool:
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    self.faces = pickle.load(f)
                logger.info(f"已加载人脸数据库：{len(self.faces)} 个人脸")
                return True
            except Exception as e:
                logger.error(f"加载数据库失败：{e}")
                self.faces = {}
                return False
        logger.info("人脸数据库不存在，将创建新数据库")
        return True

    def _save(self) -> bool:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "wb") as f:
                pickle.dump(self.faces, f)
            logger.info(f"已保存人脸数据库：{len(self.faces)} 个人脸")
            return True
        except Exception as e:
            logger.error(f"保存数据库失败：{e}")
            return False

    def add_face(self, face_info: FaceInfo) -> bool:
        self.faces[face_info.face_id] = face_info
        return self._save()

    def remove_face(self, face_id: str) -> bool:
        if face_id in self.faces:
            del self.faces[face_id]
            return self._save()
        return False

    def get_face(self, face_id: str) -> Optional[FaceInfo]:
        return self.faces.get(face_id)

    def get_all_faces(self) -> List[FaceInfo]:
        return list(self.faces.values())

    def get_face_count(self) -> int:
        return len(self.faces)

    def clear(self) -> bool:
        self.faces = {}
        return self._save()


class FaceRecognizerService:
    """人脸识别服务"""

    def __init__(self, config: Optional[FaceRecognitionConfig] = None):
        self.config = config or FaceRecognitionConfig()
        self.database = FaceDatabase(db_path=Path(self.config.data_dir) / "face_db.pkl")
        self.detector = FaceDetector(
            model_type=self.config.detection_model,
            model_path=self.config.detection_model_path,
        )
        self.recognizer = FaceRecognizer(
            model_type=self.config.recognition_model,
            model_path=self.config.mobilenet_model_path,
        )
        
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        检测图像中的人脸位置
        
        Args:
            image: BGR 格式的图像数组
            
        Returns:
            人脸位置列表 [(x, y, w, h, confidence), ...]
        """
        return self.detector.detect(image)

    def extract_face_encoding(self, image: np.ndarray, face_rect: FaceRect) -> Optional[np.ndarray]:
        """
        提取人脸特征编码
        
        Args:
            image: BGR 格式的图像数组
            face_rect: 人脸位置 (x, y, w, h)
            
        Returns:
            人脸特征编码，失败返回 None
        """
        return self.recognizer.extract_features(image, face_rect)

    @staticmethod
    def _fallback_full_image_face(image: np.ndarray) -> List[FaceDetection]:
        h, w = image.shape[:2]
        logger.info("未检测到人脸，使用整张图像作为人脸区域")
        return [(0, 0, w, h, 0.9)]

    def _detect_with_fallback(self, image: np.ndarray) -> List[FaceDetection]:
        detections = self.detect_faces(image)
        if detections:
            return detections
        return self._fallback_full_image_face(image)

    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> Tuple[bool, float]:
        """
        比较两个人脸特征

        使用余弦相似度（对于 L2 归一化的特征，等于点积）
        """
        if encoding1 is None or encoding2 is None:
            return False, 0.0

        # 计算余弦相似度
        dot_product = np.dot(encoding1, encoding2)
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)

        if norm1 == 0 or norm2 == 0:
            return False, 0.0

        # 余弦相似度
        similarity = dot_product / (norm1 * norm2)
        similarity = max(0.0, min(1.0, similarity))
        
        # 阈值判断
        is_match = similarity > self.config.recognition_threshold
        
        return is_match, float(similarity)

    def recognize_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        识别图像中的人脸

        Args:
            image: BGR 格式的图像数组

        Returns:
            识别结果列表
        """
        face_detections = self._detect_with_fallback(image)

        # 获取已知人脸
        known_faces = self.database.get_all_faces()
        known_faces = [f for f in known_faces if f.encoding is not None]

        results = []

        for i, (x, y, w, h, confidence) in enumerate(face_detections):
            face_rect: FaceRect = (x, y, w, h)

            # 提取特征
            encoding = self.extract_face_encoding(image, face_rect)

            if encoding is None:
                continue

            result = {
                "face_index": i,
                "location": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "confidence": confidence,
                "matches": []
            }

            # 与已知人脸比对
            for known_face in known_faces:
                known_encoding = np.array(known_face.encoding)
                is_match, similarity = self.compare_faces(encoding, known_encoding)
                
                if is_match:
                    result["matches"].append({
                        "face_id": known_face.face_id,
                        "name": known_face.name,
                        "similarity": round(similarity, 4)
                    })
            
            # 按相似度排序
            result["matches"].sort(key=lambda x: x["similarity"], reverse=True)
            results.append(result)
        
        return results

    def register_face(self, image: np.ndarray, face_id: str, name: Optional[str] = None,
                      face_index: int = 0) -> Optional[FaceInfo]:
        """
        注册新人脸

        Args:
            image: BGR 格式的图像数组
            face_id: 人脸唯一标识
            name: 人脸名称（可选）
            face_index: 如果图像中有多张人脸，指定注册哪一张

        Returns:
            注册的人脸信息，失败返回 None
        """
        face_detections = self._detect_with_fallback(image)

        if face_index >= len(face_detections):
            logger.error(f"指定的人脸索引 {face_index} 超出范围 (0-{len(face_detections)-1})")
            return None

        # 提取特征
        x, y, w, h, _confidence = face_detections[face_index]
        face_rect: FaceRect = (x, y, w, h)
        encoding = self.extract_face_encoding(image, face_rect)
        
        if encoding is None:
            logger.error("无法提取人脸特征")
            return None
        
        # 创建人脸信息
        face_info = FaceInfo(
            face_id=face_id,
            name=name,
            encoding=encoding.tolist(),
            location=face_rect
        )
        
        # 保存到数据库
        if self.database.add_face(face_info):
            logger.info(f"已注册人脸：{face_id} ({name})")
            return face_info
        
        return None

    def update_face_name(self, face_id: str, new_name: str) -> bool:
        face_info = self.database.get_face(face_id)
        if face_info:
            face_info.name = new_name
            return self.database.add_face(face_info)
        return False

    def delete_face(self, face_id: str) -> bool:
        return self.database.remove_face(face_id)

    def get_face_list(self) -> List[Dict[str, Any]]:
        faces = self.database.get_all_faces()
        return [
            {
                "face_id": f.face_id,
                "name": f.name,
                "image_path": f.image_path,
                "has_encoding": f.encoding is not None
            }
            for f in faces
        ]

    def clear_all_faces(self) -> bool:
        return self.database.clear()


def main():
    """命令行测试入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="人脸识别测试工具 (OpenCV 版本)")
    parser.add_argument("--register", type=str, help="注册新人脸，提供人脸 ID")
    parser.add_argument("--image", type=str, help="图像文件路径")
    parser.add_argument("--name", type=str, help="人脸名称")
    parser.add_argument("--recognize", action="store_true", help="识别人脸模式")
    parser.add_argument("--compare", action="store_true", help="比较两张图的人脸相似度")
    parser.add_argument("--image1", type=str, help="第一张图像路径（用于 compare）")
    parser.add_argument("--image2", type=str, help="第二张图像路径（用于 compare）")
    parser.add_argument("--face-index1", type=int, default=0, help="第一张图的人脸索引（默认 0）")
    parser.add_argument("--face-index2", type=int, default=0, help="第二张图的人脸索引（默认 0）")
    parser.add_argument("--list", action="store_true", help="列出已注册人脸")
    parser.add_argument("--delete", type=str, help="删除指定人脸 ID")
    parser.add_argument("--clear", action="store_true", help="清空所有人脸")
    
    args = parser.parse_args()
    
    service = FaceRecognizerService()
    
    if args.list:
        faces = service.get_face_list()
        print(f"已注册 {len(faces)} 个人脸:")
        for f in faces:
            print(f"  - {f['face_id']} ({f['name'] or '未命名'})")
        return
    
    if args.delete:
        if service.delete_face(args.delete):
            print(f"已删除人脸：{args.delete}")
        else:
            print(f"人脸不存在：{args.delete}")
        return
    
    if args.clear:
        if service.clear_all_faces():
            print("已清空所有人脸")
        return
    
    if args.register and args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"无法读取图像：{args.image}")
            return
        
        result = service.register_face(image, args.register, args.name)
        if result:
            print(f"人脸注册成功:")
            print(f"  ID: {result.face_id}")
            print(f"  名称：{result.name}")
            print(f"  位置：{result.location}")
        else:
            print("人脸注册失败")
        return

    if args.compare and args.image1 and args.image2:
        image1 = cv2.imread(args.image1)
        if image1 is None:
            print(f"无法读取第一张图像：{args.image1}")
            return

        image2 = cv2.imread(args.image2)
        if image2 is None:
            print(f"无法读取第二张图像：{args.image2}")
            return

        detections1 = service._detect_with_fallback(image1)
        detections2 = service._detect_with_fallback(image2)

        if args.face_index1 >= len(detections1):
            print(f"第一张图的人脸索引 {args.face_index1} 超出范围 (0-{len(detections1)-1})")
            return
        if args.face_index2 >= len(detections2):
            print(f"第二张图的人脸索引 {args.face_index2} 超出范围 (0-{len(detections2)-1})")
            return

        x1, y1, w1, h1, _c1 = detections1[args.face_index1]
        x2, y2, w2, h2, _c2 = detections2[args.face_index2]

        enc1 = service.extract_face_encoding(image1, (x1, y1, w1, h1))
        enc2 = service.extract_face_encoding(image2, (x2, y2, w2, h2))

        if enc1 is None or enc2 is None:
            print("无法提取人脸特征，比较失败")
            return

        is_match, similarity = service.compare_faces(enc1, enc2)
        print("人脸对比结果:")
        print(f"  图1人脸索引: {args.face_index1}, 位置: {(x1, y1, w1, h1)}")
        print(f"  图2人脸索引: {args.face_index2}, 位置: {(x2, y2, w2, h2)}")
        print(f"  相似度: {similarity:.4f}")
        print(f"  阈值: {service.config.recognition_threshold:.4f}")
        print(f"  是否同一人: {'是' if is_match else '否'}")
        return
    
    if args.recognize and args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"无法读取图像：{args.image}")
            return
        
        results = service.recognize_faces(image)
        print(f"检测到 {len(results)} 张人脸:")
        for r in results:
            print(f"  人脸 {r['face_index']}: 位置 {r['location']}, 置信度 {r['confidence']:.2f}")
            if r['matches']:
                for m in r['matches']:
                    print(f"    - 匹配：{m['name'] or m['face_id']} (相似度：{m['similarity']:.2%})")
            else:
                print("    - 未匹配到已知人脸")
        return
    
    print("请使用 --help 查看用法")


if __name__ == "__main__":
    main()
