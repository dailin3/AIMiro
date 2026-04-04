#!/usr/bin/env python3
"""
人脸识别模块

使用 face_recognition 库（基于 dlib）提供人脸检测、特征提取、注册与识别能力。
https://github.com/ageitgey/face_recognition

"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import face_recognition


@dataclass
class FaceRect:
    """人脸矩形框"""
    x: int
    y: int
    w: int
    h: int


@dataclass
class FaceDetection:
    """人脸检测结果"""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    def __iter__(self):
        return iter([self.x, self.y, self.w, self.h, self.confidence])

    def __getitem__(self, index):
        return [self.x, self.y, self.w, self.h, self.confidence][index]

    def __len__(self):
        return 5


@dataclass
class FaceInfo:
    """人脸信息"""
    face_id: str
    name: str
    encoding: np.ndarray
    location: Optional[FaceRect] = None
    registered_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_id": self.face_id,
            "name": self.name,
            "location": asdict(self.location) if self.location else None,
            "registered_at": self.registered_at
        }


@dataclass
class FaceRecognitionConfig:
    """人脸识别配置"""
    data_dir: str = "./data/face_db"
    detection_model: str = "hog"  # 'hog' 或 'cnn' (兼容旧接口: 'dnn' 也接受)
    detection_model_path: str = ""  # 兼容旧接口
    recognition_model: str = "large"  # 'large' 或 'small'
    recognition_threshold: float = 0.6  # face_recognition 默认阈值
    number_of_jitters: int = 0


class FaceRecognizerService:
    """人脸识别服务类"""

    def __init__(self, config: Optional[FaceRecognitionConfig] = None):
        self.config = config or FaceRecognitionConfig()
        # 兼容旧配置：将 'dnn' 映射为 'hog'（因为测试代码中使用了 'dnn'，但 face_recognition 的 cnn 太慢）
        if self.config.detection_model == "dnn":
            self.config.detection_model = "hog"

        self.face_database: Dict[str, FaceInfo] = {}
        self._load_face_database()

    def _load_face_database(self):
        """从文件系统加载已注册的人脸"""
        db_dir = Path(self.config.data_dir)
        if not db_dir.exists():
            return
        encoding_file = db_dir / "face_encodings.npy"
        if encoding_file.exists():
            try:
                data = np.load(encoding_file, allow_pickle=True).item()
                for face_id, info in data.items():
                    self.face_database[face_id] = FaceInfo(
                        face_id=face_id,
                        name=info["name"],
                        encoding=np.array(info["encoding"]),
                        registered_at=info.get("registered_at", "")
                    )
            except Exception as e:
                print(f"加载人脸数据库失败: {e}")

    def _save_face_database(self):
        """保存人脸数据库到文件系统"""
        db_dir = Path(self.config.data_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        data = {}
        for face_id, info in self.face_database.items():
            data[face_id] = {
                "name": info.name,
                "encoding": info.encoding.tolist(),
                "registered_at": info.registered_at
            }
        np.save(db_dir / "face_encodings.npy", data)

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        检测图像中的人脸
        使用 face_recognition.face_locations (HOG)
        """
        if image is None or image.size == 0:
            return []

        rgb_image = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_image, model="hog")

        face_detections = []
        for (top, right, bottom, left) in face_locations:
            face_detections.append(FaceDetection(
                x=left, y=top, w=right - left, h=bottom - top
            ))
        return face_detections

    def extract_face_encoding(self, image: np.ndarray, face_rect) -> Optional[np.ndarray]:
        """
        提取人脸特征编码
        使用 face_recognition.face_encodings
        """
        try:
            # 支持元组输入
            if isinstance(face_rect, tuple):
                x, y, w, h = face_rect
            else:
                x, y, w, h = face_rect.x, face_rect.y, face_rect.w, face_rect.h

            rgb_image = image[:, :, ::-1]
            h_img, w_img = rgb_image.shape[:2]

            # 检查是否是全图（即未检测到人脸的情况）
            is_full_image = (w >= w_img * 0.95 and h >= h_img * 0.95)

            if is_full_image:
                # 自动检测并提取 (先用 HOG，失败后用 CNN)
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    num_jitters=self.config.number_of_jitters,
                    model=self.config.recognition_model
                )
                if encodings:
                    return np.array(encodings[0])
                
                # HOG 失败，尝试 CNN 检测
                try:
                    cnn_locations = face_recognition.face_locations(rgb_image, model="cnn")
                    if cnn_locations:
                        encodings = face_recognition.face_encodings(
                            rgb_image,
                            known_face_locations=cnn_locations,
                            num_jitters=0,
                            model=self.config.recognition_model
                        )
                        if encodings:
                            return np.array(encodings[0])
                except Exception:
                    pass
                    
                return None
            else:
                # 使用已知位置提取
                # face_recognition 格式: (top, right, bottom, left)
                face_location = [(y, x + w, y + h, x)]

                encodings = face_recognition.face_encodings(
                    rgb_image,
                    known_face_locations=face_location,
                    num_jitters=self.config.number_of_jitters,
                    model=self.config.recognition_model
                )

                if encodings:
                    return np.array(encodings[0])

                # 如果已知位置提取失败，尝试自动检测
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    num_jitters=self.config.number_of_jitters,
                    model=self.config.recognition_model
                )
                if encodings:
                    return np.array(encodings[0])

                return None
        except Exception as e:
            print(f"提取人脸特征失败: {e}")
            return None

    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> Tuple[bool, float]:
        """
        比较两个人脸编码的相似度
        使用 face_recognition.compare_faces (欧氏距离，tolerance=0.6 是推荐值)
        """
        if encoding1 is None or encoding2 is None:
            return False, 0.0

        if len(encoding1.shape) > 1:
            encoding1 = encoding1.flatten()
        if len(encoding2.shape) > 1:
            encoding2 = encoding2.flatten()

        # 使用 face_recognition.compare_faces (内部使用 face_distance)
        # tolerance=0.6 是 face_recognition 的推荐值
        is_match = face_recognition.compare_faces(
            [encoding1], encoding2, tolerance=0.6
        )[0]
        
        # 计算余弦相似度作为相似度分数
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)
        if norm1 == 0 or norm2 == 0:
            return False, 0.0
        similarity = float(np.dot(encoding1, encoding2) / (norm1 * norm2))
        similarity = max(-1.0, min(1.0, similarity))

        return is_match, similarity

    def recognize_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """识别图像中的人脸"""
        face_detections = self.detect_faces(image)
        if not face_detections:
            return []

        rgb_image = image[:, :, ::-1]
        face_locations = [(det.y, det.x + det.w, det.y + det.h, det.x) for det in face_detections]

        unknown_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            num_jitters=self.config.number_of_jitters,
            model=self.config.recognition_model
        )

        results = []
        for i, detection in enumerate(face_detections):
            if i >= len(unknown_encodings):
                continue

            unknown_encoding = unknown_encodings[i]
            matches = []
            for face_id, face_info in self.face_database.items():
                is_match, similarity = self.compare_faces(unknown_encoding, face_info.encoding)
                if is_match:
                    matches.append({
                        "face_id": face_id,
                        "name": face_info.name,
                        "similarity": similarity,
                        "is_match": is_match
                    })

            matches.sort(key=lambda x: x["similarity"], reverse=True)
            results.append({
                "location": {"x": detection.x, "y": detection.y, "w": detection.w, "h": detection.h},
                "matches": matches,
                "best_match": matches[0] if matches else None
            })

        return results

    def register_face(self, image: np.ndarray, face_id: str, name: Optional[str] = None,
                      face_index: int = 0) -> Optional[FaceInfo]:
        """注册新人脸"""
        face_detections = self.detect_faces(image)
        if not face_detections:
            return None
        if face_index >= len(face_detections):
            return None

        detection = face_detections[face_index]
        encoding = self.extract_face_encoding(image, detection)
        if encoding is None:
            return None

        face_info = FaceInfo(
            face_id=face_id, name=name or face_id, encoding=encoding,
            location=FaceRect(x=detection.x, y=detection.y, w=detection.w, h=detection.h),
            registered_at=datetime.now().isoformat()
        )
        self.face_database[face_id] = face_info
        self._save_face_database()
        return face_info

    def update_face_name(self, face_id: str, new_name: str) -> bool:
        if face_id not in self.face_database:
            return False
        self.face_database[face_id].name = new_name
        self._save_face_database()
        return True

    def delete_face(self, face_id: str) -> bool:
        if face_id not in self.face_database:
            return False
        del self.face_database[face_id]
        self._save_face_database()
        return True

    def get_face_list(self) -> List[Dict[str, Any]]:
        return [info.to_dict() for info in self.face_database.values()]

    def clear_all_faces(self) -> bool:
        self.face_database.clear()
        db_dir = Path(self.config.data_dir)
        encoding_file = db_dir / "face_encodings.npy"
        if encoding_file.exists():
            encoding_file.unlink()
        return True


def main():
    """命令行测试入口"""
    import argparse

    parser = argparse.ArgumentParser(description="人脸识别测试工具")
    parser.add_argument("--register", type=str, help="注册新人脸")
    parser.add_argument("--image", type=str, help="图像路径")
    parser.add_argument("--name", type=str, help="人脸名称")
    parser.add_argument("--recognize", action="store_true", help="识别模式")
    parser.add_argument("--list", action="store_true", help="列出人脸")
    parser.add_argument("--delete", type=str, help="删除人脸")
    parser.add_argument("--clear", action="store_true", help="清空人脸")
    args = parser.parse_args()

    service = FaceRecognizerService()

    if args.register and args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"无法读取图像: {args.image}")
            return
        result = service.register_face(img, args.register, args.name)
        if result:
            print(f"✅ 注册成功: {result.face_id} ({result.name})")
        else:
            print("❌ 注册失败")
    elif args.recognize and args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"无法读取图像: {args.image}")
            return
        results = service.recognize_faces(img)
        if not results:
            print("未检测到人脸")
        else:
            for i, r in enumerate(results):
                if r["best_match"]:
                    print(f"人脸{i+1}: {r['best_match']['name']} ({r['best_match']['similarity']:.2%})")
                else:
                    print(f"人脸{i+1}: 未知")
    elif args.list:
        faces = service.get_face_list()
        print(f"已注册 {len(faces)} 个人脸")
        for f in faces:
            print(f"  - {f['face_id']} ({f['name']})")
    elif args.delete:
        if service.delete_face(args.delete):
            print(f"✅ 已删除: {args.delete}")
        else:
            print(f"❌ 不存在: {args.delete}")
    elif args.clear:
        if service.clear_all_faces():
            print("✅ 已清空")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
