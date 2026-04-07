#!/usr/bin/env python3
"""
人脸识别模块 - InsightFace 版本 (替代方案)

使用 insightface (ArcFace) 提供更精确的人脸识别能力。
适用于 face_recognition 无法区分的相似人脸场景。

安装:
    pip install insightface onnxruntime

使用:
    from backend.face_recognition_insightface import FaceRecognizerServiceInsightFace
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

import cv2
import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("警告: insightface 未安装，请运行: pip install insightface onnxruntime")


@dataclass
class FaceRect:
    """人脸矩形框"""
    x: int
    y: int
    w: int
    h: int


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
    recognition_threshold: float = 1.0  # insightface 使用余弦距离，阈值不同
    number_of_jitters: int = 0
    # 三级判定阈值 (余弦距离)
    threshold_certain: float = 0.80
    threshold_uncertain: float = 1.00
    use_average_encoding: bool = True
    # 检测尺寸 - 较小的值可以检测到更多人脸，但可能增加误检
    det_size: tuple = (480, 480)


class FaceRecognizerServiceInsightFace:
    """人脸识别服务类 - InsightFace 版本"""

    def __init__(self, config: Optional[FaceRecognitionConfig] = None):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("insightface 未安装")
        
        self.config = config or FaceRecognitionConfig()
        self.face_database: Dict[str, FaceInfo] = {}

        # 初始化 face analysis
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=-1, det_size=self.config.det_size)

        self._load_face_database()

    def _load_face_database(self):
        """加载人脸数据库"""
        db_dir = Path(self.config.data_dir)
        if not db_dir.exists():
            return
        encoding_file = db_dir / "face_encodings_insightface.npy"
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
        """保存人脸数据库"""
        db_dir = Path(self.config.data_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        data = {}
        for face_id, info in self.face_database.items():
            data[face_id] = {
                "name": info.name,
                "encoding": info.encoding.tolist(),
                "registered_at": info.registered_at
            }
        np.save(db_dir / "face_encodings_insightface.npy", data)

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测人脸"""
        if image is None or image.size == 0:
            return []
        
        faces = self.app.get(image)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            results.append({
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "w": int(bbox[2] - bbox[0]),
                "h": int(bbox[3] - bbox[1]),
                "confidence": float(bbox[4]) if len(bbox) > 4 else 1.0,
                "embedding": face.embedding,
                "landmark": face.landmark
            })
        return results

    def extract_face_encoding(self, image: np.ndarray, face_rect=None) -> Optional[np.ndarray]:
        """提取人脸特征"""
        faces = self.app.get(image)
        if faces:
            return faces[0].embedding
        return None

    def compare_faces_with_certainty(self, encoding1: np.ndarray, encoding2: np.ndarray) -> Dict[str, Any]:
        """三级判定比较"""
        if encoding1 is None or encoding2 is None:
            return {"is_match": False, "certainty": "none", "distance": float('inf'), "similarity": 0.0}

        # 归一化
        encoding1 = encoding1 / np.linalg.norm(encoding1)
        encoding2 = encoding2 / np.linalg.norm(encoding2)
        
        # 余弦相似度
        similarity = float(np.dot(encoding1, encoding2))
        # 余弦距离 = 1 - 余弦相似度
        distance = 1.0 - similarity

        # 三级判定
        if distance < self.config.threshold_certain:
            return {"is_match": True, "certainty": "certain", "distance": distance, "similarity": similarity}
        elif distance < self.config.threshold_uncertain:
            return {"is_match": True, "certainty": "uncertain", "distance": distance, "similarity": similarity}
        else:
            return {"is_match": False, "certainty": "no_match", "distance": distance, "similarity": similarity}

    def batch_register_faces(self, images: List[np.ndarray], face_id: str, 
                              name: Optional[str] = None) -> Optional[FaceInfo]:
        """批量注册人脸"""
        encodings = []
        for image in images:
            faces = self.app.get(image)
            if faces:
                encodings.append(faces[0].embedding)

        if not encodings:
            return None

        # 平均特征
        avg_encoding = np.mean(encodings, axis=0)

        face_info = FaceInfo(
            face_id=face_id, name=name or face_id, encoding=avg_encoding,
            registered_at=datetime.now().isoformat()
        )
        self.face_database[face_id] = face_info
        self._save_face_database()
        return face_info

    def recognize_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """识别图像中的人脸"""
        faces = self.app.get(image)
        if not faces:
            return []

        results = []
        for face in faces:
            unknown_encoding = face.embedding
            bbox = face.bbox.astype(int)
            
            matches = []
            for face_id, face_info in self.face_database.items():
                result = self.compare_faces_with_certainty(unknown_encoding, face_info.encoding)
                matches.append({
                    "face_id": face_id,
                    "name": face_info.name,
                    "similarity": result["similarity"],
                    "distance": result["distance"],
                    "certainty": result["certainty"],
                    "is_match": result["is_match"]
                })

            matches = [m for m in matches if m["is_match"]]
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            results.append({
                "location": {"x": int(bbox[0]), "y": int(bbox[1]), 
                           "w": int(bbox[2] - bbox[0]), "h": int(bbox[3] - bbox[1])},
                "matches": matches,
                "best_match": matches[0] if matches else None
            })

        return results

    def clear_all_faces(self) -> bool:
        """清空人脸数据库"""
        self.face_database.clear()
        db_dir = Path(self.config.data_dir)
        encoding_file = db_dir / "face_encodings_insightface.npy"
        if encoding_file.exists():
            encoding_file.unlink()
        return True

    def get_face_list(self) -> List[Dict[str, Any]]:
        """获取已注册人脸列表"""
        return [info.to_dict() for info in self.face_database.values()]

    def delete_face(self, face_id: str) -> bool:
        """删除指定人脸"""
        if face_id not in self.face_database:
            return False
        del self.face_database[face_id]
        self._save_face_database()
        return True

    def update_face_name(self, face_id: str, new_name: str) -> bool:
        """更新人脸名称"""
        if face_id not in self.face_database:
            return False
        self.face_database[face_id].name = new_name
        self._save_face_database()
        return True


def test_insightface():
    """测试 insightface 效果"""
    if not INSIGHTFACE_AVAILABLE:
        print("❌ insightface 未安装")
        return
    
    from pathlib import Path
    
    print("=" * 60)
    print("InsightFace (ArcFace) 测试")
    print("=" * 60)
    
    service = FaceRecognizerServiceInsightFace()
    service.clear_all_faces()
    
    # 注册
    test_data = Path("test_data/human")
    for person_dir in ["jiang", "xuefen", "rabbit"]:
        person_path = test_data / person_dir
        if not person_path.exists():
            continue
        
        images = []
        for img_file in sorted(person_path.glob("*.png")):
            img = cv2.imread(str(img_file))
            if img is not None:
                images.append(img)
        
        if images:
            result = service.batch_register_faces(images, person_dir, person_dir)
            if result:
                print(f"✅ 注册 {person_dir}")
    
    # 测试跨人距离
    print(f"\n跨人距离 (余弦距离):")
    face_ids = list(service.face_database.keys())
    for i, id1 in enumerate(face_ids):
        for id2 in face_ids[i+1:]:
            enc1 = service.face_database[id1].encoding
            enc2 = service.face_database[id2].encoding
            result = service.compare_faces_with_certainty(enc1, enc2)
            print(f"  {id1} vs {id2}: {result['distance']:.4f} ({result['certainty']})")


if __name__ == "__main__":
    test_insightface()
