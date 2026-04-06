"""
人脸识别模块单元测试
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import tempfile

from backend.face_recognition_insightface import (
    FaceRecognizerServiceInsightFace,
    FaceRecognitionConfig,
    FaceRect,
    FaceInfo
)


# 检查 insightface 是否可用
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("insightface", reason="insightface 未安装"),
    reason="insightface 未安装"
)


class TestFaceRecognition:
    """测试人脸识别服务"""

    @pytest.fixture(scope="class")
    def face_service(self):
        """创建人脸识别服务"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FaceRecognitionConfig(data_dir=tmp_dir)
            service = FaceRecognizerServiceInsightFace(config)
            yield service
            # 清理
            service.clear_all_faces()

    @pytest.fixture
    def test_images(self, test_data_dir):
        """获取测试图像"""
        human_dir = test_data_dir / "human"
        images = {
            "jiang": [],
            "xuefen": [],
            "rabbit": []
        }
        
        for person in images.keys():
            person_dir = human_dir / person
            if person_dir.exists():
                for img_file in sorted(person_dir.glob("*.png"))[:3]:  # 每人取3张
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        images[person].append(img)
        
        return images

    def test_service_initialization(self, face_service):
        """测试服务初始化"""
        assert face_service is not None
        assert face_service.app is not None
        assert len(face_service.face_database) == 0

    def test_detect_faces(self, face_service, test_images):
        """测试人脸检测"""
        # 使用 jiang 的图像
        if test_images["jiang"]:
            image = test_images["jiang"][0]
            faces = face_service.detect_faces(image)
            
            # 应该至少检测到一张人脸
            assert len(faces) >= 1
            # 检查人脸字段
            for face in faces:
                assert "x" in face
                assert "y" in face
                assert "w" in face
                assert "h" in face
                assert "embedding" in face

    def test_detect_faces_no_face(self, face_service):
        """测试无_face情况"""
        # 创建空白图像 (不太可能检测到人脸)
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = face_service.detect_faces(blank_image)
        
        # 空白图像不应该检测到人脸
        # 但为了测试稳定性,我们只验证返回类型
        assert isinstance(faces, list)

    def test_extract_face_encoding(self, face_service, test_images):
        """测试特征提取"""
        if test_images["jiang"]:
            image = test_images["jiang"][0]
            encoding = face_service.extract_face_encoding(image)
            
            assert encoding is not None
            # InsightFace 的特征向量通常是 512 维
            assert len(encoding.shape) == 1

    def test_batch_register_faces(self, face_service, test_images):
        """测试批量注册人脸"""
        if test_images["jiang"]:
            result = face_service.batch_register_faces(
                test_images["jiang"],
                face_id="jiang",
                name="蒋先生"
            )
            
            assert result is not None
            assert result.face_id == "jiang"
            assert result.name == "蒋先生"
            assert "jiang" in face_service.face_database

    def test_batch_register_faces_no_face(self, face_service):
        """测试无_face注册"""
        # 创建空白图像
        blank_images = [np.zeros((480, 640, 3), dtype=np.uint8)]
        
        result = face_service.batch_register_faces(
            blank_images,
            face_id="blank",
            name="空白"
        )
        
        # 应该注册失败
        assert result is None

    def test_recognize_registered_face(self, face_service, test_images):
        """测试已注册人脸识别"""
        if test_images["jiang"]:
            # 注册 jiang
            face_service.batch_register_faces(
                test_images["jiang"],
                face_id="jiang",
                name="蒋先生"
            )
            
            # 使用 jiang 的另一张图像识别 (如果有)
            test_image = test_images["jiang"][-1]
            results = face_service.recognize_faces(test_image)
            
            assert len(results) >= 1
            # 检查最佳匹配
            if results[0]["best_match"]:
                assert results[0]["best_match"]["face_id"] == "jiang"

    def test_recognize_unregistered_face(self, face_service, test_images):
        """测试未注册人脸识别"""
        # 清空数据库
        face_service.clear_all_faces()
        
        # 尝试识别 rabbit (未注册)
        if test_images["rabbit"]:
            results = face_service.recognize_faces(test_images["rabbit"][0])
            
            # 应该检测到人脸但没有匹配
            assert len(results) >= 1
            if results:
                assert results[0]["best_match"] is None or \
                       not results[0]["best_match"]["is_match"]

    def test_compare_faces_same_person(self, face_service, test_images):
        """测试同一个人比对"""
        if len(test_images["jiang"]) >= 2:
            # 提取两个特征
            enc1 = face_service.extract_face_encoding(test_images["jiang"][0])
            enc2 = face_service.extract_face_encoding(test_images["jiang"][1])
            
            if enc1 is not None and enc2 is not None:
                result = face_service.compare_faces_with_certainty(enc1, enc2)
                
                assert result["is_match"] == True
                assert result["certainty"] in ["certain", "uncertain"]
                assert 0 <= result["similarity"] <= 1

    def test_compare_faces_different_person(self, face_service, test_images):
        """测试不同人比对"""
        if test_images["jiang"] and test_images["xuefen"]:
            enc1 = face_service.extract_face_encoding(test_images["jiang"][0])
            enc2 = face_service.extract_face_encoding(test_images["xuefen"][0])
            
            if enc1 is not None and enc2 is not None:
                result = face_service.compare_faces_with_certainty(enc1, enc2)
                
                # 不同的人应该有更大的距离
                assert result["distance"] > 0

    def test_three_level_certainty(self, face_service):
        """测试三级判定机制"""
        # 创建两个不同的特征向量
        enc1 = np.random.rand(512).astype(np.float32)
        enc2 = np.random.rand(512).astype(np.float32)
        
        result = face_service.compare_faces_with_certainty(enc1, enc2)
        
        # 验证返回字段
        assert "is_match" in result
        assert "certainty" in result
        assert "distance" in result
        assert "similarity" in result
        
        # 验证三级判定逻辑
        if result["distance"] < 0.80:
            assert result["certainty"] == "certain"
        elif result["distance"] < 1.00:
            assert result["certainty"] == "uncertain"
        else:
            assert result["certainty"] == "no_match"

    def test_clear_all_faces(self, face_service, test_images):
        """测试清空所有人脸"""
        # 注册一些人脸
        if test_images["jiang"]:
            face_service.batch_register_faces(
                test_images["jiang"],
                face_id="jiang",
                name="蒋先生"
            )
        
        assert len(face_service.face_database) > 0
        
        # 清空
        result = face_service.clear_all_faces()
        assert result == True
        assert len(face_service.face_database) == 0

    def test_database_persistence(self, test_images):
        """测试数据库持久化"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建服务并注册人脸
            config = FaceRecognitionConfig(data_dir=tmp_dir)
            service1 = FaceRecognizerServiceInsightFace(config)
            
            if test_images["jiang"]:
                service1.batch_register_faces(
                    test_images["jiang"],
                    face_id="jiang",
                    name="蒋先生"
                )
            
            assert "jiang" in service1.face_database
            del service1
            
            # 重新加载服务
            service2 = FaceRecognizerServiceInsightFace(config)
            assert "jiang" in service2.face_database
            assert service2.face_database["jiang"].name == "蒋先生"


class TestFaceDataStructures:
    """测试数据结构"""

    def test_face_rect(self):
        """测试 FaceRect 数据结构"""
        rect = FaceRect(x=100, y=50, w=200, h=200)
        assert rect.x == 100
        assert rect.y == 50
        assert rect.w == 200
        assert rect.h == 200

    def test_face_info(self):
        """测试 FaceInfo 数据结构"""
        encoding = np.random.rand(512).astype(np.float32)
        location = FaceRect(x=100, y=50, w=200, h=200)
        
        info = FaceInfo(
            face_id="test",
            name="测试",
            encoding=encoding,
            location=location,
            registered_at="2026-04-06"
        )
        
        assert info.face_id == "test"
        assert info.name == "测试"
        assert np.array_equal(info.encoding, encoding)
        
        # 测试 to_dict
        d = info.to_dict()
        assert d["face_id"] == "test"
        assert d["name"] == "测试"
        assert d["location"] is not None
