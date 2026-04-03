#!/usr/bin/env python3
"""DNN 人脸检测测试：正向（human）与反向（nothuman）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.face_recognition_module import FaceRecognitionConfig, FaceRecognizerService

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEST_DATA_ROOT = PROJECT_ROOT / "test_data"
HUMAN_ROOT = TEST_DATA_ROOT / "human"
NOTHUMAN_ROOT = TEST_DATA_ROOT / "nothuman"
DNN_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
TMP_DATA_ROOT = TEST_DATA_ROOT / "tmp_data"
def _collect_human_images(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        pytest.skip(f"未找到 human 测试目录: {root}")

    person_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    assert person_dirs, f"human 目录下未找到人员子目录: {root}"

    person_images: Dict[str, List[Path]] = {}
    for person_dir in person_dirs:
        images = sorted([p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        assert images, f"人员目录无图片: {person_dir}"
        person_images[person_dir.name] = images

    return person_images


def _collect_nothuman_images(root: Path) -> List[Path]:
    if not root.exists():
        pytest.skip(f"未找到 nothuman 测试目录: {root}")

    image_paths = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    assert image_paths, f"nothuman 目录下未找到任何图片: {root}"
    return image_paths


def _create_dnn_service(case_name: str) -> FaceRecognizerService:
    assert DNN_MODEL_PATH.exists(), f"未找到 DNN 模型文件: {DNN_MODEL_PATH}"

    data_dir = TMP_DATA_ROOT / case_name
    data_dir.mkdir(parents=True, exist_ok=True)

    config = FaceRecognitionConfig(
        data_dir=str(data_dir),
        detection_model="dnn",
        detection_model_path=str(DNN_MODEL_PATH),
        recognition_model="mobilenet",
    )
    return FaceRecognizerService(config=config)


def _has_face_detection(service: FaceRecognizerService, image_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    assert image is not None, f"无法读取图像: {image_path}"

    detections = service.detect_faces(image)
    return len(detections) > 0


def test_dnn_positive_human_images_should_detect_face() -> None:
    """正向测试：human 子目录下所有图片都应检测到人脸。"""
    service = _create_dnn_service("dnn_positive")
    person_images = _collect_human_images(HUMAN_ROOT)

    failed: List[str] = []
    for person_name, image_paths in person_images.items():
        for image_path in image_paths:
            if not _has_face_detection(service, image_path):
                failed.append(str(image_path.relative_to(PROJECT_ROOT)))

    assert not failed, f"正向测试失败：以下 human 图片未检测到人脸: {failed}"


def test_dnn_negative_nothuman_images_should_not_detect_face() -> None:
    """反向测试：nothuman 目录下图片都不应检测到人脸。"""
    service = _create_dnn_service("dnn_negative")
    image_paths = _collect_nothuman_images(NOTHUMAN_ROOT)

    failed: List[str] = []
    for image_path in image_paths:
        if _has_face_detection(service, image_path):
            failed.append(str(image_path.relative_to(PROJECT_ROOT)))

    assert not failed, f"反向测试失败：以下 nothuman 图片误检测到人脸: {failed}"
