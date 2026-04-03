#!/usr/bin/env python3
"""测试二：不同人的人脸应被区分。"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.face_recognition_module import FaceRecognitionConfig, FaceRecognizerService

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEST_DATA_ROOT = PROJECT_ROOT / "test_data"
EXPECTED_PERSON_COUNT = 3


def _collect_person_images(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        pytest.skip(f"未找到测试数据目录: {root}")

    person_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    assert len(person_dirs) == EXPECTED_PERSON_COUNT, (
        f"期望 {EXPECTED_PERSON_COUNT} 个人员目录，实际是 {len(person_dirs)} 个: "
        f"{[p.name for p in person_dirs]}"
    )

    mapping: Dict[str, List[Path]] = {}
    for person_dir in person_dirs:
        images = sorted([p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        assert len(images) >= 1, f"人员目录 {person_dir} 至少需要 1 张人脸图"
        mapping[person_dir.name] = images

    return mapping


def _extract_encoding(service: FaceRecognizerService, image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    assert image is not None, f"无法读取图像: {image_path}"

    detections = service.detect_faces(image)
    if detections:
        x, y, w, h, _ = max(detections, key=lambda d: d[2] * d[3])
        face_rect = (x, y, w, h)
    else:
        h_img, w_img = image.shape[:2]
        face_rect = (0, 0, w_img, h_img)

    encoding = service.extract_face_encoding(image, face_rect)
    assert encoding is not None, f"无法提取特征: {image_path}"
    return encoding


def test_different_people_should_not_match() -> None:
    people = _collect_person_images(TEST_DATA_ROOT)

    config = FaceRecognitionConfig(
        data_dir="./data/tmp_faces_test_diff",
        detection_model="haar",
        recognition_threshold=0.9,
    )
    service = FaceRecognizerService(config=config)

    # 每个人取多张图像构建原型向量，降低单张样本噪声
    representatives: Dict[str, np.ndarray] = {}
    for person_name, image_paths in people.items():
        encodings = [_extract_encoding(service, p) for p in image_paths[:3]]
        proto = np.mean(np.vstack(encodings), axis=0)
        norm = np.linalg.norm(proto)
        if norm > 0:
            proto = proto / norm
        representatives[person_name] = proto

    person_names = sorted(representatives.keys())
    mismatch_pairs = []

    for name_a, name_b in itertools.combinations(person_names, 2):
        is_match, similarity = service.compare_faces(representatives[name_a], representatives[name_b])
        if is_match:
            mismatch_pairs.append((name_a, name_b, round(similarity, 4)))

    assert not mismatch_pairs, f"存在跨人误匹配: {mismatch_pairs}"
