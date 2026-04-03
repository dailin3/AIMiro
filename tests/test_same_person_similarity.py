#!/usr/bin/env python3
"""测试一：同一人的人脸应表现为相似。"""

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
        assert len(images) >= 2, f"人员目录 {person_dir} 至少需要 2 张人脸图"
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


def test_same_person_faces_should_be_similar() -> None:
    people = _collect_person_images(TEST_DATA_ROOT)

    config = FaceRecognitionConfig(
        data_dir="./data/tmp_faces_test_same",
        detection_model="haar",
        recognition_threshold=0.6,
    )
    service = FaceRecognizerService(config=config)

    for person_name, image_paths in people.items():
        encodings = [_extract_encoding(service, p) for p in image_paths]

        pair_results = []
        for enc1, enc2 in itertools.combinations(encodings, 2):
            is_match, similarity = service.compare_faces(enc1, enc2)
            pair_results.append((is_match, similarity))

        assert pair_results, f"人员 {person_name} 没有可比较的图像对"

        not_matched = [sim for matched, sim in pair_results if not matched]
        assert not not_matched, (
            f"人员 {person_name} 存在同人不匹配样本，"
            f"相似度: {[round(v, 4) for v in not_matched]}"
        )
