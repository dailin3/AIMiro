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
HUMAN_ROOT = PROJECT_ROOT / "test_data" / "human"
EXPECTED_PERSON_NAMES = {"jiang", "rabbit", "xuefen"}
DNN_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def _collect_person_images(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        pytest.skip(f"未找到测试数据目录: {root}")

    person_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    person_names = {p.name for p in person_dirs}
    assert person_names == EXPECTED_PERSON_NAMES, (
        f"期望人员目录 {sorted(EXPECTED_PERSON_NAMES)}，实际是 {sorted(person_names)}"
    )

    mapping: Dict[str, List[Path]] = {}
    for person_dir in person_dirs:
        images = sorted([p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        assert len(images) >= 3, f"人员目录 {person_dir} 至少需要 3 张人脸图（前2张建模，后续用于验证）"
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
    people = _collect_person_images(HUMAN_ROOT)
    assert DNN_MODEL_PATH.exists(), f"未找到 DNN 模型文件: {DNN_MODEL_PATH}"

    log_dir = PROJECT_ROOT / "test_data" / "tmp_data" / "same_person"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "similarity_log.txt"

    config = FaceRecognitionConfig(
        data_dir=str(log_dir),
        detection_model="dnn",
        detection_model_path=str(DNN_MODEL_PATH),
        recognition_threshold=0.8,
    )
    service = FaceRecognizerService(config=config)

    mismatches: List[str] = []
    log_lines: List[str] = []

    for person_name, image_paths in people.items():
        # 使用前 2 张图构建该人的平均原型 embedding
        base_encodings = [_extract_encoding(service, p) for p in image_paths[:2]]
        prototype = np.mean(np.vstack(base_encodings), axis=0)
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototype = prototype / norm

        base_image_names = f"{image_paths[0].name} + {image_paths[1].name}"
        log_lines.append(f"[{person_name}] base={base_image_names}")

        # 用后续图片逐张验证是否与原型一致
        for image_path in image_paths[2:]:
            query_encoding = _extract_encoding(service, image_path)
            is_match, similarity = service.compare_faces(prototype, query_encoding)
            log_lines.append(
                f"[{person_name}] query={image_path.name} base={base_image_names} similarity={similarity:.4f} match={is_match}"
            )
            if not is_match:
                mismatches.append(
                    f"{person_name}: {image_path.name} vs [{base_image_names}] 不一致（similarity={similarity:.4f}）"
                )

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    assert not mismatches, f"同人一致性失败: {mismatches}"
