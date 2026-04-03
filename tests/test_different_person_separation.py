#!/usr/bin/env python3
"""测试二：不同人的人脸应被区分。"""

from __future__ import annotations

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
TMP_DATA_ROOT = PROJECT_ROOT / "test_data" / "tmp_data"


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


def _build_prototype(service: FaceRecognizerService, image_paths: List[Path]) -> np.ndarray:
    base_encodings = [_extract_encoding(service, p) for p in image_paths[:2]]
    prototype = np.mean(np.vstack(base_encodings), axis=0)
    norm = np.linalg.norm(prototype)
    if norm > 0:
        prototype = prototype / norm
    return prototype


def test_different_people_should_not_match() -> None:
    people = _collect_person_images(HUMAN_ROOT)
    assert DNN_MODEL_PATH.exists(), f"未找到 DNN 模型文件: {DNN_MODEL_PATH}"

    log_dir = TMP_DATA_ROOT / "different_people"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "similarity_log.txt"

    config = FaceRecognitionConfig(
        data_dir=str(log_dir),
        detection_model="dnn",
        detection_model_path=str(DNN_MODEL_PATH),
        recognition_threshold=0.80,
    )
    service = FaceRecognizerService(config=config)

    mismatch_pairs: List[str] = []
    log_lines: List[str] = []

    for reference_name, reference_paths in people.items():
        reference_prototype = _build_prototype(service, reference_paths)
        reference_base = f"{reference_paths[0].name} + {reference_paths[1].name}"
        log_lines.append(f"[{reference_name}] base={reference_base}")

        for target_name, target_paths in people.items():
            if target_name == reference_name:
                continue

            for image_path in target_paths:
                target_encoding = _extract_encoding(service, image_path)
                is_match, similarity = service.compare_faces(reference_prototype, target_encoding)
                log_lines.append(
                    f"[{reference_name}] target={target_name}/{image_path.name} base={reference_base} similarity={similarity:.4f} match={is_match}"
                )
                if is_match:
                    mismatch_pairs.append(
                        f"{reference_name} -> {target_name}/{image_path.name} (similarity={similarity:.4f})"
                    )

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    assert not mismatch_pairs, f"存在跨人误匹配: {mismatch_pairs}"
