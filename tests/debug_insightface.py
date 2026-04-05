#!/usr/bin/env python3
"""调试 insightface 注册问题"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.face_recognition_insightface import FaceRecognizerServiceInsightFace

service = FaceRecognizerServiceInsightFace()

test_data = Path("test_data/human")
for person_dir in ["xuefen"]:
    person_path = test_data / person_dir
    if not person_path.exists():
        print(f"目录不存在: {person_path}")
        continue

    images = []
    for img_file in sorted(person_path.glob("*.png")):
        img = cv2.imread(str(img_file))
        if img is not None:
            print(f"加载成功: {img_file}, shape={img.shape}")
            images.append(img)
        else:
            print(f"加载失败: {img_file}")

    if images:
        print(f"\n尝试检测 {person_dir} 的人脸...")
        for i, img in enumerate(images):
            faces = service.app.get(img)
            if faces:
                print(f"  图片 {i+1}: 检测到 {len(faces)} 张人脸")
            else:
                print(f"  图片 {i+1}: ❌ 未检测到人脸")

        # 批量注册
        result = service.batch_register_faces(images, person_dir, person_dir)
        if result:
            print(f"\n✅ 注册成功: {result.face_id}")
        else:
            print(f"\n❌ 注册失败")
