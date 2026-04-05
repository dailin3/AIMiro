#!/usr/bin/env python3
"""测试不同的 det_size 参数"""

import sys
from pathlib import Path
import cv2
import insightface
from insightface.app import FaceAnalysis

sys.path.insert(0, str(Path(__file__).parent.parent))

test_data = Path("test_data/human/xuefen")
img_files = sorted(test_data.glob("*.png"))

# 测试不同的 det_size
for det_size in [(640, 640), (320, 320), (480, 480), (800, 800)]:
    print(f"\n{'='*60}")
    print(f"测试 det_size={det_size}")
    print(f"{'='*60}")

    app = FaceAnalysis()
    app.prepare(ctx_id=-1, det_size=det_size)

    # 测试第一张图片
    img_file = list(img_files)[0]
    img = cv2.imread(str(img_file))

    if img is not None:
        faces = app.get(img)
        if faces:
            print(f"✅ {img_file.name}: 检测到 {len(faces)} 张人脸")
            for face in faces:
                bbox = face.bbox
                print(f"   bbox: {bbox}, confidence: {bbox[4] if len(bbox) > 4 else 'N/A'}")
        else:
            print(f"❌ {img_file.name}: 未检测到人脸")
