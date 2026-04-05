#!/usr/bin/env python3
"""
InsightFace vs face_recognition 对比测试

测试目标：验证 InsightFace 是否能解决 jiang/xuefen 误匹配问题
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_insightface():
    """测试 InsightFace 效果"""
    print("=" * 80)
    print("InsightFace (ArcFace) 测试")
    print("=" * 80)

    try:
        from backend.face_recognition_insightface import (
            FaceRecognizerServiceInsightFace, INSIGHTFACE_AVAILABLE
        )
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return None

    if not INSIGHTFACE_AVAILABLE:
        print("❌ insightface 未安装")
        return None

    service = FaceRecognizerServiceInsightFace()
    service.clear_all_faces()

    test_data = Path("test_data/human")
    persons = {}

    # 注册每个人
    for person_dir in ["jiang", "xuefen", "rabbit"]:
        person_path = test_data / person_dir
        if not person_path.exists():
            print(f"⚠️  跳过 {person_dir} (目录不存在)")
            continue

        images = []
        for img_file in sorted(person_path.glob("*.png")):
            img = cv2.imread(str(img_file))
            if img is not None:
                images.append(img)

        if images:
            result = service.batch_register_faces(images, person_dir, person_dir)
            if result:
                persons[person_dir] = images
                print(f"✅ 注册 {person_dir}: {len(images)} 张图片")

    if len(persons) < 2:
        print("❌ 注册人数不足，无法继续测试")
        return None

    # 测试跨人距离
    print(f"\n{'='*60}")
    print(f"跨人距离测试 (余弦距离，阈值: certain<0.80, uncertain<1.00):")
    print(f"{'='*60}")

    face_ids = list(service.face_database.keys())
    results = {}

    for i, id1 in enumerate(face_ids):
        for id2 in face_ids[i+1:]:
            enc1 = service.face_database[id1].encoding
            enc2 = service.face_database[id2].encoding
            result = service.compare_faces_with_certainty(enc1, enc2)
            results[f"{id1} vs {id2}"] = result
            status = "✅" if not result["is_match"] else "❌ 误匹配"
            print(f"  {id1} vs {id2}: {result['distance']:.4f} ({result['certainty']}) {status}")

    # 测试同人距离
    print(f"\n{'='*60}")
    print(f"同人距离测试 (使用每人的前两张图片):")
    print(f"{'='*60}")

    for person_name, images in persons.items():
        if len(images) >= 2:
            # 提取两张图片的特征
            faces1 = service.app.get(images[0])
            faces2 = service.app.get(images[1])

            if faces1 and faces2:
                enc1 = faces1[0].embedding
                enc2 = faces2[0].embedding
                result = service.compare_faces_with_certainty(enc1, enc2)
                status = "✅" if result["is_match"] else "❌ 未匹配"
                print(f"  {person_name} (同人): {result['distance']:.4f} ({result['certainty']}) {status}")

    return results


def test_face_recognition():
    """测试 face_recognition 效果（作为对照）"""
    print(f"\n{'='*80}")
    print("face_recognition (ResNet-34) 测试 - 对照组")
    print(f"{'='*80}")

    try:
        from backend.face_recognition_module import FaceRecognizerService, FaceRecognitionConfig
        import face_recognition
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return None

    # 使用严格配置
    config = FaceRecognitionConfig(
        recognition_threshold=0.45,
        threshold_certain=0.40,
        threshold_uncertain=0.45,
        number_of_jitters=1,
        use_average_encoding=True
    )

    service = FaceRecognizerService(config)
    service.clear_all_faces()

    test_data = Path("test_data/human")
    persons = {}

    # 注册每个人
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
                persons[person_dir] = images
                print(f"✅ 注册 {person_dir}: {len(images)} 张图片")

    # 测试跨人距离
    print(f"\n{'='*60}")
    print(f"跨人距离测试 (欧氏距离，阈值: certain<0.40, uncertain<0.45):")
    print(f"{'='*60}")

    face_ids = list(service.face_database.keys())
    results = {}

    for i, id1 in enumerate(face_ids):
        for id2 in face_ids[i+1:]:
            enc1 = service.face_database[id1].encoding
            enc2 = service.face_database[id2].encoding
            result = service.compare_faces_with_certainty(enc1, enc2)
            results[f"{id1} vs {id2}"] = result
            status = "✅" if not result["is_match"] else "❌ 误匹配"
            print(f"  {id1} vs {id2}: {result['distance']:.4f} ({result['certainty']}) {status}")

    # 测试同人距离
    print(f"\n{'='*60}")
    print(f"同人距离测试:")
    print(f"{'='*60}")

    for person_name, images in persons.items():
        if len(images) >= 2:
            # 先检测人脸
            detections1 = service.detect_faces(images[0])
            detections2 = service.detect_faces(images[1])

            if detections1 and detections2:
                enc1 = service.extract_face_encoding(images[0], detections1[0])
                enc2 = service.extract_face_encoding(images[1], detections2[0])

                if enc1 is not None and enc2 is not None:
                    result = service.compare_faces_with_certainty(enc1, enc2)
                    status = "✅" if result["is_match"] else "❌ 未匹配"
                    print(f"  {person_name} (同人): {result['distance']:.4f} ({result['certainty']}) {status}")

    return results


def print_comparison_summary(insightface_results, face_recognition_results):
    """打印对比总结"""
    print(f"\n{'='*80}")
    print("对比总结")
    print(f"{'='*80}")

    if insightface_results:
        print(f"\n✅ InsightFace 结果:")
        for pair, result in insightface_results.items():
            print(f"   {pair}: {result['distance']:.4f} ({result['certainty']})")

    if face_recognition_results:
        print(f"\n⚠️  face_recognition 结果:")
        for pair, result in face_recognition_results.items():
            print(f"   {pair}: {result['distance']:.4f} ({result['certainty']})")

    print(f"\n{'='*80}")
    print("关键指标: jiang vs xuefen 距离对比")
    print(f"{'='*80}")

    if insightface_results and "jiang vs xuefen" in insightface_results:
        if_dist = insightface_results["jiang vs xuefen"]["distance"]
        print(f"   InsightFace:      {if_dist:.4f} {'✅ 可区分' if not insightface_results['jiang vs xuefen']['is_match'] else '❌ 误匹配'}")

    if face_recognition_results and "jiang vs xuefen" in face_recognition_results:
        fr_dist = face_recognition_results["jiang vs xuefen"]["distance"]
        print(f"   face_recognition: {fr_dist:.4f} {'✅ 可区分' if not face_recognition_results['jiang vs xuefen']['is_match'] else '❌ 误匹配'}")


if __name__ == "__main__":
    # 先测试 face_recognition（更快）
    fr_results = test_face_recognition()

    # 再测试 insightface（需要下载模型）
    if_results = test_insightface()

    # 打印对比
    print_comparison_summary(if_results, fr_results)
