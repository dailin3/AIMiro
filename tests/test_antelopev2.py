#!/usr/bin/env python3
"""
测试 antelopev2 模型效果

antelopev2 特点:
- ResNet100 骨干网络 (vs buffalo_l 的 ResNet50)
- Glint360K 训练集 (36万ID, vs WebFace600K)
- 理论上对相似人脸区分能力更强
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import insightface
from insightface.app import FaceAnalysis


def test_model(model_name: str, det_size=(480, 480)):
    """测试指定的 insightface 模型"""
    print(f"\n{'='*80}")
    print(f"测试模型: {model_name}")
    print(f"{'='*80}")

    try:
        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=-1, det_size=det_size)
        print(f"✅ 模型加载成功: {model_name}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

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
                faces = app.get(img)
                if faces:
                    images.append(faces[0].embedding)

        if images:
            # 使用平均特征
            avg_encoding = np.mean(images, axis=0)
            persons[person_dir] = avg_encoding
            print(f"✅ 注册 {person_dir}: {len(images)} 张图片")

    if len(persons) < 2:
        print("❌ 注册人数不足")
        return None

    # 测试跨人距离
    print(f"\n{'='*60}")
    print(f"跨人距离测试:")
    print(f"{'='*60}")

    person_ids = list(persons.keys())
    results = {}

    for i, id1 in enumerate(person_ids):
        for id2 in person_ids[i+1:]:
            enc1 = persons[id1]
            enc2 = persons[id2]

            # 归一化
            enc1_norm = enc1 / np.linalg.norm(enc1)
            enc2_norm = enc2 / np.linalg.norm(enc2)

            # 余弦相似度
            similarity = float(np.dot(enc1_norm, enc2_norm))
            distance = 1.0 - similarity

            # 判定
            if distance < 0.80:
                certainty = "certain"
                is_match = True
            elif distance < 1.00:
                certainty = "uncertain"
                is_match = True
            else:
                certainty = "no_match"
                is_match = False

            status = "✅" if not is_match else "❌ 误匹配"
            print(f"  {id1} vs {id2}: {distance:.4f} ({certainty}) {status}")
            results[f"{id1} vs {id2}"] = {
                "distance": distance,
                "similarity": similarity,
                "certainty": certainty,
                "is_match": is_match
            }

    # 测试同人距离
    print(f"\n{'='*60}")
    print(f"同人距离测试:")
    print(f"{'='*60}")

    for person_dir in ["jiang", "xuefen", "rabbit"]:
        person_path = test_data / person_dir
        if not person_path.exists():
            continue

        images = []
        for img_file in sorted(person_path.glob("*.png"))[:2]:  # 只用前两张
            img = cv2.imread(str(img_file))
            if img is not None:
                faces = app.get(img)
                if faces:
                    images.append(faces[0].embedding)

        if len(images) >= 2:
            enc1_norm = images[0] / np.linalg.norm(images[0])
            enc2_norm = images[1] / np.linalg.norm(images[1])
            similarity = float(np.dot(enc1_norm, enc2_norm))
            distance = 1.0 - similarity

            if distance < 0.80:
                certainty = "certain"
                is_match = True
            elif distance < 1.00:
                certainty = "uncertain"
                is_match = True
            else:
                certainty = "no_match"
                is_match = False

            status = "✅" if is_match else "❌ 未匹配"
            print(f"  {person_dir} (同人): {distance:.4f} ({certainty}) {status}")

    return results


if __name__ == "__main__":
    # 测试 buffalo_l (对照)
    print("开始测试不同的 InsightFace 模型...")

    # buffalo_l (已测试过，作为对照)
    results_l = test_model("buffalo_l")

    # antelopev2 (更强大的模型)
    results_a = test_model("antelopev2")

    # 对比总结
    print(f"\n{'='*80}")
    print("模型对比总结")
    print(f"{'='*80}")

    if results_l:
        print(f"\n📊 buffalo_l:")
        for pair, r in results_l.items():
            print(f"   {pair}: {r['distance']:.4f} ({r['certainty']}) {'❌' if r['is_match'] else '✅'}")

    if results_a:
        print(f"\n📊 antelopev2:")
        for pair, r in results_a.items():
            print(f"   {pair}: {r['distance']:.4f} ({r['certainty']}) {'❌' if r['is_match'] else '✅'}")

    # 关键指标对比
    print(f"\n{'='*80}")
    print("关键指标: jiang vs xuefen")
    print(f"{'='*80}")

    if results_l and "jiang vs xuefen" in results_l:
        print(f"   buffalo_l:   {results_l['jiang vs xuefen']['distance']:.4f} ({results_l['jiang vs xuefen']['certainty']})")

    if results_a and "jiang vs xuefen" in results_a:
        print(f"   antelopev2:  {results_a['jiang vs xuefen']['distance']:.4f} ({results_a['jiang vs xuefen']['certainty']})")
