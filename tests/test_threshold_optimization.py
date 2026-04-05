#!/usr/bin/env python3
"""
基于实际测试数据优化阈值

目标：找到一个阈值，能够最大化同人匹配率，同时最小化跨人误匹配率
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import insightface
from insightface.app import FaceAnalysis


def collect_distances(app: FaceAnalysis, model_name: str):
    """收集所有同人和跨人距离"""
    test_data = Path("test_data/human")

    # 收集同人距离
    same_person_distances = []
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

        # 所有配对
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                enc1 = images[i] / np.linalg.norm(images[i])
                enc2 = images[j] / np.linalg.norm(images[j])
                sim = float(np.dot(enc1, enc2))
                dist = 1.0 - sim
                same_person_distances.append(dist)

    # 收集跨人距离
    different_person_distances = []
    persons = {}
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
            persons[person_dir] = np.mean(images, axis=0)

    person_ids = list(persons.keys())
    for i, id1 in enumerate(person_ids):
        for id2 in person_ids[i+1:]:
            enc1 = persons[id1] / np.linalg.norm(persons[id1])
            enc2 = persons[id2] / np.linalg.norm(persons[id2])
            sim = float(np.dot(enc1, enc2))
            dist = 1.0 - sim
            different_person_distances.append(dist)

    return same_person_distances, different_person_distances


def find_optimal_threshold(same_distances: List[float],
                          different_distances: List[float]):
    """
    寻找最优阈值

    目标：最大化准确率（同人匹配 + 跨人不匹配）
    """
    best_threshold = None
    best_accuracy = 0
    best_stats = None

    print("\n阈值扫描分析:")
    print(f"{'阈值':<10} {'同人匹配率':<12} {'跨人拒绝率':<12} {'准确率':<10}")
    print("-" * 50)

    # 扫描 0.3 到 1.5 的阈值
    for threshold_int in range(30, 150, 1):
        threshold = threshold_int / 100.0

        # 同人匹配率
        same_match = sum(1 for d in same_distances if d < threshold)
        same_rate = same_match / len(same_distances) if same_distances else 0

        # 跨人拒绝率
        different_reject = sum(1 for d in different_distances if d >= threshold)
        different_rate = different_reject / len(different_distances) if different_distances else 0

        # 总准确率
        total_samples = len(same_distances) + len(different_distances)
        if total_samples > 0:
            accuracy = (same_match + different_reject) / total_samples
        else:
            accuracy = 0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_stats = {
                "threshold": threshold,
                "same_match_rate": same_rate,
                "different_reject_rate": different_rate,
                "accuracy": accuracy,
                "same_match": same_match,
                "same_total": len(same_distances),
                "different_reject": different_reject,
                "different_total": len(different_distances)
            }

        if threshold_int % 10 == 0:  # 每 0.1 打印一次
            print(f"{threshold:<10.2f} {same_rate:<12.2%} {different_rate:<12.2%} {accuracy:<10.2%}")

    return best_stats


def analyze_model(model_name: str, det_size=(480, 480)):
    """分析单个模型"""
    print(f"\n{'='*80}")
    print(f"分析模型: {model_name}")
    print(f"{'='*80}")

    try:
        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=-1, det_size=det_size)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    same_distances, different_distances = collect_distances(app, model_name)

    print(f"\n数据统计:")
    print(f"  同人距离样本数: {len(same_distances)}")
    print(f"    最小: {min(same_distances):.4f}")
    print(f"    最大: {max(same_distances):.4f}")
    print(f"    平均: {np.mean(same_distances):.4f}")
    print(f"    标准差: {np.std(same_distances):.4f}")

    print(f"\n  跨人距离样本数: {len(different_distances)}")
    print(f"    最小: {min(different_distances):.4f}")
    print(f"    最大: {max(different_distances):.4f}")
    print(f"    平均: {np.mean(different_distances):.4f}")
    print(f"    标准差: {np.std(different_distances):.4f}")

    # 检查是否有重叠
    same_max = max(same_distances)
    different_min = min(different_distances)

    print(f"\n  危险区域分析:")
    print(f"    同人最大距离: {same_max:.4f}")
    print(f"    跨人最小距离: {different_min:.4f}")

    if same_max >= different_min:
        overlap = same_max - different_min
        print(f"    ⚠️  存在重叠区域: {overlap:.4f}")
        print(f"    无法通过单一阈值完全分离同人和跨人")
        print(f"    最佳阈值范围: [{different_min:.4f}, {same_max:.4f}]")
    else:
        gap = different_min - same_max
        print(f"    ✅ 安全间隙: {gap:.4f}")
        print(f"    可以通过阈值完全分离")

    # 寻找最优阈值
    best_stats = find_optimal_threshold(same_distances, different_distances)

    if best_stats:
        print(f"\n{'='*60}")
        print(f"最优阈值: {best_stats['threshold']:.2f}")
        print(f"  准确率: {best_stats['accuracy']:.2%}")
        print(f"  同人匹配率: {best_stats['same_match_rate']:.2%} ({best_stats['same_match']}/{best_stats['same_total']})")
        print(f"  跨人拒绝率: {best_stats['different_reject_rate']:.2%} ({best_stats['different_reject']}/{best_stats['different_total']})")

        # 列出误匹配的跨人配对
        print(f"\n  在该阈值下会误匹配的跨人配对:")
        for i, dist in enumerate(different_distances):
            if dist < best_stats['threshold']:
                print(f"    ❌ 距离 {dist:.4f}")

    return best_stats


if __name__ == "__main__":
    print("=" * 80)
    print("基于实际数据的阈值优化分析")
    print("=" * 80)

    # 分析 buffalo_l
    stats_l = analyze_model("buffalo_l")

    # 分析 antelopev2
    stats_a = analyze_model("antelopev2")

    # 对比总结
    print(f"\n{'='*80}")
    print("模型对比总结")
    print(f"{'='*80}")

    if stats_l:
        print(f"\n📊 buffalo_l:")
        print(f"   最优阈值: {stats_l['threshold']:.2f}")
        print(f"   准确率: {stats_l['accuracy']:.2%}")
        print(f"   同人匹配: {stats_l['same_match']}/{stats_l['same_total']}")
        print(f"   跨人拒绝: {stats_l['different_reject']}/{stats_l['different_total']}")

    if stats_a:
        print(f"\n📊 antelopev2:")
        print(f"   最优阈值: {stats_a['threshold']:.2f}")
        print(f"   准确率: {stats_a['accuracy']:.2%}")
        print(f"   同人匹配: {stats_a['same_match']}/{stats_a['same_total']}")
        print(f"   跨人拒绝: {stats_a['different_reject']}/{stats_a['different_total']}")
