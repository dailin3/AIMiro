#!/usr/bin/env python3
"""
测试模型融合策略

核心思路：
1. 使用多个模型同时判定
2. 只有多个模型都认为是匹配时才确认
3. 通过加权融合提高判定可靠性

测试场景：
- buffalo_l + antelopev2 融合
- 投票策略 vs 加权融合
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import insightface
from insightface.app import FaceAnalysis


class ModelFusionStrategy:
    """模型融合策略"""

    def __init__(self):
        self.models = {}

    def add_model(self, name: str, app: FaceAnalysis):
        """添加一个模型"""
        self.models[name] = app

    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """提取所有模型的人脸特征"""
        features = {}
        for name, app in self.models.items():
            faces = app.get(image)
            if faces:
                features[name] = faces[0].embedding
        return features

    def compute_distance(self, enc1_dict: Dict[str, np.ndarray],
                        enc2_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算多个模型的距离"""
        distances = {}
        for name in self.models.keys():
            if name in enc1_dict and name in enc2_dict:
                enc1 = enc1_dict[name] / np.linalg.norm(enc1_dict[name])
                enc2 = enc2_dict[name] / np.linalg.norm(enc2_dict[name])
                similarity = float(np.dot(enc1, enc2))
                distance = 1.0 - similarity
                distances[name] = distance
        return distances

    def vote_strategy(self, distances: Dict[str, float],
                     thresholds: Dict[str, float]) -> Dict:
        """
        投票策略：统计多少个模型判定为匹配

        参数:
            distances: {模型名: 距离}
            thresholds: {模型名: 阈值}
        """
        votes = 0
        total = 0

        for name, dist in distances.items():
            if name in thresholds:
                total += 1
                if dist < thresholds[name]:
                    votes += 1

        # 需要超过半数模型同意
        is_match = votes > total / 2

        return {
            "is_match": is_match,
            "votes": votes,
            "total": total,
            "method": "voting"
        }

    def weighted_fusion(self, distances: Dict[str, float],
                       weights: Dict[str, float],
                       threshold: float = 0.5) -> Dict:
        """
        加权融合策略

        将不同模型的距离归一化后加权

        参数:
            distances: {模型名: 距离}
            weights: {模型名: 权重}
            threshold: 融合后的阈值
        """
        # 归一化距离到 [0, 1] 范围
        # 使用 sigmoid 函数：score = 1 / (1 + exp(k * (dist - mid)))
        normalized_scores = {}

        for name, dist in distances.items():
            if name in weights:
                # buffalo_l 和 antelopev2 的阈值大约在 0.9-1.0 之间
                # 使用 midpoint=0.95, k=10 的 sigmoid
                mid = 0.95
                k = 10
                score = 1 / (1 + np.exp(k * (dist - mid)))
                normalized_scores[name] = score

        # 加权平均
        total_weight = sum(weights.get(n, 0) for n in normalized_scores.keys())
        if total_weight > 0:
            fused_score = sum(
                normalized_scores[n] * weights.get(n, 0)
                for n in normalized_scores.keys()
            ) / total_weight
        else:
            fused_score = 0

        # fused_score 越接近 1 表示越匹配
        is_match = fused_score > threshold

        return {
            "is_match": is_match,
            "fused_score": fused_score,
            "method": "weighted_fusion"
        }

    def strict_fusion(self, distances: Dict[str, float],
                     thresholds: Dict[str, float]) -> Dict:
        """
        严格融合：所有模型都必须判定为匹配

        这是最严格的策略，可以最大程度减少误匹配
        """
        all_match = True
        for name, dist in distances.items():
            if name in thresholds:
                if dist >= thresholds[name]:
                    all_match = False
                    break

        return {
            "is_match": all_match,
            "distances": distances,
            "method": "strict_fusion"
        }


def test_fusion():
    """测试融合策略"""
    print("=" * 80)
    print("模型融合策略测试")
    print("=" * 80)

    # 初始化多个模型
    strategy = ModelFusionStrategy()

    models_to_test = {
        "buffalo_l": (480, 480),
        "antelopev2": (480, 480)
    }

    for model_name, det_size in models_to_test.items():
        try:
            app = FaceAnalysis(name=model_name)
            app.prepare(ctx_id=-1, det_size=det_size)
            strategy.add_model(model_name, app)
            print(f"✅ 加载模型: {model_name}")
        except Exception as e:
            print(f"⚠️  跳过 {model_name}: {e}")

    if len(strategy.models) < 2:
        print("❌ 需要至少 2 个模型才能融合")
        return

    # 注册人脸
    test_data = Path("test_data/human")
    person_features = {}

    for person_dir in ["jiang", "xuefen", "rabbit"]:
        person_path = test_data / person_dir
        if not person_path.exists():
            continue

        images = []
        for img_file in sorted(person_path.glob("*.png")):
            img = cv2.imread(str(img_file))
            if img is not None:
                features = strategy.extract_features(img)
                if features:
                    images.append(features)

        if images:
            # 平均特征
            avg_features = {}
            for name in strategy.models.keys():
                encs = [img[name] for img in images if name in img]
                if encs:
                    avg_features[name] = np.mean(encs, axis=0)

            person_features[person_dir] = avg_features
            print(f"✅ 注册 {person_dir}: {len(images)} 张图片")

    print(f"\n{'='*60}")
    print("跨人距离测试:")
    print(f"{'='*60}")

    person_ids = list(person_features.keys())
    thresholds = {
        "buffalo_l": 1.00,
        "antelopev2": 1.00
    }

    # 测试所有配对
    for i, id1 in enumerate(person_ids):
        for id2 in person_ids[i+1:]:
            print(f"\n{id1} vs {id2}:")

            # 计算各模型距离
            distances = strategy.compute_distance(
                person_features[id1], person_features[id2]
            )

            for name, dist in distances.items():
                if dist < 0.80:
                    certainty = "certain"
                    is_match = True
                elif dist < 1.00:
                    certainty = "uncertain"
                    is_match = True
                else:
                    certainty = "no_match"
                    is_match = False

                status = "❌ 误匹配" if is_match else "✅"
                print(f"  {name}: {dist:.4f} ({certainty}) {status}")

            # 投票策略
            vote_result = strategy.vote_strategy(distances, thresholds)
            print(f"  投票: {vote_result['votes']}/{vote_result['total']} 票 -> {'❌ 误匹配' if vote_result['is_match'] else '✅'}")

            # 加权融合
            weights = {"buffalo_l": 0.5, "antelopev2": 0.5}
            fusion_result = strategy.weighted_fusion(distances, weights, threshold=0.5)
            print(f"  加权融合: {fusion_result['fused_score']:.4f} -> {'❌ 误匹配' if fusion_result['is_match'] else '✅'}")

            # 严格融合
            strict_result = strategy.strict_fusion(distances, thresholds)
            print(f"  严格融合: {'❌ 误匹配' if strict_result['is_match'] else '✅'}")

    # 同人距离测试
    print(f"\n{'='*60}")
    print("同人距离测试:")
    print(f"{'='*60}")

    for person_dir in ["jiang", "xuefen", "rabbit"]:
        person_path = test_data / person_dir
        if not person_path.exists():
            continue

        images = []
        for img_file in sorted(person_path.glob("*.png"))[:2]:
            img = cv2.imread(str(img_file))
            if img is not None:
                features = strategy.extract_features(img)
                if features:
                    images.append(features)

        if len(images) >= 2:
            distances = strategy.compute_distance(images[0], images[1])

            print(f"\n{person_dir} (同人):")
            for name, dist in distances.items():
                if dist < 0.80:
                    certainty = "certain"
                    is_match = True
                elif dist < 1.00:
                    certainty = "uncertain"
                    is_match = True
                else:
                    certainty = "no_match"
                    is_match = False

                status = "✅" if is_match else "❌ 未匹配"
                print(f"  {name}: {dist:.4f} ({certainty}) {status}")

            # 融合策略
            vote_result = strategy.vote_strategy(distances, thresholds)
            print(f"  投票: {vote_result['votes']}/{vote_result['total']} 票 -> {'✅' if vote_result['is_match'] else '❌ 未匹配'}")

            weights = {"buffalo_l": 0.5, "antelopev2": 0.5}
            fusion_result = strategy.weighted_fusion(distances, weights, threshold=0.5)
            print(f"  加权融合: {fusion_result['fused_score']:.4f} -> {'✅' if fusion_result['is_match'] else '❌ 未匹配'}")


if __name__ == "__main__":
    test_fusion()
