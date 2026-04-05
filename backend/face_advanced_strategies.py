#!/usr/bin/env python3
"""
人脸识别高级策略模块 - 解决相似人脸误匹配问题

包含四种方案:
  A. 多模型融合/投票策略 (weighted fusion + voting)
  B. 更强大的 InsightFace 模型 (antelopev2, glint360k_r100)
  C. 辅助特征提取与融合 (landmarks, facial proportions)
  D. 自适应阈值选择 (ROC 曲线, EER 计算)

用法:
    from backend.face_advanced_strategies import (
        EnsembleRecognizer,     # 方案 A
        MultiModelRecognizer,   # 方案 B
        LandmarkAssistant,      # 方案 C
        ThresholdOptimizer,     # 方案 D
    )
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import cv2

# ============================================================
# 方案 A: 多模型融合 / 投票策略
# ============================================================

@dataclass
class ModelScore:
    """单个模型的评分结果"""
    model_name: str
    distance: float
    similarity: float
    threshold: float
    is_match: bool
    confidence: float  # 0~1, 表示该判定的置信度


@dataclass
class EnsembleResult:
    """融合判定结果"""
    is_match: bool
    certainty: str  # "certain" / "uncertain" / "no_match"
    # 各模型原始评分
    model_scores: List[ModelScore] = field(default_factory=list)
    # 融合后距离/相似度
    fused_distance: float = 0.0
    fused_similarity: float = 0.0
    # 投票结果
    vote_count: int = 0  # 投"匹配"的模型数
    vote_total: int = 0  # 总模型数
    # 加权融合结果
    weighted_score: float = 0.0  # 加权后的综合匹配分


class ModelFusionStrategy:
    """
    多模型融合策略

    三种融合模式:
    1. Voting (投票): 多数模型判定匹配则匹配
    2. WeightedAverage (加权平均): 按模型可靠性加权
    3. ScoreNormalization + Fusion (分数归一化后融合): 最精确
    """

    def __init__(
        self,
        strategy: str = "weighted",  # "voting" | "weighted" | "normalized"
        model_weights: Optional[Dict[str, float]] = None,
        voting_threshold: int = None,  # 需要多少模型同意
    ):
        self.strategy = strategy
        # 模型权重 (用于 weighted 策略), 默认等权
        self.model_weights = model_weights or {}
        self.voting_threshold = voting_threshold

    def normalize_score(self, score: ModelScore) -> float:
        """
        将不同模型的分数归一化到 [0, 1] 范围
        - 距离越小越好 -> 转换为匹配分数
        - 使用 sigmoid-like 变换: score = 1 / (1 + exp(k * (distance - threshold)))
        """
        k = 10.0  # 斜率参数
        normalized = 1.0 / (1.0 + np.exp(k * (score.distance - score.threshold)))
        return float(np.clip(normalized, 0, 1))

    def voting_decision(self, scores: List[ModelScore]) -> EnsembleResult:
        """投票决策: 统计多少模型判定匹配"""
        match_votes = sum(1 for s in scores if s.is_match)
        total = len(scores)

        # 默认需要超过半数
        required = self.voting_threshold or (total // 2 + 1)
        is_match = match_votes >= required

        # 置信度 = 匹配比例
        confidence = match_votes / total if total > 0 else 0

        # 判定 certainty
        if confidence >= 0.8:
            certainty = "certain"
        elif confidence >= 0.5:
            certainty = "uncertain"
        else:
            certainty = "no_match"

        return EnsembleResult(
            is_match=is_match,
            certainty=certainty,
            model_scores=scores,
            vote_count=match_votes,
            vote_total=total,
            fused_similarity=confidence,
            fused_distance=1.0 - confidence,
            weighted_score=confidence,
        )

    def weighted_decision(self, scores: List[ModelScore]) -> EnsembleResult:
        """加权平均决策"""
        total_weight = 0
        weighted_sim = 0
        weighted_dist = 0

        for s in scores:
            w = self.model_weights.get(s.model_name, 1.0)
            norm_score = self.normalize_score(s)
            weighted_sim += w * norm_score
            weighted_dist += w * (1.0 - norm_score)
            total_weight += w

        if total_weight > 0:
            weighted_sim /= total_weight
            weighted_dist /= total_weight

        # 加权后判定 (阈值 0.5 对应归一化后距离 0.5)
        is_match = weighted_sim >= 0.5
        certainty = (
            "certain" if weighted_sim >= 0.75
            else "uncertain" if weighted_sim >= 0.5
            else "no_match"
        )

        match_votes = sum(1 for s in scores if s.is_match)

        return EnsembleResult(
            is_match=is_match,
            certainty=certainty,
            model_scores=scores,
            vote_count=match_votes,
            vote_total=len(scores),
            fused_distance=weighted_dist,
            fused_similarity=weighted_sim,
            weighted_score=weighted_sim,
        )

    def normalized_decision(self, scores: List[ModelScore]) -> EnsembleResult:
        """
        分数归一化融合:
        1. 将每个模型的距离归一化到统一尺度
        2. 加权平均
        3. 反归一化到最终判定
        """
        return self.weighted_decision(scores)  # 内部使用相同的归一化逻辑

    def fuse(self, scores: List[ModelScore]) -> EnsembleResult:
        """根据策略融合多个模型的评分"""
        if not scores:
            return EnsembleResult(is_match=False, certainty="no_match")

        if self.strategy == "voting":
            return self.voting_decision(scores)
        elif self.strategy == "weighted":
            return self.weighted_decision(scores)
        else:  # "normalized"
            return self.normalized_decision(scores)


class EnsembleRecognizer:
    """
    多模型融合识别器 - 方案 A

    结合 face_recognition (ResNet-34) 和 InsightFace (ArcFace) 的判定,
    通过投票或加权融合降低相似人脸误匹配率。

    典型用法:
        recognizer = EnsembleRecognizer(
            fr_service=face_rec_service,
            if_service=insightface_service,
            strategy="weighted",
            model_weights={"face_recognition": 0.4, "insightface": 0.6}
        )
        result = recognizer.compare_faces(encoding_fr1, encoding_if1,
                                           encoding_fr2, encoding_if2)
    """

    def __init__(
        self,
        fr_service=None,
        if_service=None,
        strategy: str = "weighted",
        model_weights: Optional[Dict[str, float]] = None,
    ):
        self.fr_service = fr_service
        self.if_service = if_service

        # 默认权重: InsightFace 略高 (通常在相似人脸区分上表现更好)
        default_weights = {
            "face_recognition": 0.4,
            "insightface": 0.6,
        }
        if model_weights:
            default_weights.update(model_weights)

        self.fusion = ModelFusionStrategy(
            strategy=strategy,
            model_weights=default_weights,
        )

    def compare_faces(
        self,
        fr_enc1: np.ndarray, fr_enc2: np.ndarray,
        if_enc1: np.ndarray, if_enc2: np.ndarray,
    ) -> EnsembleResult:
        """
        融合两个模型的判定结果

        Args:
            fr_enc1, fr_enc2: face_recognition 特征 (128 维)
            if_enc1, if_enc2: InsightFace 特征 (512 维)
        """
        scores = []

        # face_recognition 模型评分
        if self.fr_service and fr_enc1 is not None and fr_enc2 is not None:
            dist = float(np.linalg.norm(fr_enc1.flatten() - fr_enc2.flatten()))
            threshold = getattr(self.fr_service.config, 'threshold_uncertain', 0.45)
            scores.append(ModelScore(
                model_name="face_recognition",
                distance=dist,
                similarity=1.0 - dist,
                threshold=threshold,
                is_match=dist < threshold,
                confidence=max(0, 1.0 - dist / threshold),
            ))

        # InsightFace 模型评分
        if self.if_service and if_enc1 is not None and if_enc2 is not None:
            enc1_norm = if_enc1 / np.linalg.norm(if_enc1)
            enc2_norm = if_enc2 / np.linalg.norm(if_enc2)
            sim = float(np.dot(enc1_norm, enc2_norm))
            dist = 1.0 - sim
            threshold = getattr(self.if_service.config, 'threshold_uncertain', 1.0)
            scores.append(ModelScore(
                model_name="insightface",
                distance=dist,
                similarity=sim,
                threshold=threshold,
                is_match=dist < threshold,
                confidence=max(0, 1.0 - dist / threshold),
            ))

        return self.fusion.fuse(scores)

    def recognize_face(
        self,
        unknown_fr_encoding: np.ndarray,
        unknown_if_encoding: np.ndarray,
        known_faces: Dict[str, Dict[str, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """
        识别未知人脸 (融合多模型)

        Args:
            known_faces: {face_id: {"fr_encoding": ..., "if_encoding": ...}}
        """
        results = []
        for face_id, known_encs in known_faces.items():
            result = self.compare_faces(
                unknown_fr_encoding, known_encs["fr_encoding"],
                unknown_if_encoding, known_encs["if_encoding"],
            )
            results.append({
                "face_id": face_id,
                "ensemble": result,
            })

        results.sort(key=lambda x: x["ensemble"].fused_similarity, reverse=True)
        return results


# ============================================================
# 方案 B: 更强大的 InsightFace 模型
# ============================================================

class MultiModelRecognizer:
    """
    多 InsightFace 模型识别器 - 方案 B

    支持加载不同的 InsightFace 模型包:
    - buffalo_l (默认, ResNet50@WebFace600K)
    - antelopev2 (ResNet100@Glint360K, 旗舰精度)
    - 自定义 ONNX 模型路径

    典型用法:
        # 使用 antelopev2
        recognizer = MultiModelRecognizer(model_name="antelopev2")

        # 使用自定义 ONNX 模型
        recognizer = MultiModelRecognizer(
            model_name="custom",
            model_path="/path/to/glint360k_r100.onnx"
        )

        # 对比两个模型的判定
        rec_buffalo = MultiModelRecognizer(model_name="buffalo_l")
        rec_antelope = MultiModelRecognizer(model_name="antelopev2")
    """

    SUPPORTED_MODELS = {
        "buffalo_l": {
            "arch": "ResNet50@WebFace600K",
            "dim": 512,
            "size_mb": 326,
            "det_size": (640, 640),
            "desc": "默认模型, 跨人种均衡性优秀",
        },
        "antelopev2": {
            "arch": "ResNet100@Glint360K",
            "dim": 512,
            "size_mb": 407,
            "det_size": (640, 640),
            "desc": "旗舰精度, Glint360K 训练集",
        },
        "buffalo_m": {
            "arch": "ResNet50@WebFace600K",
            "dim": 512,
            "size_mb": 313,
            "det_size": (480, 480),
            "desc": "与 buffalo_l 精度一致, 体积略小",
        },
        "buffalo_s": {
            "arch": "MBF@WebFace600K",
            "dim": 512,
            "size_mb": 159,
            "det_size": (320, 320),
            "desc": "轻量级模型",
        },
    }

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_path: Optional[str] = None,
        det_size: Tuple[int, int] = None,
        ctx_id: int = -1,
    ):
        self.model_name = model_name
        self.ctx_id = ctx_id

        if model_name in self.SUPPORTED_MODELS:
            info = self.SUPPORTED_MODELS[model_name]
            self.det_size = det_size or info["det_size"]
            self.embedding_dim = info["dim"]
            self.app = None
            self._init_insightface_app(model_name)
        elif model_path:
            # 自定义 ONNX 模型
            self.det_size = det_size or (640, 640)
            self.embedding_dim = 512
            self.app = None
            self._init_custom_model(model_path)
        else:
            raise ValueError(f"未知模型: {model_name}")

    def _init_insightface_app(self, model_name: str):
        """初始化 InsightFace 应用"""
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name=model_name)
            self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        except ImportError:
            raise ImportError("请安装 insightface: pip install insightface onnxruntime")

    def _init_custom_model(self, model_path: str):
        """
        初始化自定义 ONNX 模型

        用于加载独立下载的 ONNX 模型文件, 例如:
        - glint360k_r100.onnx
        - webface600k_iresnet50.onnx
        """
        try:
            import onnxruntime as ort
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.model_path = model_path
        except ImportError:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")

    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """提取人脸特征向量"""
        if self.app:
            faces = self.app.get(image)
            if faces:
                return faces[0].embedding
            return None
        else:
            # 使用自定义 ONNX 模型
            return self._extract_custom_embedding(image)

    def _extract_custom_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用自定义 ONNX 模型提取特征"""
        # 需要先用检测器找到人脸, 这里简化处理
        # 实际应用中需要先用 RetinaFace/MTCNN 检测
        return None

    def compare_faces(self, enc1: np.ndarray, enc2: np.ndarray,
                      threshold: float = None) -> Dict[str, Any]:
        """比较两个人脸特征"""
        if enc1 is None or enc2 is None:
            return {"is_match": False, "distance": float('inf'), "similarity": 0.0}

        # 归一化
        enc1 = enc1 / np.linalg.norm(enc1)
        enc2 = enc2 / np.linalg.norm(enc2)

        # 余弦相似度
        similarity = float(np.dot(enc1, enc2))
        distance = 1.0 - similarity

        # 默认阈值: antelopev2 建议 0.5~0.6, buffalo_l 建议 0.8~1.0
        if threshold is None:
            threshold = 0.6 if self.model_name == "antelopev2" else 1.0

        return {
            "is_match": distance < threshold,
            "distance": distance,
            "similarity": similarity,
            "model": self.model_name,
            "threshold": threshold,
        }


# ============================================================
# 方案 C: 辅助特征提取与融合
# ============================================================

class LandmarkAssistant:
    """
    人脸辅助特征提取 - 方案 C

    提取几何特征作为深度特征的补充:
    1. 68/106 关键点欧氏距离
    2. 面部比例特征 (三庭五眼)
    3. 肤色直方图特征

    典型用法:
        assistant = LandmarkAssistant()

        # 从 InsightFace 检测结果中提取
        faces = insightface_app.get(image)
        landmark_dist = assistant.compute_landmark_distance(
            faces[0].landmark, other_landmark
        )
        proportion_dist = assistant.compute_proportion_distance(
            faces[0].landmark, other_landmark
        )

        # 与深度特征融合
        fused = assistant.fuse_with_depth(
            depth_distance=0.937,
            landmark_distance=0.15,
            proportion_distance=0.08,
        )
    """

    # 68 点面部关键区域定义
    # 左眼 (36-41), 右眼 (42-47), 鼻子 (27-35), 嘴 (48-67), 下颌 (0-16), 左眉(22-26), 右眉(17-21)
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))
    NOSE = list(range(27, 36))
    MOUTH = list(range(48, 68))
    JAW = list(range(0, 17))

    def __init__(self):
        pass

    def compute_landmark_distance(
        self, lm1: np.ndarray, lm2: np.ndarray, normalize: bool = True
    ) -> float:
        """
        计算两个面部关键点的归一化欧氏距离

        Args:
            lm1, lm2: 关键点坐标 (N, 2) 或 (N, 3)
            normalize: 是否使用瞳孔距离归一化

        Returns:
            归一化后的关键点距离 (越小越相似)
        """
        if lm1 is None or lm2 is None:
            return float('inf')

        lm1 = np.array(lm1)
        lm2 = np.array(lm2)

        # 取 (x, y) 坐标
        if lm1.shape[-1] == 3:
            lm1 = lm1[:, :2]
            lm2 = lm2[:, :2]

        # 点对点欧氏距离的均值
        point_dists = np.sqrt(np.sum((lm1 - lm2) ** 2, axis=1))
        mean_dist = np.mean(point_dists)

        if normalize:
            # 使用瞳孔距离归一化
            pupil_dist = self._compute_pupil_distance(lm1, lm2)
            if pupil_dist > 0:
                mean_dist = mean_dist / pupil_dist

        return float(mean_dist)

    def _compute_pupil_distance(self, lm1: np.ndarray, lm2: np.ndarray) -> float:
        """计算平均瞳孔距离 (用于归一化)"""
        # 左眼中心
        left_eye_1 = np.mean(lm1[self.LEFT_EYE], axis=0)
        left_eye_2 = np.mean(lm2[self.LEFT_EYE], axis=0)
        # 右眼中心
        right_eye_1 = np.mean(lm1[self.RIGHT_EYE], axis=0)
        right_eye_2 = np.mean(lm2[self.RIGHT_EYE], axis=0)

        pupil_dist_1 = np.linalg.norm(left_eye_1 - right_eye_1)
        pupil_dist_2 = np.linalg.norm(left_eye_2 - right_eye_2)

        return (pupil_dist_1 + pupil_dist_2) / 2.0

    def compute_proportion_distance(
        self, lm1: np.ndarray, lm2: np.ndarray
    ) -> float:
        """
        计算面部比例差异 (三庭五眼)

        Returns:
            比例差异 (越小越相似)
        """
        if lm1 is None or lm2 is None:
            return float('inf')

        lm1 = np.array(lm1)
        lm2 = np.array(lm2)
        if lm1.shape[-1] == 3:
            lm1 = lm1[:, :2]
            lm2 = lm2[:, :2]

        proportions1 = self._compute_facial_proportions(lm1)
        proportions2 = self._compute_facial_proportions(lm2)

        # 比例差异
        diff = np.abs(np.array(proportions1) - np.array(proportions2))
        return float(np.mean(diff))

    def _compute_facial_proportions(self, landmarks: np.ndarray) -> List[float]:
        """
        计算面部比例特征

        Returns:
            比例列表: [上庭/中庭, 中庭/下庭, 眼宽/脸宽, 鼻宽/脸宽, 嘴宽/脸宽]
        """
        if len(landmarks) < 68:
            return [0.0] * 5

        # 关键位置
        chin = landmarks[8]       # 下巴
        forehead = landmarks[27]  # 眉心上方 (近似额头)
        nose_bottom = landmarks[33]  # 鼻尖
        nose_left = landmarks[31]    # 鼻翼左
        nose_right = landmarks[35]   # 鼻翼右
        mouth_left = landmarks[48]   # 嘴角左
        mouth_right = landmarks[54]  # 嘴角右
        eye_left_outer = landmarks[36]  # 左眼外角
        eye_right_outer = landmarks[45]  # 右眼外角

        # 脸宽 (近似)
        face_width = np.linalg.norm(
            landmarks[0] - landmarks[16]
        )  # 下颌宽度
        if face_width == 0:
            face_width = 1.0

        # 三庭
        upper_face = np.linalg.norm(forehead - landmarks[27])
        middle_face = np.linalg.norm(landmarks[27] - nose_bottom)
        lower_face = np.linalg.norm(nose_bottom - chin)

        # 五眼相关比例
        eye_width = np.linalg.norm(eye_left_outer - eye_right_outer)
        nose_width = np.linalg.norm(nose_left - nose_right)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)

        # 计算比例 (避免除零)
        mid = middle_face if middle_face > 0 else 1.0
        low = lower_face if lower_face > 0 else 1.0

        return [
            upper_face / mid,      # 上庭/中庭
            mid / low,              # 中庭/下庭
            eye_width / face_width,  # 眼距/脸宽
            nose_width / face_width, # 鼻宽/脸宽
            mouth_width / face_width, # 嘴宽/脸宽
        ]

    def compute_color_histogram_distance(
        self, face_roi1: np.ndarray, face_roi2: np.ndarray, bins: int = 32
    ) -> float:
        """
        计算肤色直方图距离 (Bhattacharyya distance)

        Args:
            face_roi1, face_roi2: 裁剪出的人脸区域 (BGR)

        Returns:
            直方图距离 (越小越相似)
        """
        if face_roi1 is None or face_roi2 is None:
            return float('inf')

        # 转换到 HSV 色彩空间, 使用 H 和 S 通道
        hsv1 = cv2.cvtColor(face_roi1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(face_roi2, cv2.COLOR_BGR2HSV)

        # 计算 H-S 联合直方图
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [bins, bins], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [bins, bins], [0, 180, 0, 256])

        # 归一化
        hist1 = cv2.normalize(hist1, None).flatten()
        hist2 = cv2.normalize(hist2, None).flatten()

        # Bhattacharyya 距离
        dist = float(np.sqrt(1.0 - np.sum(np.sqrt(hist1 * hist2))))
        return dist

    def fuse_with_depth(
        self,
        depth_distance: float,
        landmark_distance: float = None,
        proportion_distance: float = None,
        color_distance: float = None,
        weights: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        融合深度特征与辅助特征

        Args:
            depth_distance: 深度特征距离 (归一化)
            landmark_distance: 关键点距离
            proportion_distance: 面部比例距离
            color_distance: 肤色距离
            weights: 各特征权重

        Returns:
            融合后的判定结果
        """
        # 默认权重: 深度特征为主, 辅助特征为辅
        default_weights = {
            "depth": 0.7,
            "landmark": 0.15,
            "proportion": 0.1,
            "color": 0.05,
        }
        if weights:
            default_weights.update(weights)

        w = default_weights

        # 将所有距离归一化到 [0, 1], 越小越相似
        features = {"depth": depth_distance}
        if landmark_distance is not None:
            features["landmark"] = min(landmark_distance, 1.0)
        if proportion_distance is not None:
            features["proportion"] = min(proportion_distance, 1.0)
        if color_distance is not None:
            features["color"] = min(color_distance, 1.0)

        # 加权融合
        total_weight = 0
        fused_distance = 0
        for name, dist in features.items():
            weight = w.get(name, 0)
            fused_distance += weight * dist
            total_weight += weight

        if total_weight > 0:
            fused_distance /= total_weight

        # 判定
        threshold = 0.5  # 可调整
        is_match = fused_distance < threshold
        certainty = (
            "certain" if fused_distance < 0.3
            else "uncertain" if fused_distance < 0.5
            else "no_match"
        )

        return {
            "is_match": is_match,
            "certainty": certainty,
            "fused_distance": fused_distance,
            "depth_distance": depth_distance,
            "landmark_distance": landmark_distance,
            "proportion_distance": proportion_distance,
            "color_distance": color_distance,
        }


# ============================================================
# 方案 D: 自适应阈值选择
# ============================================================

class ThresholdOptimizer:
    """
    自适应阈值优化 - 方案 D

    基于测试数据自动选择最优判定阈值:
    1. ROC 曲线分析
    2. Equal Error Rate (EER) 计算
    3. Youden's J 统计量
    4. 自定义代价函数优化

    典型用法:
        optimizer = ThresholdOptimizer()

        # 添加正负样本对
        # 同人距离 (正样本)
        optimizer.add_same_person_distances([0.12, 0.18, 0.25, ...])
        # 跨人距离 (负样本)
        optimizer.add_diff_person_distances([0.45, 0.67, 0.93, ...])

        # 计算最优阈值
        result = optimizer.find_optimal_threshold()
        print(f"最优阈值: {result['threshold']:.4f}")
        print(f"EER: {result['eer']:.4f}")

        # 绘制 ROC 曲线
        optimizer.plot_roc_curve("roc_curve.png")
    """

    def __init__(self):
        self.same_person_distances: List[float] = []
        self.diff_person_distances: List[float] = []

    def add_same_person_distances(self, distances: List[float]):
        """添加同人距离 (正样本, genuine pairs)"""
        self.same_person_distances.extend(distances)

    def add_diff_person_distances(self, distances: List[float]):
        """添加跨人距离 (负样本, impostor pairs)"""
        self.diff_person_distances.extend(distances)

    def compute_roc(self, thresholds: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        计算 ROC 曲线数据

        Returns:
            {"thresholds": ..., "fpr": ..., "tpr": ..., "fnr": ...}
        """
        if thresholds is None:
            all_dists = self.same_person_distances + self.diff_person_distances
            if not all_dists:
                return {"thresholds": np.array([]), "fpr": np.array([]), "tpr": np.array([]), "fnr": np.array([])}
            thresholds = np.linspace(min(all_dists) - 0.1, max(all_dists) + 0.1, 1000)

        tprs = []  # True Positive Rate (TPR) = Recall
        fprs = []  # False Positive Rate (FPR)

        for thresh in thresholds:
            # 同人中被正确判定为匹配的比率 (TPR)
            tp = sum(1 for d in self.same_person_distances if d < thresh)
            tpr = tp / len(self.same_person_distances) if self.same_person_distances else 0

            # 跨人被错误判定为匹配的比率 (FPR)
            fp = sum(1 for d in self.diff_person_distances if d < thresh)
            fpr = fp / len(self.diff_person_distances) if self.diff_person_distances else 0

            tprs.append(tpr)
            fprs.append(fpr)

        return {
            "thresholds": thresholds,
            "tpr": np.array(tprs),
            "fpr": np.array(fprs),
            "fnr": 1.0 - np.array(tprs),  # False Negative Rate
        }

    def compute_eer(self) -> Dict[str, float]:
        """
        计算 Equal Error Rate (EER)
        EER = 使 FPR == FNR 的阈值处的错误率

        Returns:
            {"eer": ..., "threshold": ..., "far": ..., "frr": ...}
        """
        roc = self.compute_roc()
        if len(roc["thresholds"]) == 0:
            return {"eer": 0.5, "threshold": 0.5, "far": 0.5, "frr": 0.5}

        # FPR 和 FNR 的交点
        fpr = roc["fpr"]
        fnr = roc["fnr"]
        thresholds = roc["thresholds"]

        # 找到 FPR 和 FNR 最接近的点
        diff = np.abs(fpr - fnr)
        idx = np.argmin(diff)

        eer = float((fpr[idx] + fnr[idx]) / 2)
        return {
            "eer": eer,
            "threshold": float(thresholds[idx]),
            "far": float(fpr[idx]),
            "frr": float(fnr[idx]),
        }

    def find_optimal_threshold(
        self,
        method: str = "youden",
        cost_fn: float = 1.0,
        cost_fp: float = 1.0,
    ) -> Dict[str, Any]:
        """
        寻找最优阈值

        Args:
            method: "youden" | "eer" | "cost"
                - youden: Youden's J 统计量 (最大化 TPR - FPR)
                - eer: Equal Error Rate
                - cost: 自定义代价函数
            cost_fn: 假阴性代价 (同人被误判为不同)
            cost_fp: 假阳性代价 (跨人被误判为相同)

        Returns:
            最优阈值和相关指标
        """
        if method == "eer":
            eer_result = self.compute_eer()
            return {
                "method": "eer",
                "threshold": eer_result["threshold"],
                "eer": eer_result["eer"],
                "far": eer_result["far"],
                "frr": eer_result["frr"],
            }

        elif method == "youden":
            roc = self.compute_roc()
            if len(roc["thresholds"]) == 0:
                return {"method": "youden", "threshold": 0.5, "j_statistic": 0}

            j_statistics = roc["tpr"] - roc["fpr"]
            idx = np.argmax(j_statistics)

            return {
                "method": "youden",
                "threshold": float(roc["thresholds"][idx]),
                "j_statistic": float(j_statistics[idx]),
                "tpr": float(roc["tpr"][idx]),
                "fpr": float(roc["fpr"][idx]),
            }

        elif method == "cost":
            roc = self.compute_roc()
            if len(roc["thresholds"]) == 0:
                return {"method": "cost", "threshold": 0.5, "total_cost": 0}

            # 代价 = cost_fn * FNR * n_genuine + cost_fp * FPR * n_impostor
            n_genuine = len(self.same_person_distances)
            n_impostor = len(self.diff_person_distances)

            costs = (
                cost_fn * roc["fnr"] * n_genuine
                + cost_fp * roc["fpr"] * n_impostor
            )
            idx = np.argmin(costs)

            return {
                "method": "cost",
                "threshold": float(roc["thresholds"][idx]),
                "total_cost": float(costs[idx]),
                "tpr": float(roc["tpr"][idx]),
                "fpr": float(roc["fpr"][idx]),
            }

        else:
            raise ValueError(f"未知方法: {method}")

    def plot_roc_curve(self, save_path: str = None):
        """绘制 ROC 曲线"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("请安装 matplotlib: pip install matplotlib")
            return

        roc = self.compute_roc()
        if len(roc["thresholds"]) == 0:
            print("无数据可绘制")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ROC 曲线
        ax1.plot(roc["fpr"], roc["tpr"], 'b-', linewidth=2, label='ROC Curve')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax1.set_xlabel('False Positive Rate (FPR)')
        ax1.set_ylabel('True Positive Rate (TPR)')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 标注 EER
        eer_result = self.compute_eer()
        ax1.plot(eer_result["far"], 1 - eer_result["frr"], 'ro',
                 label=f'EER = {eer_result["eer"]:.4f}')
        ax1.legend()

        # 阈值分布
        ax2.hist(self.same_person_distances, bins=50, alpha=0.5,
                 label='Same Person (Genuine)', color='green', density=True)
        ax2.hist(self.diff_person_distances, bins=50, alpha=0.5,
                 label='Diff Person (Impostor)', color='red', density=True)

        # 最优阈值线
        opt = self.find_optimal_threshold()
        ax2.axvline(x=opt["threshold"], color='blue', linestyle='--',
                    label=f'Optimal Threshold = {opt["threshold"]:.4f}')

        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Density')
        ax2.set_title('Distance Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC 曲线已保存至: {save_path}")
        else:
            plt.show()

        plt.close()

    def analyze_test_data(self) -> str:
        """生成测试数据分析报告"""
        n_genuine = len(self.same_person_distances)
        n_impostor = len(self.diff_person_distances)

        if n_genuine == 0 or n_impostor == 0:
            return "数据不足，请添加正负样本对"

        report = []
        report.append("=" * 60)
        report.append("阈值分析报告")
        report.append("=" * 60)
        report.append(f"同人距离对: {n_genuine}")
        report.append(f"跨人距离对: {n_impostor}")
        report.append("")

        report.append("同人距离统计:")
        report.append(f"  均值: {np.mean(self.same_person_distances):.4f}")
        report.append(f"  标准差: {np.std(self.same_person_distances):.4f}")
        report.append(f"  最小值: {np.min(self.same_person_distances):.4f}")
        report.append(f"  最大值: {np.max(self.same_person_distances):.4f}")
        report.append("")

        report.append("跨人距离统计:")
        report.append(f"  均值: {np.mean(self.diff_person_distances):.4f}")
        report.append(f"  标准差: {np.std(self.diff_person_distances):.4f}")
        report.append(f"  最小值: {np.min(self.diff_person_distances):.4f}")
        report.append(f"  最大值: {np.max(self.diff_person_distances):.4f}")
        report.append("")

        # 各方法最优阈值
        for method in ["youden", "eer", "cost"]:
            result = self.find_optimal_threshold(method=method)
            report.append(f"{method.upper()} 方法:")
            report.append(f"  阈值: {result['threshold']:.4f}")
            if "eer" in result:
                report.append(f"  EER: {result['eer']:.4f}")
            if "j_statistic" in result:
                report.append(f"  TPR: {result['tpr']:.4f}, FPR: {result['fpr']:.4f}")
            report.append("")

        # 危险区域分析: 两个分布重叠的部分
        genuine_max = np.max(self.same_person_distances)
        impostor_min = np.min(self.diff_person_distances)
        overlap = max(0, genuine_max - impostor_min)
        report.append("危险区域分析:")
        report.append(f"  同人最大距离: {genuine_max:.4f}")
        report.append(f"  跨人最小距离: {impostor_min:.4f}")
        report.append(f"  重叠区域: {overlap:.4f}")

        if overlap > 0:
            report.append("  *** 警告: 存在无法通过阈值区分的情况 ***")
            report.append("  建议使用多模型融合或辅助特征方案")

        report.append("=" * 60)
        return "\n".join(report)


# ============================================================
# 综合使用示例
# ============================================================

class AdvancedFaceRecognizer:
    """
    高级人脸识别器 - 综合方案 A+B+C+D

    整合所有优化策略:
    1. 多模型融合 (A)
    2. 更强模型选择 (B)
    3. 辅助特征验证 (C)
    4. 自适应阈值 (D)

    典型用法:
        recognizer = AdvancedFaceRecognizer(
            fusion_strategy="weighted",
            model_name="antelopev2",
            use_landmarks=True,
        )

        # 自动优化阈值
        optimizer = recognizer.optimize_threshold()
        optimizer.add_same_person_distances([...])
        optimizer.add_diff_person_distances([...])
        opt_result = optimizer.find_optimal_threshold()

        # 识别
        result = recognizer.recognize(image, known_faces_db)
    """

    def __init__(
        self,
        fusion_strategy: str = "weighted",
        model_name: str = "buffalo_l",
        use_landmarks: bool = False,
        model_weights: Optional[Dict[str, float]] = None,
    ):
        # 方案 B: 模型选择
        self.if_recognizer = MultiModelRecognizer(model_name=model_name)
        self.model_name = model_name

        # 方案 A: 融合策略
        self.fusion_strategy = fusion_strategy
        self.fusion = ModelFusionStrategy(
            strategy=fusion_strategy,
            model_weights=model_weights,
        )

        # 方案 C: 辅助特征
        self.use_landmarks = use_landmarks
        self.landmark_assistant = LandmarkAssistant() if use_landmarks else None

        # 方案 D: 阈值优化
        self.threshold_optimizer = ThresholdOptimizer()

        # 数据库
        self.face_database: Dict[str, Dict[str, Any]] = {}

    def register_face(
        self,
        face_id: str,
        image: np.ndarray,
        fr_encoding: np.ndarray = None,
        name: str = None,
    ):
        """注册人脸 (同时提取多种特征)"""
        # InsightFace 特征
        if_encoding = self.if_recognizer.extract_embedding(image)

        if if_encoding is None:
            return None

        self.face_database[face_id] = {
            "name": name or face_id,
            "if_encoding": if_encoding,
            "fr_encoding": fr_encoding,
            "image": image,  # 用于辅助特征提取
            "registered_at": datetime.now().isoformat(),
        }

        return self.face_database[face_id]

    def compare_faces_comprehensive(
        self,
        if_enc1: np.ndarray, if_enc2: np.ndarray,
        fr_enc1: np.ndarray = None, fr_enc2: np.ndarray = None,
        image1: np.ndarray = None, image2: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        综合比较 (A + B + C)

        Returns:
            包含所有维度的比较结果
        """
        result = {
            "if_comparison": self.if_recognizer.compare_faces(if_enc1, if_enc2),
        }

        # face_recognition 比较
        if fr_enc1 is not None and fr_enc2 is not None:
            dist = float(np.linalg.norm(fr_enc1.flatten() - fr_enc2.flatten()))
            result["fr_comparison"] = {
                "distance": dist,
                "is_match": dist < 0.45,
            }

        # 辅助特征比较
        if self.use_landmarks and image1 is not None and image2 is not None:
            result["auxiliary"] = self._compare_auxiliary(image1, image2)

        # 融合判定
        scores = []
        if_enc1_norm = if_enc1 / np.linalg.norm(if_enc1)
        if_enc2_norm = if_enc2 / np.linalg.norm(if_enc2)
        if_sim = float(np.dot(if_enc1_norm, if_enc2_norm))
        if_dist = 1.0 - if_sim

        scores.append(ModelScore(
            model_name="insightface",
            distance=if_dist,
            similarity=if_sim,
            threshold=0.6 if self.model_name == "antelopev2" else 1.0,
            is_match=if_dist < (0.6 if self.model_name == "antelopev2" else 1.0),
            confidence=max(0, 1.0 - if_dist),
        ))

        if "fr_comparison" in result:
            fr_dist = result["fr_comparison"]["distance"]
            scores.append(ModelScore(
                model_name="face_recognition",
                distance=fr_dist,
                similarity=1.0 - fr_dist,
                threshold=0.45,
                is_match=fr_dist < 0.45,
                confidence=max(0, 1.0 - fr_dist / 0.45),
            ))

        ensemble = self.fusion.fuse(scores)
        result["ensemble"] = ensemble
        result["is_match"] = ensemble.is_match
        result["certainty"] = ensemble.certainty
        result["fused_distance"] = ensemble.fused_distance
        result["fused_similarity"] = ensemble.fused_similarity

        return result

    def _compare_auxiliary(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
        """提取和比较辅助特征"""
        aux = {}

        # 需要检测人脸并获取关键点
        faces1 = self.if_recognizer.app.get(image1)
        faces2 = self.if_recognizer.app.get(image2)

        if faces1 and faces2:
            lm_dist = self.landmark_assistant.compute_landmark_distance(
                faces1[0].landmark, faces2[0].landmark
            )
            prop_dist = self.landmark_assistant.compute_proportion_distance(
                faces1[0].landmark, faces2[0].landmark
            )

            # 裁剪人脸区域用于肤色比较
            for img, faces, prefix in [(image1, faces1, "1"), (image2, faces2, "2")]:
                bbox = faces[0].bbox.astype(int)
                exec(f'aux["roi_{prefix}"] = True')

            aux["landmark_distance"] = lm_dist
            aux["proportion_distance"] = prop_dist

        return aux

    def optimize_threshold(self) -> ThresholdOptimizer:
        """获取阈值优化器"""
        return self.threshold_optimizer


# ============================================================
# 测试与使用示例
# ============================================================

def demo_all_strategies():
    """演示所有方案"""
    print("=" * 60)
    print("人脸识别高级策略演示")
    print("=" * 60)

    # ---- 方案 D: 阈值优化演示 ----
    print("\n--- 方案 D: 自适应阈值选择 ---")

    optimizer = ThresholdOptimizer()

    # 模拟数据 (同人距离)
    same_person_dists = [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30,
                         0.32, 0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.48]

    # 模拟数据 (跨人距离)
    diff_person_dists = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
                         0.90, 0.93, 0.95, 0.97, 1.00, 1.05, 1.10, 1.20]

    optimizer.add_same_person_distances(same_person_dists)
    optimizer.add_diff_person_distances(diff_person_dists)

    print(optimizer.analyze_test_data())

    # ---- 方案 A: 模型融合演示 ----
    print("\n--- 方案 A: 多模型融合 ---")

    # 模拟 jiang vs xuefen 的场景
    fr_dist = 0.4499  # face_recognition 欧氏距离
    if_sim = 0.9371   # InsightFace 余弦相似度
    if_dist = 1.0 - if_sim  # 0.0629

    # 情况1: 两个模型都接近阈值 (边缘情况)
    print(f"模拟场景: face_recognition 距离={fr_dist:.4f}, InsightFace 距离={if_dist:.4f}")

    # 投票策略
    fusion_voting = ModelFusionStrategy(strategy="voting")
    scores_voting = [
        ModelScore("face_recognition", fr_dist, 1-fr_dist, 0.45, fr_dist < 0.45, 0.5),
        ModelScore("insightface", if_dist, if_sim, 1.0, if_dist < 1.0, 0.94),
    ]
    result_voting = fusion_voting.fuse(scores_voting)
    print(f"  投票策略: is_match={result_voting.is_match}, "
          f"certainty={result_voting.certainty}, votes={result_voting.vote_count}/{result_voting.vote_total}")

    # 加权策略 (InsightFace 权重更高)
    fusion_weighted = ModelFusionStrategy(
        strategy="weighted",
        model_weights={"face_recognition": 0.3, "insightface": 0.7}
    )
    result_weighted = fusion_weighted.fuse(scores_voting)
    print(f"  加权策略: is_match={result_weighted.is_match}, "
          f"certainty={result_weighted.certainty}, "
          f"fused_score={result_weighted.fused_similarity:.4f}")

    # ---- 方案 B: 更强模型演示 ----
    print("\n--- 方案 B: 更强模型 ---")
    for model_name, info in MultiModelRecognizer.SUPPORTED_MODELS.items():
        print(f"  {model_name}: {info['arch']}, {info['size_mb']}MB - {info['desc']}")

    # ---- 方案 C: 辅助特征演示 ----
    print("\n--- 方案 C: 辅助特征融合 ---")

    landmark_assistant = LandmarkAssistant()
    fusion_result = landmark_assistant.fuse_with_depth(
        depth_distance=0.0629,       # InsightFace 距离
        landmark_distance=0.15,      # 假设关键点距离
        proportion_distance=0.08,    # 假设比例距离
    )
    print(f"  融合结果: is_match={fusion_result['is_match']}, "
          f"certainty={fusion_result['certainty']}, "
          f"fused_distance={fusion_result['fused_distance']:.4f}")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_all_strategies()
