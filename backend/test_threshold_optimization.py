#!/usr/bin/env python3
"""
阈值优化测试工具

基于实际测试数据自动选择最优人脸识别阈值。
支持两种模型的阈值分析:
  - face_recognition (欧氏距离)
  - InsightFace ArcFace (余弦距离)

用法:
    # 方式 1: 命令行
    python backend/test_threshold_optimization.py --model insightface

    # 方式 2: 编程接口
    from backend.test_threshold_optimization import run_threshold_analysis
    result = run_threshold_analysis(same_dists, diff_dists, model="insightface")
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# 导入方案 D 的阈值优化器
from backend.face_advanced_strategies import ThresholdOptimizer


def load_test_data_from_test_dir(test_data_dir: str = "test_data") -> Dict[str, List[float]]:
    """
    从 test_data 目录加载实际的测试结果
    假设有 precomputed_distances.json 文件或类似格式
    """
    json_path = Path(test_data_dir) / "precomputed_distances.json"
    if json_path.exists():
        import json
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def simulate_jiang_xuefen_data() -> Dict[str, List[float]]:
    """
    模拟 jiang vs xuefen 的测试数据

    基于已知的测量结果:
    - face_recognition: 跨人 0.4499, 同人 ~0.20-0.35
    - InsightFace: 跨人 0.0629 (sim=0.9371), 同人 ~0.01-0.04
    """
    return {
        "face_recognition": {
            "same_person": [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40],
            "diff_person": [0.45, 0.48, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            "edge_case": {
                "jiang_vs_xuefen": 0.4499,  # 非常接近默认阈值
            }
        },
        "insightface": {
            "same_person": [0.005, 0.008, 0.012, 0.015, 0.018, 0.022, 0.025, 0.030, 0.035, 0.040],
            "diff_person": [0.063, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0],
            "edge_case": {
                "jiang_vs_xuefen": 0.0629,  # 余弦距离 = 1 - 0.9371
            }
        }
    }


def run_threshold_analysis(
    same_person_dists: List[float],
    diff_person_dists: List[float],
    model: str = "insightface",
    save_plot: bool = True,
    output_dir: str = "docs",
) -> Dict[str, Any]:
    """
    运行完整的阈值分析

    Args:
        same_person_dists: 同人距离列表
        diff_person_dists: 跨人距离列表
        model: 模型名称
        save_plot: 是否保存 ROC 曲线图
        output_dir: 输出目录

    Returns:
        分析结果
    """
    optimizer = ThresholdOptimizer()
    optimizer.add_same_person_distances(same_person_dists)
    optimizer.add_diff_person_distances(diff_person_dists)

    # 生成报告
    report = optimizer.analyze_test_data()
    print(report)

    # 绘制 ROC 曲线
    if save_plot:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_path = Path(output_dir) / f"roc_curve_{model}.png"
        optimizer.plot_roc_curve(str(plot_path))

    # 各方法的最优阈值
    results = {}
    for method in ["youden", "eer", "cost"]:
        results[method] = optimizer.find_optimal_threshold(method=method)

    return results


def main():
    parser = argparse.ArgumentParser(description="人脸识别阈值优化分析")
    parser.add_argument(
        "--model",
        choices=["face_recognition", "insightface", "both"],
        default="both",
        help="要分析的模型"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="测试数据 JSON 文件路径 (不提供则使用模拟数据)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs",
        help="输出目录"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不生成 ROC 曲线图"
    )
    args = parser.parse_args()

    # 加载数据
    if args.data_file:
        data = load_test_data_from_test_dir(args.data_file)
        if data is None:
            print(f"无法加载数据: {args.data_file}, 使用模拟数据")
            data = simulate_jiang_xuefen_data()
    else:
        data = simulate_jiang_xuefen_data()

    # 分析
    models_to_analyze = []
    if args.model in ["face_recognition", "both"]:
        models_to_analyze.append("face_recognition")
    if args.model in ["insightface", "both"]:
        models_to_analyze.append("insightface")

    for model_name in models_to_analyze:
        model_data = data[model_name]
        same_dists = model_data["same_person"]
        diff_dists = model_data["diff_person"]

        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")

        results = run_threshold_analysis(
            same_dists, diff_dists,
            model=model_name,
            save_plot=not args.no_plot,
            output_dir=args.output_dir,
        )

        # 检查边缘案例
        edge_cases = model_data.get("edge_case", {})
        for name, dist in edge_cases.items():
            print(f"\n边缘案例检查: {name}")
            print(f"  实际距离: {dist:.4f}")

            for method, result in results.items():
                threshold = result["threshold"]
                would_match = dist < threshold
                print(f"  {method} 阈值={threshold:.4f} -> {'匹配' if would_match else '不匹配'}")


if __name__ == "__main__":
    main()
