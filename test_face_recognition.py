#!/usr/bin/env python3
"""
人脸识别测试脚本

使用 test_data 中的人脸数据进行测试：
1. 注册每个人第一张照片
2. 用剩余照片进行识别测试
3. 输出识别准确率统计
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# 添加 backend 路径
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from face_recognition_module import FaceRecognizerService

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def main():
    print("=" * 60)
    print("人脸识别测试")
    print("=" * 60)
    
    # 初始化服务
    service = FaceRecognizerService()
    
    # 清空已有数据库
    service.clear_all_faces()
    
    # 获取所有人物文件夹
    persons = [d for d in TEST_DATA_DIR.iterdir() if d.is_dir()]
    
    if not persons:
        print(f"错误：未在 {TEST_DATA_DIR} 中找到测试数据")
        return
    
    print(f"\n找到 {len(persons)} 个人物：{[p.name for p in persons]}")
    
    # ========== 第一步：注册 ==========
    print("\n" + "=" * 60)
    print("第一步：注册人脸（使用每个人的第一张照片）")
    print("=" * 60)
    
    registered_count = 0
    for person_dir in persons:
        images = sorted(person_dir.glob("*.png")) + sorted(person_dir.glob("*.jpg"))
        if not images:
            print(f"  ⚠️  {person_dir.name}: 未找到图片")
            continue
        
        # 使用第一张照片注册
        first_image = images[0]
        image = cv2.imread(str(first_image))
        
        if image is None:
            print(f"  ❌ {person_dir.name}: 无法读取 {first_image.name}")
            continue
        
        result = service.register_face(image, face_id=person_dir.name, name=person_dir.name, face_index=0)
        
        if result:
            registered_count += 1
            print(f"  ✅ {person_dir.name}: 注册成功 (使用 {first_image.name})")
        else:
            print(f"  ❌ {person_dir.name}: 注册失败")
    
    print(f"\n共注册 {registered_count} 个人脸")
    
    # ========== 第二步：识别测试 ==========
    print("\n" + "=" * 60)
    print("第二步：识别测试（使用每个人的剩余照片）")
    print("=" * 60)
    
    total_tests = 0
    correct_predictions = 0
    wrong_predictions = 0
    no_face_detected = 0
    
    for person_dir in persons:
        images = sorted(person_dir.glob("*.png")) + sorted(person_dir.glob("*.jpg"))
        if len(images) <= 1:
            continue
        
        print(f"\n--- {person_dir.name} ---")
        
        # 从第二张开始测试
        for test_image_path in images[1:]:
            test_image = cv2.imread(str(test_image_path))
            
            if test_image is None:
                print(f"  ⚠️  无法读取 {test_image_path.name}")
                continue
            
            # 执行识别
            results = service.recognize_faces(test_image)
            
            if not results:
                print(f"  ⚠️  {test_image_path.name}: 未检测到人脸")
                no_face_detected += 1
                total_tests += 1
                continue
            
            # 检查识别结果
            # 取第一个检测结果（假设每张图只有一个人脸）
            first_result = results[0]
            matches = first_result.get("matches", [])
            
            if matches:
                best_match = matches[0]  # 最匹配的
                predicted_name = best_match["name"]
                similarity = best_match["similarity"]
                
                is_correct = (predicted_name == person_dir.name)
                if is_correct:
                    correct_predictions += 1
                    print(f"  ✅ {test_image_path.name}: 识别为 {predicted_name} (相似度: {similarity:.2%})")
                else:
                    wrong_predictions += 1
                    print(f"  ❌ {test_image_path.name}: 错误识别为 {predicted_name} (应为 {person_dir.name}, 相似度: {similarity:.2%})")
            else:
                wrong_predictions += 1
                print(f"  ❌ {test_image_path.name}: 未匹配到已知人脸")
            
            total_tests += 1
    
    # ========== 第三步：统计结果 ==========
    print("\n" + "=" * 60)
    print("第三步：测试统计")
    print("=" * 60)
    
    if total_tests == 0:
        print("没有进行任何测试")
        return
    
    accuracy = correct_predictions / total_tests * 100
    
    print(f"\n总测试次数: {total_tests}")
    print(f"✅ 正确识别: {correct_predictions}")
    print(f"❌ 错误识别: {wrong_predictions}")
    print(f"⚠️  未检测到人脸: {no_face_detected}")
    print(f"\n📊 识别准确率: {accuracy:.1f}%")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
