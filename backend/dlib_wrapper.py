#!/usr/bin/env python3
"""dlib 人脸特征提取服务 - 通过 stdin/stdout 通信以避免重复启动开销"""
import sys
import json
import cv2
import dlib
import numpy as np
import face_recognition_models

# 预加载模型
predictor_path = face_recognition_models.__file__.replace('__init__.py', 'models/shape_predictor_68_face_landmarks.dat')
model_path = face_recognition_models.__file__.replace('__init__.py', 'models/dlib_face_recognition_resnet_model_v1.dat')
predictor = dlib.shape_predictor(predictor_path)
encoder = dlib.face_recognition_model_v1(model_path)

def extract_encodings(image_path, face_rect=None):
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    h, w = img.shape[:2]
    
    # 如果提供了人脸坐标，直接使用
    if face_rect:
        x, y, fw, fh = face_rect
    else:
        # 否则使用 OpenCV DNN 检测
        import os
        detector_path = os.path.join(os.path.dirname(__file__), 'models', 'face_detection_yunet_2023mar.onnx')
        if not os.path.exists(detector_path):
            detector_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_detection_yunet_2023mar.onnx')
        if not os.path.exists(detector_path):
            detector_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'face_detection_yunet_2023mar.onnx')
            
        if not os.path.exists(detector_path):
            return []
        
        detector = cv2.FaceDetectorYN_create(detector_path, '', (w, h), 0.6, 0.3)
        _, dets = detector.detect(img)
        
        if dets is None or len(dets) == 0:
            return []
            
        # 使用第一个检测到的人脸
        x, y, fw, fh = dets[0][:4].astype(int)
    
    rect = dlib.rectangle(x, y, x + fw, y + fh)
    landmarks = predictor(img, rect)
    encoding = encoder.compute_face_descriptor(img, landmarks, 0, 0.25)
    return [list(encoding)]

if __name__ == "__main__":
    # 服务模式：从 stdin 读取 JSON，输出 JSON 结果
    if "--service" in sys.argv:
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                    
                # 支持 JSON 格式或简单路径
                if line.startswith('{'):
                    data = json.loads(line)
                    image_path = data['image_path']
                    face_rect = data.get('face_rect')
                else:
                    image_path = line
                    face_rect = None
                    
                encodings = extract_encodings(image_path, face_rect)
                print(json.dumps(encodings), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)
    else:
        # 单次模式
        if len(sys.argv) < 2:
            sys.exit(1)
        image_path = sys.argv[1]
        face_rect = None
        if len(sys.argv) >= 6:
            face_rect = [int(sys.argv[i]) for i in range(2, 6)]
        encodings = extract_encodings(image_path, face_rect)
        print(json.dumps(encodings))
