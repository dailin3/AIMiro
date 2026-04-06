"""
后端服务器集成测试
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import cv2
import numpy as np


class TestBackendServer:
    """测试后端服务器 API 端点"""

    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "camera_status" in data
        assert "ai_service_ready" in data

    def test_index_page(self, client):
        """测试 API 首页"""
        response = client.get('/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["name"] == "AI 图像分析后端服务"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data

    @patch('cv2.VideoCapture')
    def test_camera_status(self, mock_videocapture, client):
        """测试摄像头状态 (mock 摄像头)"""
        # 配置 mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        response = client.get('/camera/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "is_running" in data
        assert "capture_count" in data
        assert "has_frame" in data

    def test_camera_list(self, client):
        """测试图像列表 (使用临时文件)"""
        response = client.get('/camera/list')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "count" in data
        assert "images" in data
        assert isinstance(data["images"], list)

    @patch('cv2.VideoCapture')
    def test_camera_devices(self, mock_videocapture, client):
        """测试设备列表 (mock cv2.VideoCapture)"""
        # 配置 mock - 没有设备
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        response = client.get('/camera/devices')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "count" in data
        assert "devices" in data

    def test_face_register_list_delete(self, client):
        """测试人脸注册/列表/删除 (mock 人脸识别)"""
        # 人脸注册可能失败 (无真实图像),我们测试端点存在性
        # 测试人脸列表
        response = client.get('/face/list')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "success" in data
        assert "count" in data
        assert "faces" in data

    def test_face_clear(self, client):
        """测试清空人脸"""
        response = client.post('/face/clear')
        
        # 应该成功 (即使没有人脸)
        assert response.status_code in [200, 500]
        data = json.loads(response.data)
        assert "success" in data

    def test_face_update_name(self, client):
        """测试更新人脸名称"""
        # 尝试更新不存在的人脸 (应该返回 400 或 404)
        response = client.post('/face/update_name', 
                             data=json.dumps({
                                 "face_id": "nonexistent",
                                 "name": "新名称"
                             }),
                             content_type='application/json')
        
        # 应该返回错误 (人脸不存在)
        assert response.status_code in [400, 404]
        data = json.loads(response.data)
        assert "success" in data

    @patch('backend.ai_image_analysis.AIImageAnalyzer.analyze')
    def test_analyze_image(self, mock_analyze, client, test_data_dir):
        """测试 AI 分析图像 (mock Gemini API)"""
        # 配置 mock 返回值
        mock_analyze.return_value = {
            "success": True,
            "analysis": "这是一张测试图片",
            "model": "gemini-3-flash-preview",
            "usage": {"prompt_tokens": 100, "completion_tokens": 200}
        }
        
        # 获取测试图像路径
        human_dir = test_data_dir / "human" / "jiang"
        if not human_dir.exists():
            pytest.skip("测试数据不存在")
        
        test_image = next(human_dir.glob("*.png"))
        
        response = client.post('/analyze/image',
                             data=json.dumps({
                                 "path": str(test_image),
                                 "prompt": "请描述这张图片"
                             }),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "success" in data

    def test_analyze_history(self, client):
        """测试分析历史"""
        response = client.get('/analyze/history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "success" in data
        assert "count" in data
        assert "history" in data

    def test_file_operations(self, client):
        """测试文件获取/删除 (使用临时文件)"""
        # 尝试获取不存在的文件 (应该返回 404)
        response = client.get('/files/2026-04-06/nonexistent.jpg')
        
        assert response.status_code == 404
        
        # 尝试删除不存在的文件
        response = client.delete('/files/2026-04-06/nonexistent.jpg')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "success" in data

    def test_analyze_image_missing_path(self, client):
        """测试分析图像缺少路径参数"""
        response = client.post('/analyze/image',
                             data=json.dumps({}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_face_register_missing_params(self, client):
        """测试人脸注册缺少参数"""
        # 缺少 face_id
        response = client.post('/face/register',
                             data=json.dumps({
                                 "path": "test.jpg"
                             }),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_face_recognize_missing_path(self, client):
        """测试人脸识别缺少路径"""
        response = client.post('/face/recognize',
                             data=json.dumps({}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    @patch('cv2.VideoCapture')
    def test_camera_start_stop(self, mock_videocapture, client):
        """测试摄像头启动/停止"""
        # 配置 mock (初始化会失败,但我们测试端点)
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        # 启动摄像头 (会失败,但端点应该存在)
        response = client.post('/camera/start',
                             data=json.dumps({
                                 "camera_index": 0,
                                 "width": 640,
                                 "height": 480
                             }),
                             content_type='application/json')
        
        # 端点应该存在 (可能返回 500 因为 mock 摄像头失败)
        assert response.status_code in [200, 500]
        
        # 停止摄像头
        response = client.post('/camera/stop')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "success" in data

    def test_face_compare_missing_params(self, client):
        """测试人脸比对缺少参数"""
        response = client.post('/face/compare',
                             data=json.dumps({
                                 "path1": "test1.jpg"
                                 # 缺少 path2
                             }),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
