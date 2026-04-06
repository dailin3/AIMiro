"""
AI 图像分析模块单元测试
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from backend.ai_image_analysis import (
    AIImageAnalyzer,
    AnalysisConfig,
    load_env_file,
    load_prompt_file
)


class TestAIImageAnalyzer:
    """测试 AI 图像分析器"""

    @pytest.fixture
    def analyzer(self):
        """创建分析器实例 (无 API Key)"""
        config = AnalysisConfig(api_key="test_key_123")
        return AIImageAnalyzer(config)

    @pytest.fixture
    def test_image(self, test_data_dir):
        """获取测试图像"""
        # 使用 nothuman 目录下的图像 (不需要人脸)
        nothuman_dir = test_data_dir / "nothuman"
        if nothuman_dir.exists():
            for img_file in nothuman_dir.glob("*.png"):
                return img_file
        
        # 如果没有,使用 human 目录
        human_dir = test_data_dir / "human" / "jiang"
        if human_dir.exists():
            for img_file in human_dir.glob("*.png"):
                return img_file
        
        pytest.skip("没有可用的测试图像")

    def test_analyzer_initialization(self, analyzer):
        """测试分析器初始化"""
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.config.model == "gemini-3-flash-preview"

    @patch('requests.post')
    def test_analyze_image(self, mock_post, analyzer, test_image):
        """测试单张图像分析 (使用 mock Gemini API)"""
        # 配置 mock 响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "这是一张测试图片,显示了一个场景。"}]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 200,
                "totalTokenCount": 300
            }
        }
        mock_post.return_value = mock_response

        # 执行分析
        result = analyzer.analyze(test_image)

        assert result["success"] == True
        assert "analysis" in result
        assert result["model"] == "gemini-3-flash-preview"
        assert "usage" in result

    @patch('requests.post')
    def test_analyze_image_with_prompt(self, mock_post, analyzer, test_image):
        """测试自定义提示分析"""
        # 配置 mock 响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "自定义提示的响应"}]
                }
            }],
            "usageMetadata": {}
        }
        mock_post.return_value = mock_response

        custom_prompt = "请描述这张图片中的所有内容"
        result = analyzer.analyze(test_image, prompt=custom_prompt)

        assert result["success"] == True
        
        # 验证请求中包含了自定义提示
        call_args = mock_post.call_args
        request_body = call_args.kwargs.get('json', {})
        assert custom_prompt in str(request_body)

    @patch('requests.post')
    def test_analyze_batch(self, mock_post, analyzer, test_image):
        """测试批量分析"""
        # 配置 mock 响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "批量分析结果"}]
                }
            }],
            "usageMetadata": {}
        }
        mock_post.return_value = mock_response

        # 批量分析 (同一张图像2次)
        image_paths = [test_image, test_image]
        results = analyzer.analyze_batch(image_paths)

        assert len(results) == 2
        assert all(r["success"] == True for r in results)

    @patch('requests.post')
    def test_compare_images(self, mock_post, analyzer, test_image):
        """测试图像比较"""
        # 配置 mock 响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "这两张图片非常相似"}]
                }
            }],
            "usageMetadata": {}
        }
        mock_post.return_value = mock_response

        # 比较图像
        result = analyzer.compare_images([test_image, test_image])

        assert result["success"] == True
        assert "comparison" in result
        assert len(result["image_paths"]) == 2

    @patch('requests.post')
    def test_extract_features(self, mock_post, analyzer, test_image):
        """测试特征提取"""
        # 配置 mock 响应 (返回 JSON 格式)
        features_json = json.dumps({
            "objects": ["人", "桌子", "电脑"],
            "scene": "办公室",
            "colors": ["蓝色", "白色"],
            "mood": "专业",
            "text_content": "",
            "tags": ["办公", "工作"]
        })
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": f"分析结果\n\n{features_json}"}]
                }
            }],
            "usageMetadata": {}
        }
        mock_post.return_value = mock_response

        result = analyzer.extract_features(test_image)

        assert result["success"] == True
        # 应该尝试解析 JSON
        assert "features" in result

    def test_load_env_file(self, tmp_path):
        """测试环境变量加载"""
        # 创建临时 .env 文件
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=test_value\n# 注释\nINVALID_LINE\nKEY2=value2")
        
        # 加载环境变量
        load_env_file(str(env_file))
        
        assert os.environ.get("TEST_KEY") == "test_value"
        assert os.environ.get("KEY2") == "value2"

    def test_load_prompt_file(self, tmp_path):
        """测试提示词文件加载"""
        # 创建临时 prompt 文件
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("# 自定义提示\n\n请详细描述这张图片")
        
        # 加载提示词
        prompt = load_prompt_file(str(prompt_file))
        
        assert "请详细描述这张图片" in prompt

    def test_load_prompt_file_not_exists(self):
        """测试提示词文件不存在时使用默认"""
        prompt = load_prompt_file("/nonexistent/prompt.md")
        
        # 应该返回默认提示词
        assert prompt != ""
        assert "请描述" in prompt or "描述" in prompt

    @patch('requests.post')
    def test_analyze_image_api_error(self, mock_post, analyzer, test_image):
        """测试 API 错误处理"""
        # 配置 mock 返回错误
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response

        result = analyzer.analyze(test_image, retries=1)

        assert result["success"] == False
        assert "error" in result

    def test_analyze_image_file_not_exists(self, analyzer):
        """测试图像文件不存在"""
        # 该测试会抛出 FileNotFoundError,因为模块没有在不存在文件时提前检查
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/nonexistent/image.jpg")

    def test_compare_images_less_than_two(self, analyzer, test_image):
        """测试比较图像数量不足"""
        with pytest.raises(ValueError, match="至少需要 2 张"):
            analyzer.compare_images([test_image])

    def test_get_image_mime_type(self, analyzer):
        """测试 MIME 类型获取"""
        assert analyzer._get_image_mime_type("test.jpg") == "image/jpeg"
        assert analyzer._get_image_mime_type("test.png") == "image/png"
        assert analyzer._get_image_mime_type("test.gif") == "image/gif"
        assert analyzer._get_image_mime_type("test.unknown") == "image/jpeg"  # 默认
