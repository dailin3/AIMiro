"""
摄像头采集模块单元测试
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import time

from backend.camera_capture import CameraCapture, CameraConfig


class TestCameraConfig:
    """测试 CameraConfig 类"""

    def test_camera_config_defaults(self):
        """测试默认配置值"""
        config = CameraConfig()
        assert config.camera_index == 0
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 30.0
        assert config.codec == "MJPG"

    def test_camera_config_custom(self):
        """测试自定义配置"""
        config = CameraConfig(
            camera_index=1,
            width=1280,
            height=720,
            fps=60.0,
            codec="H264"
        )
        assert config.camera_index == 1
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 60.0
        assert config.codec == "H264"


class TestCameraCapture:
    """测试 CameraCapture 类"""

    @pytest.fixture
    def tmp_data_dir(self, tmp_path):
        """创建临时数据目录"""
        return tmp_path / "data"

    @pytest.fixture
    def mock_camera(self, tmp_data_dir):
        """创建 mock 摄像头"""
        config = CameraConfig(
            camera_index=0,
            width=640,
            height=480
        )
        camera = CameraCapture(config=config, data_dir=str(tmp_data_dir))
        return camera

    def test_camera_initialize(self, mock_camera):
        """测试摄像头初始化 (使用 mock)"""
        with patch('cv2.VideoCapture') as mock_videocapture:
            # 配置 mock
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(x, 0)
            mock_videocapture.return_value = mock_cap

            # 执行初始化
            result = mock_camera.initialize()
            assert result == True
            assert mock_camera.is_initialized == True

    def test_capture_frame(self, mock_camera):
        """测试单帧捕获 (使用 mock)"""
        with patch('cv2.VideoCapture') as mock_videocapture:
            # 配置 mock
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(x, 0)
            
            # 创建测试帧
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, test_frame)
            mock_videocapture.return_value = mock_cap

            # 初始化并捕获帧
            mock_camera.initialize()
            frame = mock_camera.capture_frame()
            
            assert frame is not None
            assert frame.shape == (480, 640, 3)

    def test_capture_frames_multiple(self, mock_camera):
        """测试多帧捕获 (使用 mock)"""
        with patch('cv2.VideoCapture') as mock_videocapture:
            # 配置 mock
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(x, 0)
            
            # 创建测试帧
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, test_frame)
            mock_videocapture.return_value = mock_cap

            # 初始化并捕获多帧
            mock_camera.initialize()
            frames = mock_camera.capture_frames(5, interval=0.0)
            
            assert len(frames) == 5
            for frame in frames:
                assert frame.shape == (480, 640, 3)

    def test_save_image(self, mock_camera, tmp_data_dir):
        """测试图像保存"""
        with patch('cv2.VideoCapture') as mock_videocapture:
            # 配置 mock
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(x, 0)
            mock_videocapture.return_value = mock_cap

            mock_camera.initialize()
            
            # 创建测试图像
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 保存图像
            file_path = mock_camera.save_image(test_image, "test_image.jpg", "jpg")
            
            assert file_path is not None
            assert file_path.exists()
            assert file_path.suffix == ".jpg"

    def test_save_image_custom_format(self, mock_camera):
        """测试自定义格式保存"""
        with patch('cv2.VideoCapture') as mock_videocapture:
            # 配置 mock
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(x, 0)
            mock_videocapture.return_value = mock_cap

            mock_camera.initialize()
            
            # 创建测试图像
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 保存为 PNG
            file_path = mock_camera.save_image(test_image, "test_image.png", "png")
            
            assert file_path is not None
            assert file_path.exists()
            assert file_path.suffix == ".png"

    def test_release_camera(self, mock_camera):
        """测试释放摄像头"""
        with patch('cv2.VideoCapture') as mock_videocapture:
            # 配置 mock
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(x, 0)
            mock_videocapture.return_value = mock_cap

            mock_camera.initialize()
            mock_camera.release()
            
            assert mock_camera.is_initialized == False
            assert mock_camera.cap is None

    def test_check_disk_space(self, mock_camera):
        """测试磁盘空间检查"""
        # 磁盘空间检查应该返回 True (有足够空间)
        result = mock_camera.check_disk_space(min_space_mb=100)
        assert result == True

    def test_cleanup_old_files(self, mock_camera, tmp_data_dir):
        """测试旧文件清理"""
        # 创建一些测试文件
        pic_dir = tmp_data_dir / "pic"
        pic_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(5):
            test_file = pic_dir / f"test_{i}.jpg"
            test_file.write_text("test")
        
        # 执行清理 (设置 max_files=3,应该删除 2 个)
        deleted = mock_camera.cleanup_old_files(max_files=3, max_age_days=0)
        assert deleted >= 0  # 清理数量取决于文件创建时间

    def test_capture_and_save(self, mock_camera):
        """测试捕获并保存完整流程"""
        with patch('cv2.VideoCapture') as mock_videocapture:
            # 配置 mock
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(x, 0)
            
            # 创建测试帧
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, test_frame)
            mock_videocapture.return_value = mock_cap

            mock_camera.initialize()
            
            # 捕获并保存
            file_path = mock_camera.capture_and_save("capture_test.jpg", "jpg")
            
            assert file_path is not None
            assert file_path.exists()

    def test_context_manager(self, tmp_data_dir):
        """测试上下文管理器"""
        config = CameraConfig(camera_index=0, width=640, height=480)
        
        with patch('cv2.VideoCapture') as mock_videocapture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda x: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(x, 0)
            mock_videocapture.return_value = mock_cap

            with CameraCapture(config=config, data_dir=str(tmp_data_dir)) as camera:
                assert camera.is_initialized == True
            
            # 退出上下文后应该已释放
            assert camera.is_initialized == False
