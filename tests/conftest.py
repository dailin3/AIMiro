"""
pytest 配置与全局 fixture
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到 Python 路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    data_dir = Path(__file__).parent.parent / "test_data"
    if not data_dir.exists():
        pytest.skip("测试数据目录不存在")
    return data_dir


@pytest.fixture(scope="session")
def human_images(test_data_dir):
    """获取所有人脸测试图像"""
    human_dir = test_data_dir / "human"
    images = []
    
    for person_dir in human_dir.iterdir():
        if person_dir.is_dir():
            for img_file in person_dir.glob("*.png"):
                images.append(img_file)
    
    if not images:
        pytest.skip("没有可用的人脸测试图像")
    
    return images


@pytest.fixture(scope="function")
def tmp_path_custom(tmp_path):
    """创建临时测试目录"""
    test_dir = tmp_path / "miro_test"
    test_dir.mkdir()
    return test_dir


@pytest.fixture(scope="module")
def app():
    """创建 Flask 测试应用"""
    from backend.backend_server import BackendServer, ServerConfig
    
    # 使用临时目录作为数据目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = ServerConfig(
            host="127.0.0.1",
            port=5000,
            debug=False,
            data_dir=tmp_dir
        )
        server = BackendServer(config)
        yield server.app


@pytest.fixture(scope="module")
def client(app):
    """创建 Flask 测试客户端"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="module")
def face_service():
    """创建人脸识别服务 (测试后清理)"""
    try:
        from backend.face_recognition_insightface import FaceRecognizerServiceInsightFace, FaceRecognitionConfig
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FaceRecognitionConfig(
                data_dir=tmp_dir
            )
            service = FaceRecognizerServiceInsightFace(config)
            yield service
            # 测试后清理
            service.clear_all_faces()
    except ImportError:
        pytest.skip("insightface 未安装")


@pytest.fixture(scope="module")
def camera_config():
    """默认摄像头配置"""
    from backend.camera_capture import CameraConfig
    return CameraConfig(
        camera_index=0,
        width=640,
        height=480
    )
