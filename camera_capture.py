#!/usr/bin/env python3
"""
摄像头采集与储存模块

负责从摄像头采集图像/视频流，并储存到指定目录。
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraConfig:
    """摄像头配置类"""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        codec: str = "MJPG"
    ):
        """
        初始化摄像头配置

        Args:
            camera_index: 摄像头设备索引，0 表示默认摄像头
            width: 分辨率宽度
            height: 分辨率高度
            fps: 帧率
            codec: 编码格式
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec


class CameraCapture:
    """摄像头采集类"""

    def __init__(self, config: Optional[CameraConfig] = None, data_dir: str = "./data"):
        """
        初始化摄像头采集器

        Args:
            config: 摄像头配置，使用默认配置如果为 None
            data_dir: 图像保存目录
        """
        self.config = config or CameraConfig()
        self.data_dir = Path(data_dir)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False

        # 自动创建数据目录
        self._create_data_dir()

    def _create_data_dir(self) -> None:
        """创建数据目录"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"数据目录已创建：{self.data_dir.absolute()}")

    def initialize(self) -> bool:
        """
        初始化摄像头

        Returns:
            初始化是否成功
        """
        try:
            # 打开摄像头
            self.cap = cv2.VideoCapture(self.config.camera_index)

            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 (索引：{self.config.camera_index})")
                return False

            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # 设置编码格式
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # 验证配置是否应用成功（实际值可能与设置值不同）
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"摄像头初始化成功")
            logger.info(f"实际分辨率：{actual_width}x{actual_height}")
            logger.info(f"实际帧率：{actual_fps} FPS")

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"初始化摄像头时出错：{e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获单帧图像

        Returns:
            图像帧（BGR 格式的 numpy 数组），失败返回 None
        """
        if not self.is_initialized or self.cap is None:
            logger.error("摄像头未初始化")
            return None

        try:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.warning("无法读取图像帧")
                return None

            return frame

        except Exception as e:
            logger.error(f"捕获图像帧时出错：{e}")
            return None

    def capture_frames(self, count: int, interval: float = 0.0) -> List[np.ndarray]:
        """
        捕获多帧图像

        Args:
            count: 要捕获的帧数
            interval: 帧间间隔（秒）

        Returns:
            图像帧列表
        """
        frames = []

        for _ in range(count):
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame)

            if interval > 0:
                time.sleep(interval)

        logger.info(f"成功捕获 {len(frames)}/{count} 帧图像")
        return frames

    def save_image(
        self,
        image: np.ndarray,
        filename: Optional[str] = None,
        format: str = "jpg"
    ) -> Optional[Path]:
        """
        保存图像到文件

        Args:
            image: 要保存的图像
            filename: 文件名，如果为 None 则使用时间戳生成
            format: 图像格式（jpg, png 等）

        Returns:
            保存的文件路径，失败返回 None
        """
        if image is None:
            logger.error("图像为空，无法保存")
            return None

        try:
            now = datetime.now()

            # 按日期创建子文件夹：data/pic/2026-03-26/
            date_folder = self.data_dir / "pic" / now.strftime("%Y-%m-%d")
            date_folder.mkdir(parents=True, exist_ok=True)

            # 生成文件名：时分秒.png
            if filename is None:
                filename = now.strftime("%H%M%S") + f".{format}"

            # 确保扩展名正确
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"

            # 完整路径
            file_path = date_folder / filename

            # 保存图像
            success = cv2.imwrite(str(file_path), image)

            if success:
                logger.info(f"图像已保存：{file_path}")
                return file_path
            else:
                logger.error(f"保存图像失败：{file_path}")
                return None

        except Exception as e:
            logger.error(f"保存图像时出错：{e}")
            return None

    def capture_and_save(
        self,
        filename: Optional[str] = None,
        format: str = "jpg"
    ) -> Optional[Path]:
        """
        捕获当前帧并保存

        Args:
            filename: 文件名
            format: 图像格式

        Returns:
            保存的文件路径，失败返回 None
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        return self.save_image(frame, filename, format)

    def get_video_stream(self) -> Optional[cv2.VideoCapture]:
        """
        获取视频流对象

        Returns:
            VideoCapture 对象，如果未初始化返回 None
        """
        if not self.is_initialized or self.cap is None:
            logger.error("摄像头未初始化")
            return None
        return self.cap

    def check_disk_space(self, min_space_mb: int = 100) -> bool:
        """
        检查磁盘空间

        Args:
            min_space_mb: 最小可用空间（MB）

        Returns:
            空间是否充足
        """
        import shutil
        try:
            stat = shutil.disk_usage(self.data_dir)
            free_mb = stat.free / (1024 * 1024)

            if free_mb < min_space_mb:
                logger.warning(f"磁盘空间不足：{free_mb:.2f} MB < {min_space_mb} MB")
                return False

            logger.info(f"磁盘可用空间：{free_mb:.2f} MB")
            return True

        except Exception as e:
            logger.error(f"检查磁盘空间时出错：{e}")
            return False

    def cleanup_old_files(
        self,
        max_files: int = 100,
        max_age_days: int = 7
    ) -> int:
        """
        清理旧文件

        Args:
            max_files: 最大文件数量
            max_age_days: 文件最大保存天数

        Returns:
            删除的文件数量
        """
        deleted_count = 0
        now = datetime.now()

        try:
            # 获取所有文件并按时间排序
            files = sorted(
                self.data_dir.glob("*"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )

            # 删除超过数量限制的文件
            if len(files) > max_files:
                for file_path in files[max_files:]:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"已删除旧文件：{file_path}")

            # 删除超过时间的文件
            for file_path in files[:max_files]:
                file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.days > max_age_days:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"已删除过期文件：{file_path}")

            logger.info(f"共删除 {deleted_count} 个文件")

        except Exception as e:
            logger.error(f"清理旧文件时出错：{e}")

        return deleted_count

    def release(self) -> None:
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.is_initialized = False
            logger.info("摄像头资源已释放")

    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """上下文管理器出口"""
        self.release()


def main():
    """主函数 - 演示摄像头采集功能"""
    import argparse

    parser = argparse.ArgumentParser(description="摄像头采集工具")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头设备索引 (默认：0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="分辨率宽度 (默认：1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="分辨率高度 (默认：1080)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="输出目录 (默认：./data)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="捕获图像数量 (默认：1)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="图像格式 (默认：jpg)"
    )

    args = parser.parse_args()

    # 创建配置
    config = CameraConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height
    )

    # 创建采集器
    capture = CameraCapture(config=config, data_dir=args.output)

    try:
        # 初始化摄像头
        if not capture.initialize():
            logger.error("摄像头初始化失败")
            return 1

        # 检查磁盘空间
        if not capture.check_disk_space():
            logger.warning("磁盘空间可能不足")

        saved_files = []
        for i in range(args.count):
            logger.info(f"正在捕获第 {i + 1}/{args.count} 张图像...")
            file_path = capture.capture_and_save(format=args.format)
            if file_path:
                saved_files.append(file_path)

            if args.count > 1 and i < args.count - 1:
                time.sleep(0.5)

        logger.info(f"成功保存 {len(saved_files)}/{args.count} 张图像")
        for f in saved_files:
            logger.info(f"  - {f}")

        return 0

    except KeyboardInterrupt:
        logger.info("用户中断")
        return 0
    finally:
        capture.release()


if __name__ == "__main__":
    exit(main())
