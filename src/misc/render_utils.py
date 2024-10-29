import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import os


class VideoCreator:
    """视频创建器类"""

    def __init__(self, output_path: str, fps: int = 30):
        """
        初始化视频创建器

        参数:
            output_path (str): 输出视频的保存路径
            fps (int): 视频帧率
        """
        self.output_path = output_path
        self.fps = fps
        self.video_writer = None
        self.initialized = False

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_writer(self, size: Tuple[int, int], codec: str = 'mp4v'):
        """初始化视频写入器"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                size
            )
            self.initialized = True
        except Exception as e:
            self.logger.error(f"初始化视频写入器失败: {str(e)}")
            raise

    def _process_image(self, pil_image: Image.Image, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """处理单张图片"""
        try:
            # 调整大小
            if size and pil_image.size != size:
                pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)

            # 确保图片是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # 转换为OpenCV格式
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            self.logger.error(f"处理图片失败: {str(e)}")
            raise

    def create_video(
            self,
            pil_images: List[Image.Image],
            size: Optional[Tuple[int, int]] = None,
            codec: str = 'mp4v',
            progress_bar: bool = True
    ) -> bool:
        """
        创建视频

        参数:
            pil_images (List[Image.Image]): Pillow图片列表
            size (Tuple[int, int], optional): 输出视频尺寸
            codec (str): 视频编码器
            progress_bar (bool): 是否显示进度条

        返回:
            bool: 是否成功创建视频
        """
        try:
            if not pil_images:
                raise ValueError("图片列表为空")

            # 确定视频尺寸
            if size is None:
                size = pil_images[0].size

            # 初始化视频写入器
            self._init_writer(size, codec)

            # 处理图片
            images_iterator = tqdm(pil_images) if progress_bar else pil_images
            for img in images_iterator:
                cv2_image = self._process_image(img, size)
                self.video_writer.write(cv2_image)

            return True

        except Exception as e:
            self.logger.error(f"创建视频失败: {str(e)}")
            return False

        finally:
            if self.video_writer is not None:
                self.video_writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.video_writer is not None:
            self.video_writer.release()