#!/usr/bin/env python3
"""
AI 图像分析模块

使用 Gemini 原生 API 进行图像分析。
"""

import os
import base64
import json
import time
import requests
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass


def load_env_file(env_path: Union[str, Path] = ".env") -> None:
    """从 .env 文件加载环境变量"""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ[key] = value


def load_prompt_file(prompt_path: Union[str, Path] = "prompt.md") -> str:
    """
    从 prompt.md 加载提示词

    如果文件不存在，返回默认提示词
    """
    prompt_file = Path(prompt_path)
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    # 默认提示词
    return "请详细描述这张图片中的内容，包括主要对象、场景、颜色、氛围等。"


# 启动时自动加载 .env
load_env_file()


@dataclass
class AnalysisConfig:
    """AI 分析配置类"""
    model: str = "gemini-3-flash-preview"  # Gemini 多模态模型（preview 版本配额更多）
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7


class AIImageAnalyzer:
    """AI 图像分析器 - 使用 Gemini 原生 API"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        初始化分析器

        Args:
            config: 分析配置，如果为 None 则从环境变量读取
        """
        self.config = config or AnalysisConfig()
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> str:
        """获取 API Key"""
        if self._api_key is None:
            self._api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
            if not self._api_key:
                raise ValueError(
                    "GEMINI_API_KEY 未设置。请在环境变量中设置 GEMINI_API_KEY "
                    "或在初始化时传入 api_key"
                )
        return self._api_key

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        将图像编码为 base64

        Args:
            image_path: 图像文件路径

        Returns:
            base64 编码的图像字符串
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_mime_type(self, image_path: Union[str, Path]) -> str:
        """
        获取图像的 MIME 类型

        Args:
            image_path: 图像文件路径

        Returns:
            MIME 类型字符串
        """
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_types.get(ext, "image/jpeg")

    def _build_request_body(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
        image_paths: Optional[List[Union[str, Path]]] = None
    ) -> Dict[str, Any]:
        """
        构建 Gemini API 请求体

        Args:
            prompt: 提示词
            image_path: 单张图片路径
            image_paths: 多张图片路径

        Returns:
            请求体字典
        """
        parts = [{"text": prompt}]

        # 处理单张图片
        if image_path:
            base64_image = self._encode_image(image_path)
            mime_type = self._get_image_mime_type(image_path)
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_image
                }
            })

        # 处理多张图片
        if image_paths:
            for img_path in image_paths:
                base64_image = self._encode_image(img_path)
                mime_type = self._get_image_mime_type(img_path)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_image
                    }
                })

        return {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
        }

    def analyze(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        分析单张图像

        Args:
            image_path: 图像文件路径
            prompt: 分析提示词（默认为 None，会从 prompt.md 读取）
            system_prompt: 系统提示词（可选，Gemini 通过不同方式支持）

        Returns:
            包含分析结果的字典
        """
        # 如果未提供 prompt，从 prompt.md 读取
        if prompt is None:
            prompt = load_prompt_file()

        api_key = self._get_api_key()

        # 如果有系统提示词，合并到用户提示词中
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        # 构建请求
        request_body = self._build_request_body(prompt, image_path=image_path)

        # 调用 API（带重试）
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent?key={api_key}"

        last_error = None
        for attempt in range(retries):
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=request_body,
                    timeout=60
                )

                if response.status_code == 429:
                    wait_time = (attempt + 1) * 2
                    print(f"遇到 rate limit，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                # 提取生成的内容
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        text_parts = [
                            part["text"] for part in candidate["content"]["parts"]
                            if "text" in part
                        ]
                        analysis_text = "".join(text_parts)

                        # 获取 token 使用情况
                        usage = data.get("usageMetadata", {})

                        return {
                            "success": True,
                            "image_path": str(image_path),
                            "analysis": analysis_text,
                            "model": self.config.model,
                            "usage": {
                                "prompt_tokens": usage.get("promptTokenCount", 0),
                                "completion_tokens": usage.get("candidatesTokenCount", 0),
                                "total_tokens": usage.get("totalTokenCount", 0)
                            }
                        }

                return {
                    "success": False,
                    "image_path": str(image_path),
                    "error": "API 返回了空结果",
                    "raw_response": data
                }

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"请求失败，{wait_time} 秒后重试...")
                    time.sleep(wait_time)
                continue
            except Exception as e:
                return {
                    "success": False,
                    "image_path": str(image_path),
                    "error": str(e)
                }

        return {
            "success": False,
            "image_path": str(image_path),
            "error": f"请求失败（已重试 {retries} 次）: {str(last_error)}"
        }

    def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量分析多张图像

        Args:
            image_paths: 图像文件路径列表
            prompt: 分析提示词（默认为 None，会从 prompt.md 读取）
            system_prompt: 系统提示词（可选）

        Returns:
            分析结果列表
        """
        # 如果未提供 prompt，从 prompt.md 读取
        if prompt is None:
            prompt = load_prompt_file()

        results = []
        for i, image_path in enumerate(image_paths):
            print(f"正在分析第 {i+1}/{len(image_paths)} 张图片: {image_path}")
            result = self.analyze(image_path, prompt, system_prompt)
            results.append(result)
        return results

    def compare_images(
        self,
        image_paths: List[Union[str, Path]],
        comparison_prompt: str = "请比较这些图片，分析它们的相似点和不同点。"
    ) -> Dict[str, Any]:
        """
        比较多张图像

        Args:
            image_paths: 图像文件路径列表（2-4张）
            comparison_prompt: 比较提示词

        Returns:
            比较结果
        """
        if len(image_paths) < 2:
            raise ValueError("比较图像数量至少需要 2 张")

        api_key = self._get_api_key()

        # 构建请求
        request_body = self._build_request_body(
            comparison_prompt,
            image_paths=image_paths
        )

        # 调用 API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent?key={api_key}"

        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=request_body,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # 提取生成的内容
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    text_parts = [
                        part["text"] for part in candidate["content"]["parts"]
                        if "text" in part
                    ]
                    comparison_text = "".join(text_parts)

                    usage = data.get("usageMetadata", {})

                    return {
                        "success": True,
                        "image_paths": [str(p) for p in image_paths],
                        "comparison": comparison_text,
                        "model": self.config.model,
                        "usage": {
                            "prompt_tokens": usage.get("promptTokenCount", 0),
                            "completion_tokens": usage.get("candidatesTokenCount", 0),
                            "total_tokens": usage.get("totalTokenCount", 0)
                        }
                    }

            return {
                "success": False,
                "image_paths": [str(p) for p in image_paths],
                "error": "API 返回了空结果",
                "raw_response": data
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "image_paths": [str(p) for p in image_paths],
                "error": f"请求失败: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "image_paths": [str(p) for p in image_paths],
                "error": str(e)
            }

    def extract_features(
        self,
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        提取图像特征（以结构化格式返回）

        Args:
            image_path: 图像文件路径

        Returns:
            特征提取结果
        """
        prompt = """请分析这张图片并以 JSON 格式返回以下信息：
{
    "objects": ["对象1", "对象2", ...],  // 图片中的主要对象
    "scene": "场景描述",  // 场景类型
    "colors": ["颜色1", "颜色2", ...],  // 主要颜色
    "mood": "氛围描述",  // 整体氛围
    "text_content": "图片中的文字内容（如有）",  // OCR
    "tags": ["标签1", "标签2", ...]  // 关键词标签
}
请确保返回的是有效的 JSON 格式。"""

        result = self.analyze(image_path, prompt)

        # 尝试从分析结果中解析 JSON
        if result.get("success"):
            import re

            content = result["analysis"]
            # 尝试提取 JSON 部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    features = json.loads(json_match.group())
                    result["features"] = features
                except json.JSONDecodeError:
                    result["features"] = None
            else:
                result["features"] = None

        return result


def main():
    """命令行入口点"""
    import argparse

    parser = argparse.ArgumentParser(description="AI 图像分析工具")
    parser.add_argument("image", help="图像文件路径")
    parser.add_argument("-p", "--prompt", default=None, help="自定义分析提示词")
    parser.add_argument("--api-key", default=None, help="Gemini API Key")
    parser.add_argument("--extract-features", action="store_true", help="提取结构化特征")

    args = parser.parse_args()

    # 初始化配置
    config = AnalysisConfig()
    if args.api_key:
        config.api_key = args.api_key

    # 创建分析器
    analyzer = AIImageAnalyzer(config)

    # 执行分析
    if args.extract_features:
        result = analyzer.extract_features(args.image)
    else:
        prompt = args.prompt or load_prompt_file()
        result = analyzer.analyze(args.image, prompt)

    # 输出结果
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
