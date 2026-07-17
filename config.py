"""
项目配置模块
统一管理 LLM、搜索、工作流的配置参数
"""

import os
from dotenv import load_dotenv

# 加载 .env 环境变量
# 使用绝对路径确保在任何工作目录下都能找到
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

# ========== LLM 配置 ==========
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "") or None
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ========== 搜索配置 ==========
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

# ========== 工作流配置 ==========
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

# ========== 路径配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MEMORY_INDEX_DIR = os.path.join(PROJECT_ROOT, "mem_store", "index")


def get_llm(temperature: float | None = None, streaming: bool = True):
    """
    获取 LangChain LLM 实例
    支持 OpenAI 及所有 OpenAI 兼容接口（DeepSeek、Moonshot 等）

    Args:
        temperature: 生成温度，默认使用全局配置
        streaming: 是否流式输出

    Returns:
        ChatOpenAI 实例
    """
    from langchain_openai import ChatOpenAI

    if not LLM_API_KEY:
        raise ValueError(
            "LLM_API_KEY 未设置！请在 .env 文件中配置。\n"
            "参考 .env.example 文件。"
        )

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature or LLM_TEMPERATURE,
        streaming=streaming,
    )
