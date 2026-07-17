#!/usr/bin/env python
"""
便捷运行脚本 - 自动屏蔽 DuckDuckGo DNS 噪音

用法:
    python run.py "RAG系统的混合检索有哪些优化方向"
    python run.py  # 使用默认问题
"""

import sys
import os

# 在导入任何搜索相关模块前，屏蔽 DNS 噪音
_real_stderr = sys.stderr
class FilteredStderr:
    def __init__(self, real):
        self.real = real
    def write(self, msg):
        if "detected forged question section" in msg:
            return  # 丢弃 DNS 噪音
        self.real.write(msg)
    def flush(self):
        self.real.flush()

sys.stderr = FilteredStderr(_real_stderr)

# 现在安全导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import run_single

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "RAG系统的混合检索有哪些优化方向"
    run_single(question, None)
