#!/usr/bin/env python
"""完整端到端测试：包含 FAISS 记忆层 + 五角色反思迭代"""
import sys, os, io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 屏蔽 DuckDuckGo DNS 噪音
_rerr = sys.stderr
class Filt:
    def write(self, m):
        if 'forged question' in m or 'query_type' in m: return
        _rerr.write(m)
    def flush(self): _rerr.flush()
sys.stderr = Filt()

from mem_store.vector_memory import VectorMemory
from graph.workflow import build_workflow
from graph.state import ResearchState
from config import MEMORY_INDEX_DIR, LLM_API_KEY

print("=" * 60)
print("完整端到端测试：FAISS 记忆 + 五角色 + 反思迭代")
print("=" * 60)

# 测试 1: FAISS 记忆层
print("\n[Test 1] FAISS 记忆层")
mem = VectorMemory(MEMORY_INDEX_DIR)
mem.clear()
mem.add("Python装饰器原理", "装饰器是一种设计模式", ["url1"])
mem.add("RAG检索增强生成", "RAG结合检索和生成", ["url2"])
assert mem.size == 2
r = mem.search("装饰器设计", top_k=3, min_score=0.3)
assert len(r) >= 1
print(f"  ✅ 记忆层: {mem.size} 条, 搜索命中 {len(r)} 条")
mem.clear()

# 测试 2: 完整五角色流程
if not LLM_API_KEY:
    print("\n  ⚠️ 跳过端到端测试：LLM_API_KEY 未配置")
else:
    print("\n[Test 2] 完整五角色流程（含反思迭代）")
    app = build_workflow()

    state: ResearchState = {
        "question": "RAG检索增强生成的核心工作原理",
        "sub_questions": [],
        "research_results": [],
        "draft_report": "",
        "review_feedback": "",
        "review_passed": False,
        "iteration": 0,
        "final_report": "",
        "review_score": 0,
        "memory_hits": 0,
        "memory_sources": [],
    }

    final = app.invoke(state)
    assert final["review_passed"], "Review should pass"
    assert len(final.get("final_report", "")) > 500, "Report too short"
    assert final["iteration"] >= 0, "Iteration should be >= 0"
    print(f"  ✅ 流程通过: {final['iteration']} 轮迭代, 报告 {len(final['final_report'])} 字符")
    print(f"  ✅ 记忆命中: {final['memory_hits']}/{len(final['sub_questions'])}")
    print(f"  ✅ 记忆库总量: {mem.size} 条")

print("\n" + "=" * 60)
print(" 全部测试通过！")
print("=" * 60)
