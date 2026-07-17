#!/usr/bin/env python
"""独立测试脚本 - FAISS 记忆层"""
import sys, os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Starting test...")
print(f"Python: {sys.executable}")
print(f"Path: {sys.path[:3]}")

from mem_store.vector_memory import VectorMemory
print("Import OK")

mem = VectorMemory("./mem_store/test_index")
print(f"Created, size: {mem.size}")

mem.add("test question", "test summary", ["http://source1.com"])
print(f"Added, size: {mem.size}")

results = mem.search("test", top_k=3)
print(f"Search: {len(results)} results")
if results:
    print(f"  Best: {results[0][0].question} ({results[0][1]:.3f})")

mem.save()
print("Saved. Test passed!")
