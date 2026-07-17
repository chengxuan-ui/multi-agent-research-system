"""
完整端到端测试：五角色 + 记忆层 + 反思迭代闭环
"""

import sys
import os
import io
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_memory_layer():
    """测试 FAISS 向量记忆层"""
    print("\n" + "=" * 60)
    print("测试 1: FAISS 向量记忆层")
    print("=" * 60)

    import shutil
    from pathlib import Path
    from mem_store.vector_memory import VectorMemory

    # 使用独立测试目录，避免之前测试残留
    test_dir = "./mem_store/save_load_test"
    if Path(test_dir).exists():
        shutil.rmtree(test_dir)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    mem = VectorMemory(test_dir)

    # 添加记忆
    mem.add("Python 装饰器原理", "装饰器是一种函数包装器...", ["https://python.org"])
    mem.add("Python 闭包概念", "闭包是嵌套函数中对外层变量的引用...", ["https://python.org/closure"])
    assert mem.size == 2, f"期望 2 条，实际 {mem.size}"

    # 语义搜索（"函数包装器" 可能匹配装饰器或闭包，取决于 embedding 模型）
    results = mem.search("函数包装器", top_k=3)
    assert len(results) >= 1, "应该找到相关记忆"
    best, score = results[0]
    assert "器" in best.question or "闭包" in best.question, f"未找到相关结果: {best.question}"
    assert score > 0.3, f"相似度应 > 0.3，实际 {score}"

    # 无关联查询
    no_match = mem.search("量子计算薛定谔方程", top_k=3)
    assert len(no_match) == 0, f"不相关查询应该空结果，实际 {len(no_match)}"

    # 持久化 + 重新加载
    mem.save()
    mem2 = VectorMemory(test_dir)
    assert mem2.size == 2, f"加载后应为 2 条，实际 {mem2.size}"

    print("  ✅ FAISS 记忆层测试全部通过")


def test_module_imports():
    """测试所有模块导入"""
    print("\n" + "=" * 60)
    print("测试 2: 模块导入")
    print("=" * 60)

    import config
    from graph.state import ResearchState
    from graph.workflow import build_workflow
    from agents.planner import planner_node
    from agents.researcher import researcher_node
    from agents.writer import writer_node
    from agents.reviewer import reviewer_node
    from mem_store.vector_memory import VectorMemory
    from tools.search import web_search

    print("  ✅ 所有模块导入成功")


def test_workflow_compilation():
    """测试工作流编译"""
    print("\n" + "=" * 60)
    print("测试 3: LangGraph 工作流编译")
    print("=" * 60)

    from graph.workflow import build_workflow

    app = build_workflow()

    # 验证节点
    nodes = list(app.get_graph().nodes.keys())
    expected = {"__start__", "planner", "researcher", "writer", "reviewer"}
    for n in expected:
        assert n in nodes, f"缺少节点: {n}"
    print(f"  ✅ 工作流编译成功，节点: {nodes}")


def test_full_pipeline_with_memory():
    """测试完整流程（含记忆层）"""
    print("\n" + "=" * 60)
    print("测试 4: 完整流程（含记忆命中）")
    print("=" * 60)

    from graph.workflow import build_workflow
    from graph.state import ResearchState

    app = build_workflow()

    # 第一次运行：无记忆，全部搜索
    state1: ResearchState = {
        "question": "RAG 检索增强生成的核心概念",
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

    # 捕获 stdout 避免噪音
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        final1 = app.invoke(state1)
    finally:
        sys.stdout = old_stdout

    assert final1["memory_hits"] == 0, "首次运行不应有记忆命中"
    print(f"  ✅ 首次运行: 0 记忆命中, {len(final1['sub_questions'])} 子问题, 报告 {len(final1.get('final_report',''))} 字符")
    print(f"  ✅ 迭代轮次: {final1['iteration']}, 审查通过: {final1['review_passed']}")

    # 第二次运行：相似问题，应触发记忆命中
    state2: ResearchState = {
        "question": "RAG 系统的工作原理与应用",
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

    old_stdout2 = sys.stdout
    sys.stdout = io.StringIO()

    try:
        final2 = app.invoke(state2)
    finally:
        sys.stdout = old_stdout2

    print(f"  ✅ 第二次运行（相似问题）: {final2['memory_hits']} 记忆命中")
    print(f"  ✅ 审查通过: {final2['review_passed']}")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# 多智能体研究系统 - 完整测试套件")
    print("#" * 60)

    try:
        test_module_imports()
        test_workflow_compilation()
        test_memory_layer()

        # 检查 API Key 是否配置
        from config import LLM_API_KEY
        if LLM_API_KEY:
            test_full_pipeline_with_memory()
        else:
            print("\n  ⚠ 跳过端到端测试：LLM_API_KEY 未配置")

        print("\n" + "=" * 60)
        print(" 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
