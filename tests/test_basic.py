"""
测试脚本 - 验证项目各模块可正常导入和基本功能

运行: python tests/test_basic.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """测试所有模块可正常导入"""
    print("测试模块导入...")

    try:
        from config import LLM_MODEL, LLM_PROVIDER
        print(f"  [OK] config.py - provider={LLM_PROVIDER}, model={LLM_MODEL}")
    except Exception as e:
        print(f"  [FAIL] config.py: {e}")
        return False

    try:
        from graph.state import ResearchState, ResearchResult
        print(f"  [OK] graph/state.py")
    except Exception as e:
        print(f"  [FAIL] graph/state.py: {e}")
        return False

    try:
        from agents.planner import planner_node, ResearchPlan
        print(f"  [OK] agents/planner.py")
    except Exception as e:
        print(f"  [FAIL] agents/planner.py: {e}")
        return False

    try:
        from agents.researcher import researcher_node
        print(f"  [OK] agents/researcher.py")
    except Exception as e:
        print(f"  [FAIL] agents/researcher.py: {e}")
        return False

    try:
        from graph.workflow import build_workflow
        print(f"  [OK] graph/workflow.py")
    except Exception as e:
        print(f"  [FAIL] graph/workflow.py: {e}")
        return False

    return True


def test_search_tool():
    """测试搜索工具（不需要 API Key）"""
    print("\n测试搜索工具...")

    try:
        from tools.search import search
        result = search("Python 编程语言")
        print(f"  [OK] 搜索成功，结果长度: {len(result)} 字符")
        print(f"  预览: {result[:100]}...")
        return True
    except Exception as e:
        print(f"  [FAIL] 搜索失败: {e}")
        return False


def test_workflow_build():
    """测试工作流可以编译（不需要 API Key）"""
    print("\n测试工作流编译...")

    try:
        from graph.workflow import build_workflow
        app = build_workflow()
        print(f"  [OK] 工作流编译成功")
        print(f"  节点列表: {list(app.get_graph().nodes.keys())}")
        return True
    except Exception as e:
        print(f"  [FAIL] 工作流编译失败: {e}")
        return False


def main():
    print("=" * 50)
    print("多智能体研究系统 - 基础测试")
    print("=" * 50 + "\n")

    results = []

    results.append(("模块导入", test_imports()))
    results.append(("搜索工具", test_search_tool()))
    results.append(("工作流编译", test_workflow_build()))

    print("\n" + "=" * 50)
    print("测试结果汇总:")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  所有测试通过!")
    else:
        print("\n  部分测试失败，请检查。")

    print("=" * 50)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
