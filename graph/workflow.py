"""
LangGraph 工作流定义

使用 LangGraph StateGraph 编排多智能体协同流程。

架构:
    START → Planner → Researcher → Writer → Reviewer
                                                ├─ pass → END
                                                └─ fail → Planner (反思迭代)

反思迭代闭环:
    当 Reviewer 判定报告不通过时，将反馈注入 Planner，
    Planner 根据反馈重新分解子问题 → Researcher 重新搜索 → Writer 重新撰写 → Reviewer 再次审查
    最多迭代 MAX_ITERATIONS 次，超限后强制通过
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from graph.state import ResearchState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.reviewer import reviewer_node
from config import MAX_ITERATIONS


def should_continue(state: ResearchState) -> str:
    """
    条件边判断函数：Reviewer 完成后应该走哪条路

    Args:
        state: 当前全局状态

    Returns:
        "end": 审查通过或超限，结束工作流
        "planner": 审查未通过，回到 Planner 重新规划
    """
    if state.get("review_passed", False):
        return "end"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "end"
    return "planner"


def build_workflow():
    """
    构建并编译 LangGraph 工作流（完整版，含 Writer + Reviewer 反思迭代）
    """
    workflow = StateGraph(ResearchState)

    # ========== 添加节点 ==========
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # ========== 添加边 ==========
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")

    # ========== 条件边（反思迭代）==========
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "end": END,       # 通过 → 结束
            "planner": "planner",  # 不通过 → 回到 Planner 重新规划
        }
    )

    # ========== 编译 ==========
    app = workflow.compile()
    return app


# ========== 便捷函数 ==========

def run_research(question: str) -> ResearchState:
    """一键运行完整研究流程"""
    app = build_workflow()

    initial_state: ResearchState = {
        "question": question,
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

    return app.invoke(initial_state)
