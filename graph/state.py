"""
LangGraph 状态定义

这是整个多智能体系统的核心数据结构。
所有 Agent 共享这个 State，通过读写不同的字段来协同工作。

状态流转:
    用户问题
        → Planner: 分解为子问题列表
        → Researcher: 逐个搜索子问题，收集结果
        → Writer: 综合研究结果，撰写报告
        → Reviewer: 评估报告质量，pass/fail
            pass → 输出最终报告
            fail → 回到 Planner 重新规划（带上 Reviewer 反馈）
"""

from __future__ import annotations

from typing import TypedDict


class ResearchResult(TypedDict):
    """单个子问题的研究结果"""

    sub_question: str  # 子问题原文
    search_snippets: list[str]  # 搜索引擎返回的摘要列表
    summary: str  # LLM 对搜索结果的综合摘要
    sources: list[str]  # 来源 URL 列表


class ResearchState(TypedDict):
    """
    多智能体研究系统的全局状态

    字段说明:
        question:          用户输入的原始研究问题
        sub_questions:     Planner 分解出的子问题列表
        research_results:  Researcher 对每个子问题的研究结果
        draft_report:      Writer 生成的报告草稿
        review_feedback:   Reviewer 的评估反馈
        review_passed:     Reviewer 是否通过
        iteration:         当前迭代轮次（反思循环计数）
        final_report:      最终输出报告
    """

    question: str
    sub_questions: list[str]
    research_results: list[ResearchResult]
    draft_report: str
    review_feedback: str
    review_passed: bool
    iteration: int
    final_report: str
    review_score: int            # Reviewer 综合评分 (1-10)
    memory_hits: int            # 记忆命中次数
    memory_sources: list[str]   # 记忆命中的来源问题列表
