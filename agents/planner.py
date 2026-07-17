"""
Planner Agent - 研究规划师

职责:
    接收用户的研究问题，将其分解为 3-5 个可独立搜索的子问题。
    如果存在上一轮 Reviewer 的反馈，会根据反馈调整规划方向。

技术要点:
    - 使用 LangChain structured output (with_structured_output)
    - Pydantic 模型约束输出格式，确保子问题列表可解析
    - 支持反思迭代：反馈注入 prompt，实现闭环优化

在 LangGraph 中的角色:
    START → [Planner] → Researcher → ...
    Reviewer fail → [Planner] (with feedback) → Researcher → ...
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from config import get_llm
from graph.state import ResearchState


# ========== 结构化输出模型 ==========

class ResearchPlan(BaseModel):
    """Planner 的结构化输出：研究计划"""

    sub_questions: list[str] = Field(
        description="3-5 个子问题，每个问题应可独立通过网络搜索研究，"
        "合起来能完整回答原始问题"
    )
    reasoning: str = Field(
        description="为什么这样分解，每个子问题的研究重点是什么"
    )


# ========== Prompt 模板 ==========

PLANNER_SYSTEM_PROMPT = """你是一个专业的研究规划师（Research Planner）。

你的任务是将用户的研究问题分解为 3-5 个子问题，每个子问题可以独立通过网络搜索来研究。

分解原则:
1. 子问题之间应该互补，合起来能完整覆盖原始问题
2. 每个子问题应该具体、可搜索（避免太宽泛或太窄）
3. 避免子问题之间有过多重叠
4. 子问题的措辞应该适合作为搜索引擎的查询关键词

请用中文输出。"""

PLANNER_FEEDBACK_SECTION = """
以下是上一轮 Reviewer 的反馈，请在重新规划时充分考虑:
{feedback}
"""


# ========== LangGraph 节点函数 ==========

def planner_node(state: ResearchState) -> dict:
    """
    Planner 节点函数

    输入: state["question"] (用户问题), state["review_feedback"] (可选反馈)
    输出: {"sub_questions": [...], "iteration": N}

    工作流程:
        1. 构造 system prompt（如果有反馈，注入反馈内容）
        2. 调用 LLM with structured output 获取结构化计划
        3. 返回子问题列表和迭代计数
    """
    print("\n" + "=" * 60)
    print("[Planner] 正在分析研究问题并分解子任务...")
    print("=" * 60)

    question = state["question"]
    feedback = state.get("review_feedback", "")
    iteration = state.get("iteration", 0) + 1

    # 构造 system prompt
    system_content = PLANNER_SYSTEM_PROMPT
    if feedback:
        system_content += "\n" + PLANNER_FEEDBACK_SECTION.format(feedback=feedback)
        print(f"  → 检测到 Reviewer 反馈，第 {iteration} 轮迭代规划")

    # 调用 LLM with structured output (DeepSeek 不支持 json_schema，用 function_calling)
    llm = get_llm(temperature=0.3, streaming=False)
    structured_llm = llm.with_structured_output(ResearchPlan, method="function_calling")

    plan: ResearchPlan = structured_llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=f"请分解以下研究问题:\n\n{question}"),
    ])

    # 打印规划结果
    print(f"\n  原始问题: {question}")
    print(f"  分解为 {len(plan.sub_questions)} 个子问题:")
    for i, sq in enumerate(plan.sub_questions, 1):
        print(f"    {i}. {sq}")
    print(f"\n  规划理由: {plan.reasoning[:200]}...")
    print(f"  当前迭代轮次: {iteration}")

    return {
        "sub_questions": plan.sub_questions,
        "iteration": iteration,
    }
