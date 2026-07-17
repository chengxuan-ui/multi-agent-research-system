"""
Reviewer Agent - 质量审查员

职责:
    用 LLM-as-Judge 评估 Writer 生成的报告质量，
    给出 pass/fail 判断和具体改进建议。

技术要点:
    - structured output (function_calling) 确保返回可解析的判断结果
    - 三维度评估：完整性 / 准确性 / 结构性
    - 详细反馈：fail 时给出具体修改建议，供下一轮 Planner 和 Writer 参考

在 LangGraph 中的角色:
    Writer → [Reviewer]
                ├─ pass → 输出 final_report → END
                └─ fail → Planner (with feedback) → Researcher → Writer → [Reviewer]
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from config import get_llm, MAX_ITERATIONS
from graph.state import ResearchState


# ========== 结构化输出模型 ==========

class ReviewResult(BaseModel):
    """Reviewer 的结构化输出：审查结果"""

    passed: bool = Field(
        description="报告是否通过审查。True 表示质量合格，False 表示需要修改。"
    )
    completeness_score: int = Field(
        description="完整性评分 (1-10)。报告是否完整回答了原始问题，是否有遗漏的重要方面。"
    )
    accuracy_score: int = Field(
        description="准确性评分 (1-10)。报告内容是否准确，是否有事实错误或逻辑漏洞。"
    )
    structure_score: int = Field(
        description="结构性评分 (1-10)。报告结构是否清晰、逻辑是否连贯、格式是否规范。"
    )
    overall_score: int = Field(
        description="综合评分 (1-10)。三项评分的加权综合。"
    )
    feedback: str = Field(
        description="审查反馈。如果未通过，必须包含具体的修改建议，"
        "明确指出问题所在和改进方向。如果通过，简要表扬亮点。"
    )


# ========== Prompt 模板 ==========

REVIEWER_SYSTEM_PROMPT = """你是一个严格的技术报告审查员（Quality Reviewer）。

你的任务是评估一份研究报告的质量，从以下三个维度打分:

1. **完整性 (Completeness)**: 报告是否完整回答了原始问题？是否有遗漏的重要子问题或关键信息？
2. **准确性 (Accuracy)**: 报告内容是否准确？是否有事实错误、逻辑矛盾或编造的内容？
3. **结构性 (Structure)**: 报告结构是否清晰？逻辑是否连贯？格式是否规范？

审查标准:
- 综合评分 >= 7 分且各项均 >= 5 分 → 通过 (passed=True)
- 综合评分 < 7 或任一项 < 5 → 不通过 (passed=False)
- 不通过时，反馈必须具体、可操作，明确指出问题在哪里、怎么改
- 通过时，反馈可以简要肯定

请用中文输出反馈。"""

REVIEWER_USER_PROMPT = """请评估以下研究报告。

## 原始问题
{question}

## 待审查的报告
{draft_report}

## 研究数据（供参考）
共 {result_count} 个子问题的研究结果。

请给出你的评估。"""


# ========== LangGraph 节点函数 ==========

def reviewer_node(state: ResearchState) -> dict:
    """
    Reviewer 节点函数

    输入: state["question"], state["draft_report"], state["research_results"], state["iteration"]
    输出: {"review_passed": bool, "review_feedback": str, "final_report": str (if passed)}

    工作流程:
        1. 构造包含报告和原始问题的评估 prompt
        2. 调用 LLM with structured output 获取结构化的审查结果
        3. 如果通过 → 设置 final_report
        4. 如果未通过但已达最大迭代 → 强制通过（防无限循环）
    """
    print("\n" + "=" * 60)
    print("[Reviewer] 正在审查报告质量...")
    print("=" * 60)

    question = state["question"]
    draft_report = state["draft_report"]
    iteration = state.get("iteration", 0)

    # 构造 prompt
    user_content = REVIEWER_USER_PROMPT.format(
        question=question,
        draft_report=draft_report,
        result_count=len(state.get("research_results", [])),
    )

    # 调用 LLM with structured output
    llm = get_llm(temperature=0.1, streaming=False)
    structured_llm = llm.with_structured_output(ReviewResult, method="function_calling")

    review: ReviewResult = structured_llm.invoke([
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    # 打印审查结果
    status = "通过" if review.passed else "不通过"
    print(f"\n  审查结果: {status}")
    print(f"  完整性: {review.completeness_score}/10")
    print(f"  准确性: {review.accuracy_score}/10")
    print(f"  结构性: {review.structure_score}/10")
    print(f"  综合评分: {review.overall_score}/10")
    print(f"  当前迭代: 第 {iteration} 轮")

    # 强制通过逻辑：超过最大迭代次数时无论评分如何都通过
    if not review.passed and iteration >= MAX_ITERATIONS:
        print(f"\n  ⚠ 已达最大迭代次数 ({MAX_ITERATIONS})，强制通过")
        review.passed = True
        review.feedback += f"\n（已达最大迭代次数 {MAX_ITERATIONS}，系统强制通过）"

    if review.passed:
        print(f"  → 审查通过！")
        if review.feedback:
            print(f"  → Reviewer 评语: {review.feedback[:200]}...")
        return {
            "review_passed": True,
            "review_feedback": review.feedback,
            "review_score": review.overall_score,
            "final_report": draft_report,
        }
    else:
        print(f"  → 审查未通过，准备开始第 {iteration + 1} 轮迭代")
        print(f"  → Reviewer 反馈: {review.feedback[:200]}...")
        return {
            "review_passed": False,
            "review_feedback": review.feedback,
            "review_score": review.overall_score,
        }
