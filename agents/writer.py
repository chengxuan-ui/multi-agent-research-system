"""
Writer Agent - 报告撰稿人

职责:
    将 Researcher 的所有研究结果综合为一篇结构化研究报告。
    如果存在上一轮 Reviewer 的反馈，会在特定章节融入改进。

技术要点:
    - 结构化输出：引言 → 分章分析 → 综合结论
    - 反馈融入：根据 Reviewer 的批评重点调整报告结构

在 LangGraph 中的角色:
    Researcher → [Writer] → Reviewer → ...
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm
from graph.state import ResearchState


# ========== Prompt 模板 ==========

WRITER_SYSTEM_PROMPT = """你是一个专业的技术报告撰稿人（Research Writer）。

你的任务是将多个子问题的研究结果综合为一篇高质量的结构化研究报告。

写作要求:
1. **结构清晰**: 包含引言、分章节分析、综合结论三部分
2. **信息准确**: 只使用研究结果中实际包含的信息，不编造
3. **逻辑连贯**: 各章节之间要有逻辑关联，形成完整的论述
4. **语言专业**: 使用专业但易懂的语言，适合技术从业者阅读
5. **标注来源**: 关键数据或观点后标注来源编号

输出格式:
- 使用 Markdown 格式
- 标题层级: ## 一级标题, ### 二级标题
- 章节间用空行分隔"""

WRITER_USER_PROMPT = """请将以下研究结果综合为一篇结构化报告。

## 原始问题
{question}

## 研究结果
{research_results}

## 要求
1. 写一个引言段落，概述问题背景和研究范围
2. 按逻辑重新组织子问题的研究发现，形成连贯的分析章节
3. 写一个结论段落，总结核心发现和关键洞察
4. 如果能看出不同方向之间的关联，请在报告中体现这些联系"""

WRITER_FEEDBACK_PROMPT = """
## 上一轮的修改意见
上一轮报告被 Reviewer 驳回，以下是具体反馈:

{feedback}

请在重新撰写时重点关注以上反馈，确保本次报告能够解决 Reviewer 指出的问题。"""


# ========== LangGraph 节点函数 ==========

def writer_node(state: ResearchState) -> dict:
    """
    Writer 节点函数

    输入: state["question"], state["research_results"], state["review_feedback"]（可选）
    输出: {"draft_report": "..."}

    工作流程:
        1. 格式化所有研究结果为文本
        2. 构造 prompt（如有反馈则注入）
        3. 调用 LLM 生成结构化报告
    """
    print("\n" + "=" * 60)
    print("[Writer] 正在综合研究结果，撰写报告...")
    print("=" * 60)

    question = state["question"]
    research_results = state["research_results"]
    feedback = state.get("review_feedback", "")

    # 格式化研究结果
    results_text_parts = []
    for i, r in enumerate(research_results, 1):
        sub_q = r["sub_question"]
        summary = r["summary"]
        sources = r.get("sources", [])
        src_text = ", ".join(f"[{j+1}] {s}" for j, s in enumerate(sources)) if sources else "无来源"
        results_text_parts.append(
            f"### 子问题 {i}: {sub_q}\n\n**研究摘要:**\n{summary}\n\n**来源:** {src_text}"
        )
    results_text = "\n\n---\n\n".join(results_text_parts)

    # 构造 prompt
    user_content = WRITER_USER_PROMPT.format(
        question=question,
        research_results=results_text,
    )
    if feedback:
        user_content += "\n\n" + WRITER_FEEDBACK_PROMPT.format(feedback=feedback)
        print("  → 检测到 Reviewer 反馈，已融入修改建议")

    # 调用 LLM 生成报告
    llm = get_llm(temperature=0.4, streaming=False)
    response = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    draft_report = response.content
    print(f"  → 报告生成完成，共 {len(draft_report)} 字符")

    return {"draft_report": draft_report}
