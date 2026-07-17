"""
Researcher Agent - 研究员

职责:
    接收 Planner 分解的子问题列表，逐个搜索互联网获取信息，
    对每个子问题的搜索结果进行 LLM 摘要，返回结构化研究结果。

技术要点:
    - DuckDuckGo 搜索工具（免费，无需 API Key）
    - 搜索失败时自动降级为 LLM 知识兜底
    - LLM 对原始搜索结果做信息提取和摘要
    - 保留来源 URL，实现可溯源
    - ★ 向量记忆层：搜索前先查 FAISS 记忆库，语义相似的直接复用

在 LangGraph 中的角色:
    Planner → [Researcher] → Writer → ...
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm
from graph.state import ResearchState, ResearchResult
from tools.search import web_search, format_search_results

# 记忆库（模块级单例，跨多次调用保持状态）
_memory_instance = None

# 记忆命中阈值（0~1）
MEMORY_SIMILARITY_THRESHOLD = 0.55


def get_memory():
    """获取全局记忆实例（延迟初始化）"""
    global _memory_instance
    if _memory_instance is None:
        from mem_store.vector_memory import VectorMemory
        from config import MEMORY_INDEX_DIR
        _memory_instance = VectorMemory(MEMORY_INDEX_DIR)
    return _memory_instance


# ========== Prompt 模板 ==========

RESEARCHER_SYSTEM_PROMPT = """你是一个专业的网络研究员（Research Agent）。

你的任务是对给定的子问题进行搜索，然后对搜索结果做信息提取和摘要。

要求:
1. 摘要要客观、准确，只包含搜索结果中实际出现的信息
2. 保留信息来源，不要编造数据或观点
3. 摘要长度控制在 200-400 字

请用中文输出摘要。"""

SUMMARY_PROMPT_TEMPLATE = """请基于以下搜索结果，针对子问题做信息摘要。

子问题: {sub_question}

搜索结果:
---
{search_results}
---

请提取关键信息，写成一段结构化的摘要。只包含搜索结果中实际出现的信息，不要编造。"""

FALLBACK_PROMPT_TEMPLATE = """网络搜索暂时不可用。请基于你的知识，针对以下子问题提供一段专业摘要。

子问题: {sub_question}

请提供 200-400 字的结构化摘要，包含关键概念、主要方向和代表性方法。
注意：请标明这是基于模型知识的回答，可能需要进一步验证。"""

MEMORY_HIT_PROMPT = """以下是一段来自记忆库的历史研究摘要，与当前问题高度相关：

记忆中的问题: {cached_question}
历史摘要:
---
{cached_summary}
---

请基于以上历史摘要，结合当前问题"当前问题: {current_question}"，
重新整理出一段 200-400 字的针对性摘要。可以补充你的知识，但不要编造。"""


# ========== LangGraph 节点函数 ==========

def researcher_node(state: ResearchState) -> dict:
    """
    Researcher 节点函数

    输入: state["sub_questions"] (子问题列表)
    输出: {"research_results": [ResearchResult, ...], "memory_hits": int}

    工作流程（含记忆）:
        对每个子问题:
        1. 查 FAISS 记忆库，看有没有语义相似的历史结果
        2. 若命中（相似度 > 阈值）→ 直接用缓存摘要 + 来源
        3. 若未命中 → 搜索 → 摘要 → 写入记忆库
    """
    print("\n" + "=" * 60)
    print("[Researcher] 开始逐个搜索子问题...")
    print("=" * 60)

    sub_questions = state["sub_questions"]
    llm = get_llm(temperature=0.2, streaming=False)
    memory = get_memory()

    results: list[ResearchResult] = []
    memory_hits = 0
    memory_sources = []

    for i, sub_q in enumerate(sub_questions, 1):
        print(f"\n  [{i}/{len(sub_questions)}] 搜索: {sub_q}")

        # ====== Step 1: 查记忆库 ======
        memory_matches = memory.search(sub_q, top_k=3, min_score=MEMORY_SIMILARITY_THRESHOLD)

        if memory_matches:
            best_entry, best_score = memory_matches[0]
            print(f"    → 记忆命中! 相似度: {best_score:.2f}, 来源问题: {best_entry.question[:60]}")

            # 用 LLM 基于缓存摘要重新生成针对性回答
            memory_prompt = MEMORY_HIT_PROMPT.format(
                cached_question=best_entry.question,
                cached_summary=best_entry.summary,
                current_question=sub_q,
            )
            summary_response = llm.invoke([
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                HumanMessage(content=memory_prompt),
            ])
            summary = summary_response.content
            sources = best_entry.sources or ["[记忆库缓存]"]
            memory_hits += 1
            memory_sources.append(f"{sub_q[:40]} ← {best_entry.question[:40]}")

        else:
            # ====== Step 2: 搜索 ======
            print(f"    → 记忆未命中，执行网络搜索...")
            search_results = web_search(sub_q)
            sources = []

            if search_results:
                search_text = format_search_results(search_results)
                sources = [r["href"] for r in search_results if r.get("href")]
                print(f"    → 搜索完成，获取 {len(search_results)} 条结果")
                summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
                    sub_question=sub_q,
                    search_results=search_text[:3000],
                )
            else:
                print(f"    → 搜索失败，使用 LLM 知识兜底")
                summary_prompt = FALLBACK_PROMPT_TEMPLATE.format(sub_question=sub_q)

            summary_response = llm.invoke([
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                HumanMessage(content=summary_prompt),
            ])
            summary = summary_response.content

            # ====== Step 3: 写入记忆库 ======
            memory.add(sub_q, summary, sources)
            print(f"    → 结果已存入记忆库")

        print(f"    → 摘要完成: {summary[:100]}...")

        result: ResearchResult = {
            "sub_question": sub_q,
            "search_snippets": [],
            "summary": summary,
            "sources": sources,
        }
        results.append(result)

    # 持久化记忆库
    memory.save()
    print(f"\n  研究完成，共收集 {len(results)} 个子问题的结果")
    print(f"  记忆命中: {memory_hits}/{len(sub_questions)} ({memory_hits * 100 // max(len(sub_questions), 1)}%)")
    print(f"  记忆库总量: {memory.size} 条")

    return {
        "research_results": results,
        "memory_hits": memory_hits,
        "memory_sources": memory_sources,
    }
