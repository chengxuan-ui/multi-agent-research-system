#!/usr/bin/env python
"""
多智能体研究系统 - Streamlit 可视化 Demo

提供 Web 界面用于交互式研究体验：
- 输入研究问题
- 实时展示各 Agent 的输出（Planner → Researcher → Writer → Reviewer）
- 反思迭代过程可追溯
- 记忆库统计展示
- 报告 Markdown 渲染

启动方式:
    streamlit run app.py
    或
    python -m streamlit run app.py

依赖: streamlit, sentence-transformers (已包含在 requirements.txt 中)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import LLM_API_KEY, LLM_MODEL, LLM_PROVIDER, MEMORY_INDEX_DIR
from graph.state import ResearchState


# ========== 页面配置 ==========

st.set_page_config(
    page_title="多智能体研究系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 自定义 CSS ==========

st.markdown("""
<style>
    .agent-card {
        border-left: 4px solid #4CAF50;
        padding: 12px 16px;
        margin: 8px 0;
        background: #f5f5f5;
        border-radius: 4px;
    }
    .agent-card.planner { border-left-color: #2196F3; }
    .agent-card.researcher { border-left-color: #FF9800; }
    .agent-card.writer { border-left-color: #9C27B0; }
    .agent-card.reviewer { border-left-color: #F44336; }
    .agent-card.memory { border-left-color: #00BCD4; }
    .result-box {
        padding: 10px 14px;
        margin: 4px 0;
        background: #fafafa;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .score-pass { color: #4CAF50; font-weight: bold; }
    .score-fail { color: #F44336; font-weight: bold; }
    .stMarkdown h3 { margin-top: 24px; }
    .status-bar {
        display: flex;
        gap: 16px;
        align-items: center;
    }
    .status-item {
        background: #e3f2fd;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# ========== 侧边栏：系统信息 ==========

with st.sidebar:
    st.title(" 多智能体研究系统")
    st.markdown("基于 **LangGraph** 的多智能体协同研究工具")
    st.divider()

    st.subheader("系统配置")
    st.markdown(f"- LLM: **{LLM_PROVIDER}** / {LLM_MODEL}")
    st.markdown(f"- 搜索: DuckDuckGo（免费）")
    st.markdown(f"- 记忆: FAISS 向量库")

    # 记忆库统计
    st.divider()
    st.subheader(" 记忆库")
    try:
        from mem_store.vector_memory import VectorMemory
        mem = VectorMemory(MEMORY_INDEX_DIR)
        st.metric("已存储条目", mem.size)
    except Exception:
        st.metric("已存储条目", "--")

    st.divider()
    st.subheader("架构说明")
    st.markdown("""
    **五角色协同**:
    1.  **Planner** — 问题分解
    2.  **Researcher** — 搜索 + 记忆
    3.  **Writer** — 报告撰写
    4.  **Reviewer** — 质量审查
    5.  **反思迭代** — 不通过自动重来
    """)

    st.divider()
    st.caption("技术栈: LangGraph + LangChain + FAISS + Streamlit")


# ========== 主页面 ==========

st.title("多智能体协同研究系统")
st.markdown("输入一个研究问题，多个 AI Agent 将协同完成研究并生成报告")

# 输入区
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "研究问题",
        placeholder="例如: RAG 系统中文档分块策略有哪些优化方向？",
        label_visibility="collapsed",
    )
with col2:
    run_btn = st.button("开始研究", type="primary", use_container_width=True)


# ========== 研究执行 ==========

if run_btn and question.strip():
    # 检查配置
    if not LLM_API_KEY:
        st.error(" 请在项目根目录的 `.env` 文件中配置 `LLM_API_KEY`")
        st.stop()

    # 清空之前的输出
    st.divider()

    # ===== 状态栏 =====
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    planner_status = status_col1.empty()
    researcher_status = status_col2.empty()
    writer_status = status_col3.empty()
    reviewer_status = status_col4.empty()

    # ===== 执行区 =====
    planner_placeholder = st.empty()
    researcher_placeholder = st.empty()
    memory_placeholder = st.empty()
    writer_placeholder = st.empty()
    reviewer_placeholder = st.empty()
    report_placeholder = st.empty()

    try:
        from graph.workflow import build_workflow
        from agents.planner import planner_node
        from agents.researcher import researcher_node, get_memory
        from agents.writer import writer_node
        from agents.reviewer import reviewer_node

        # ===== 初始状态 =====
        state: ResearchState = {
            "question": question,
            "sub_questions": [],
            "research_results": [],
            "draft_report": "",
            "review_feedback": "",
            "review_passed": False,
            "iteration": 0,
            "final_report": "",
            "memory_hits": 0,
            "memory_sources": [],
        }

        # ===== Phase 1: Planner =====
        with st.spinner("  Planner 正在分解问题..."):
            planner_status.markdown(" Planner **规划中**")
            update = planner_node(state)
            state.update(update)

        planner_status.markdown(" Planner ✅ **完成**")
        with planner_placeholder.container():
            st.subheader(" 任务分解 (Planner)")
            for i, sq in enumerate(state["sub_questions"], 1):
                st.markdown(f"- **子问题 {i}:** {sq}")

        # ===== Phase 2: Researcher =====
        researcher_status.markdown(" Researcher **搜索中**")
        with researcher_placeholder.container():
            st.subheader(" 研究执行 (Researcher)")
            results_container = st.empty()

            # 捕获 Researcher 输出
            import io
            old_stdout = sys.stdout
            captured = io.StringIO()
            sys.stdout = captured

            try:
                update = researcher_node(state)
                state.update(update)
            finally:
                sys.stdout = old_stdout

            # 展示结果
            with results_container.container():
                for j, r in enumerate(state["research_results"], 1):
                    with st.expander(f"子问题 {j}: {r['sub_question']}", expanded=(j == 1)):
                        st.markdown(r["summary"])
                        if r.get("sources"):
                            st.caption("来源: " + ", ".join(r["sources"][:3]))

        researcher_status.markdown(f" Researcher ✅ **完成** ({len(state['research_results'])} 项)")

        # 记忆命中展示
        if state.get("memory_hits", 0) > 0:
            with memory_placeholder.container():
                st.subheader(" 记忆命中")
                hit_rate = state["memory_hits"] * 100 // max(len(state["sub_questions"]), 1)
                st.success(f"从记忆库击中 **{state['memory_hits']}** 条 (命中率 {hit_rate}%)")
                if state.get("memory_sources"):
                    for ms in state["memory_sources"]:
                        st.caption(f"  {ms}")

        # ===== Phase 3: Writer =====
        with st.spinner("  Writer 正在撰写报告..."):
            writer_status.markdown(" Writer **撰写中**")
            update = writer_node(state)
            state.update(update)

        writer_status.markdown(" Writer ✅ **完成**")
        with writer_placeholder.container():
            st.subheader(" 报告生成 (Writer)")
            report_preview = state.get("draft_report", "")[:500]
            st.caption(f"报告长度: {len(state.get('draft_report', ''))} 字符")
            st.markdown(report_preview + ("..." if len(state.get("draft_report", "")) > 500 else ""))

        # ===== Phase 4: Reviewer =====
        with st.spinner("  Reviewer 正在评估报告质量..."):
            reviewer_status.markdown(" Reviewer **审查中**")
            update = reviewer_node(state)
            state.update(update)

        if state["review_passed"]:
            reviewer_status.markdown(" Reviewer ✅ **通过**")
            with reviewer_placeholder.container():
                st.success(f" 报告质量审查**通过** (评分: {state.get('review_score', '-')}分)")
                if state.get("review_feedback"):
                    with st.expander("审查评价"):
                        st.markdown(state["review_feedback"])
        else:
            reviewer_status.markdown(" Reviewer ❌ **不通过**")
            feedback = state.get("review_feedback", "")
            with reviewer_placeholder.container():
                st.warning(f" 报告质量审查**不通过**，第 {state.get('iteration', 1)} 轮反馈")

            # 如果开启了反思迭代，继续展示迭代过程
            if state.get("iteration", 1) < 3:
                st.info(" 触发反思迭代，Planner 将根据反馈重新规划...")

                # 第二轮：Planner
                with st.spinner("🔄 反思迭代中..."):
                    update2 = planner_node(state)
                    state.update(update2)

                st.markdown("**第二轮 — 重新规划的子问题:**")
                for si2, sq2 in enumerate(state["sub_questions"], 1):
                    st.markdown(f"- **{si2}.** {sq2}")

                # 第二轮：Researcher
                update3 = researcher_node(state)
                state.update(update3)

                # 第二轮：Writer
                update4 = writer_node(state)
                state.update(update4)

                # 第二轮：Reviewer
                update5 = reviewer_node(state)
                state.update(update5)

                if state["review_passed"]:
                    reviewer_status.markdown(" Reviewer ✅ **通过** (第2轮)")
                    with reviewer_placeholder.container():
                        st.success(" 反思迭代完成，报告质量**通过**")
                else:
                    reviewer_status.markdown(" Reviewer ❌ 最终未通过")

        # ===== 最终报告 =====
        st.divider()
        st.header(" 最终研究报告")

        report_body = state.get("final_report") or state.get("draft_report", "")
        st.markdown(report_body)

        # 元信息
        with st.expander(" 研究元信息"):
            st.markdown(f"- **原始问题:** {state['question']}")
            st.markdown(f"- **迭代轮次:** {state.get('iteration', 0)}")
            st.markdown(f"- **子问题数:** {len(state.get('sub_questions', []))}")
            st.markdown(f"- **研究结果数:** {len(state.get('research_results', []))}")
            st.markdown(f"- **审查结果:** {'通过' if state.get('review_passed') else '未通过'}")
            st.markdown(f"- **记忆命中:** {state.get('memory_hits', 0)} 条")

        # 记忆库统计
        memory = get_memory()
        st.sidebar.metric("已存储条目", memory.size)

    except Exception as e:
        st.error(f" 研究过程出错: {e}")
        import traceback
        with st.expander("错误详情"):
            st.code(traceback.format_exc())

elif not question.strip() and run_btn:
    st.warning("请先输入研究问题")


# ========== 底部 ==========

st.divider()
st.caption("多智能体研究系统 | LangGraph + LangChain + FAISS + Streamlit | 2026")
