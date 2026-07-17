#!/usr/bin/env python
"""
多智能体研究系统 - 入口文件

用法:
    # 基本用法
    python main.py "RAG 系统的混合检索有哪些优化方向"

    # 指定输出文件
    python main.py "LangGraph 多智能体框架的优缺点" --output report.md

    # 交互模式
    python main.py --interactive

环境变量:
    需要在项目根目录创建 .env 文件，配置 LLM_API_KEY
    参考 .env.example
"""

import argparse
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LLM_API_KEY, LLM_MODEL, LLM_PROVIDER
from graph.workflow import build_workflow
from graph.state import ResearchState


def check_config():
    """检查配置是否完整"""
    if not LLM_API_KEY:
        print("=" * 60)
        print("错误: LLM_API_KEY 未设置！")
        print("=" * 60)
        print("\n请在项目根目录创建 .env 文件，配置 API Key。")
        print("参考 .env.example 文件。\n")
        print("OpenAI 申请: https://platform.openai.com/api-keys")
        print("DeepSeek 申请（更便宜）: https://platform.deepseek.com")
        print()
        sys.exit(1)

    print(f"  LLM Provider: {LLM_PROVIDER}")
    print(f"  LLM Model: {LLM_MODEL}")
    print()


def format_report(state: ResearchState) -> str:
    """将研究结果格式化为可读报告"""
    lines = []
    lines.append("# 研究报告")
    lines.append(f"\n## 原始问题\n\n{state['question']}\n")

    # 优先使用 Writer 生成的 draft_report / final_report
    report_body = state.get("final_report") or state.get("draft_report", "")
    if report_body:
        lines.append("## 正文\n")
        lines.append(report_body)
        lines.append("")
    else:
        lines.append("## 研究结果\n")
        for i, result in enumerate(state["research_results"], 1):
            lines.append(f"### 子问题 {i}: {result['sub_question']}\n")
            lines.append(f"**摘要:**\n{result['summary']}\n")
            if result.get("sources"):
                lines.append("**来源:**")
                for j, src in enumerate(result["sources"], 1):
                    lines.append(f"  {j}. {src}")
                lines.append("")
            lines.append("---\n")

        if not state["research_results"]:
            lines.append("（无研究结果）\n")

    lines.append(f"\n## 元信息\n")
    lines.append(f"- 迭代轮次: {state.get('iteration', 0)}")
    lines.append(f"- 子问题数: {len(state.get('sub_questions', []))}")
    lines.append(f"- 审查结果: {'通过' if state.get('review_passed', False) else '未通过/未审查'}")
    lines.append(f"- 记忆命中: {state.get('memory_hits', 0)}/{len(state.get('sub_questions', []))}")
    if state.get("memory_sources"):
        lines.append(f"- 记忆来源: {', '.join(state['memory_sources'])}")

    return "\n".join(lines)


def run_single(question: str, output_file: str | None = None):
    """运行单次研究"""
    check_config()

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

    print("\n" + "#" * 60)
    print(f"# 多智能体研究系统")
    print(f"# 研究问题: {question}")
    print("#" * 60 + "\n")

    # 使用 invoke 一次性运行，避免重复执行
    final_state = app.invoke(initial_state)

    # 生成报告
    report = format_report(final_state)

    # 输出到文件
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存到: {output_file}")

    # 打印报告
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    return final_state


def run_interactive():
    """交互模式"""
    check_config()

    print("多智能体研究系统 - 交互模式")
    print("输入研究问题开始，输入 'quit' 退出\n")

    while True:
        question = input("研究问题> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        try:
            run_single(question)
        except KeyboardInterrupt:
            print("\n已中断\n")
        except Exception as e:
            print(f"\n错误: {e}\n")

    print("再见！")


def main():
    parser = argparse.ArgumentParser(
        description="多智能体研究系统 - 基于 LangGraph 的多智能体协同研究工具"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="研究问题（直接运行模式）",
    )
    parser.add_argument(
        "-o", "--output",
        help="将报告保存到指定文件",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="交互模式",
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.question:
        run_single(args.question, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
