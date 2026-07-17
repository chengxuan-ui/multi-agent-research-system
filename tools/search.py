"""
搜索工具模块

封装网络搜索功能，提供统一的搜索接口供 Researcher Agent 调用。
默认使用 DuckDuckGo (ddgs 库)，搜索失败时自动降级为 LLM 知识兜底。
"""

from __future__ import annotations

from ddgs import DDGS

from config import SEARCH_MAX_RESULTS


def web_search(query: str, max_results: int = None) -> list[dict]:
    """
    使用 DuckDuckGo 执行网络搜索

    Args:
        query: 搜索关键词
        max_results: 最大结果数

    Returns:
        搜索结果列表，每项包含 title, body, href
    """
    if max_results is None:
        max_results = SEARCH_MAX_RESULTS

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })
    except Exception as e:
        print(f"    [搜索警告] DuckDuckGo 搜索失败: {e}")
        print(f"    [搜索降级] 将使用 LLM 知识库替代搜索")

    return results


def format_search_results(results: list[dict]) -> str:
    """
    将搜索结果列表格式化为文本

    Args:
        results: web_search 返回的结果列表

    Returns:
        格式化后的文本，供 LLM 摘要使用
    """
    if not results:
        return "（搜索未返回结果）"

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        body = r.get("body", "无内容")
        href = r.get("href", "")
        parts.append(f"[{i}] {title}\n    {body}\n    来源: {href}")

    return "\n\n".join(parts)


def search(query: str) -> str:
    """
    直接执行搜索并返回格式化文本（兼容旧接口）

    Args:
        query: 搜索关键词

    Returns:
        搜索结果文本
    """
    results = web_search(query)
    return format_search_results(results)


if __name__ == "__main__":
    # 快速测试搜索功能
    print("测试搜索...")
    result = search("Python LangChain RAG 教程")
    print(f"\n搜索结果:\n{result[:500]}...")
