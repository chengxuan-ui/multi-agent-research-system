"""
FAISS 向量记忆层

为多智能体研究系统提供长期记忆能力：
- 将每次研究的（子问题、摘要、来源）存入 FAISS 向量库
- 新问题到来时先查记忆，语义相似的直接复用，避免重复搜索
- 支持持久化到磁盘，重启后记忆不丢失

技术选型:
    - sentence-transformers (all-MiniLM-L6-v2): 本地 embedding，零 API 成本，~80MB
    - FAISS: Facebook 开源向量检索库，速度快，支持持久化
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

# NOTE: numpy/faiss/sentence_transformers 延迟导入,避免模块级导入导致 Windows 沙箱段错误


# ========== 懒加载 embedding 模型 ==========
_embedding_model = None


def _get_embedding_model():
    """懒加载 sentence-transformers 模型（首次调用才下载）"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _text_to_vector(text: str):
    """将文本转为 384 维向量"""
    import numpy as np
    model = _get_embedding_model()
    return model.encode(text, normalize_embeddings=True)


# ========== 记忆条目 ==========

class MemoryEntry:
    """单条记忆"""

    __slots__ = ("id", "question", "summary", "sources", "timestamp", "embedding")

    def __init__(self, question: str, summary: str, sources: list[str]):
        self.id = hashlib.md5(question.encode()).hexdigest()[:12]
        self.question = question
        self.summary = summary
        self.sources = sources or []
        self.timestamp = None
        self.embedding = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "summary": self.summary,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        entry = cls(d["question"], d["summary"], d.get("sources", []))
        entry.id = d["id"]
        entry.timestamp = d.get("timestamp", "")
        return entry


# ========== FAISS 向量记忆库 ==========

class VectorMemory:
    """
    FAISS 向量记忆库

    用法:
        mem = VectorMemory("./memory/index")
        mem.add("Python 装饰器原理", "装饰器是...", ["url1", "url2"])
        results = mem.search("Python 函数包装", top_k=3)
        mem.save()  # 持久化
    """

    def __init__(self, index_dir: str = "./memory/index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.json"

        self.entries: list[MemoryEntry] = []
        self.index = None
        self._dim = 384  # all-MiniLM-L6-v2 输出维度

        # 试着加载已有索引
        self._load()

    # ========== 核心操作 ==========

    def add(self, question: str, summary: str, sources: list[str]) -> MemoryEntry:
        """添加一条记忆，返回条目对象"""
        from datetime import datetime
        import faiss

        entry = MemoryEntry(question, summary, sources)
        entry.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 生成向量
        import numpy as np
        vec = _text_to_vector(question).astype(np.float32).reshape(1, -1)
        entry.embedding = vec

        # 加入 FAISS 索引
        if self.index is None:
            self.index = faiss.IndexFlatIP(self._dim)  # 内积相似度（归一化后=余弦相似度）

        self.index.add(vec)
        self.entries.append(entry)

        return entry

    def search(self, query: str, top_k: int = 3, min_score: float = 0.5) -> list[tuple[MemoryEntry, float]]:
        """
        语义搜索记忆库

        Args:
            query: 查询文本
            top_k: 返回前 K 条
            min_score: 最低相似度阈值（0~1），低于此值的不返回

        Returns:
            [(MemoryEntry, score), ...] 按相似度降序
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        import numpy as np
        query_vec = _text_to_vector(query).astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            results.append((self.entries[idx], float(score)))

        return results

    def get_by_question(self, question: str) -> Optional[MemoryEntry]:
        """精确匹配（MD5）"""
        qid = hashlib.md5(question.encode()).hexdigest()[:12]
        for entry in self.entries:
            if entry.id == qid:
                return entry
        return None

    # ========== 持久化 ==========

    def save(self):
        """保存索引和元数据到磁盘"""
        import faiss

        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

        metadata = [e.to_dict() for e in self.entries]
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从磁盘加载索引和元数据"""
        import faiss

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.entries = [MemoryEntry.from_dict(d) for d in metadata]

    # ========== 统计 ==========

    @property
    def size(self) -> int:
        return len(self.entries)

    def clear(self):
        """清空记忆"""
        self.entries = []
        self.index = None
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def stats(self) -> dict:
        """返回记忆库统计信息"""
        return {
            "total_entries": self.size,
            "index_path": str(self.index_path),
        }
