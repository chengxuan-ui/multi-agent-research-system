# 多智能体研究系统 (Multi-Agent Research System)

基于 **LangGraph** 的多智能体协同研究系统。五个 AI Agent 协同工作：Planner 分解问题 → Researcher 搜索 + 记忆复用 → Writer 撰写报告 → Reviewer 质量审查，不通过自动反思迭代。

## 系统架构

```
                    用户研究问题
                         │
                         ▼
              ┌──────────────────┐
              │     Planner      │  分解为 3-5 个子问题
              │   (研究规划师)    │  + 如为迭代轮次，注入 Reviewer 反馈
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    Researcher    │  逐个搜索子问题
              │    (研究员)      │  ⇄ FAISS 记忆库（语义检索命中则复用）
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │     Writer       │  综合研究结果
              │   (报告撰写师)    │  生成结构化 Markdown 报告
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    Reviewer      │  LLM-as-Judge 三维度评分
              │   (质量审查员)    │  完整性 / 准确性 / 结构性
              └────┬───────┬─────┘
                   │       │
              pass │       │ fail（最多 3 轮）
                   │       │
                   ▼       ▼
                  END    Planner（带上反馈重新规划）
```

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| Agent 编排 | **LangGraph** | StateGraph 状态机，条件边实现反思循环 |
| LLM | **LangChain + OpenAI 兼容** | DeepSeek / OpenAI / Moonshot 任意切换 |
| 搜索 | **DuckDuckGo** | 免费，无需 API Key；失败自动 LLM 降级 |
| 向量记忆 | **FAISS + sentence-transformers** | 本地 embedding，零 API 成本，~80MB 模型 |
| 结构化输出 | **Pydantic** | Planner/Reviewer 输出格式约束 |
| Web Demo | **Streamlit** | 可视化各 Agent 工作过程和最终报告 |

## 快速开始

### 1. 安装依赖

```bash
# 激活虚拟环境（已创建）
.venv\Scripts\activate

# 安装依赖（首次）
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件，填入你的 API Key：

```env
# 推荐 DeepSeek（国内快、便宜，注册送免费额度）
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-你的密钥
LLM_BASE_URL=https://api.deepseek.com/v1

# 或 OpenAI
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini
# LLM_API_KEY=sk-你的密钥
```

### 3. 运行

```bash
# CLI 模式
python main.py "RAG 系统中文档分块策略有哪些优化方向"

# 保存报告
python main.py "LangGraph 多智能体框架的优缺点" -o report.md

# 交互模式
python main.py -i

# Streamlit Web Demo（可视化！）
streamlit run app.py

# 便捷脚本（自动过滤 DNS 噪音）
python run.py "你的研究问题"
```

### 4. 测试

```bash
python tests/test_basic.py     # 基础测试
python tests/test_full.py      # 完整测试（含记忆层 + 反思迭代）
```

## 项目结构

```
multi-agent-research-system/
├── main.py                  # CLI 入口
├── app.py                   # Streamlit 可视化 Demo
├── run.py                   # 便捷运行脚本
├── config.py                # 配置管理
├── requirements.txt         # 依赖
├── .env / .env.example      # API Key 配置
│
├── agents/                  # 五个 AI Agent
│   ├── planner.py           # 研究规划师：分解 + 反馈注入
│   ├── researcher.py        # 研究员：搜索 + FAISS 记忆复用
│   ├── writer.py            # 报告撰写师：结构化 Markdown
│   ├── reviewer.py          # 质量审查员：三维度 LLM-as-Judge
│   └── __init__.py
│
├── graph/                   # LangGraph 工作流
│   ├── state.py             # 全局状态定义
│   ├── workflow.py          # StateGraph 编排（含条件边）
│   └── __init__.py
│
├── tools/                   # Agent 工具
│   ├── search.py            # DuckDuckGo 搜索 + 降级
│   └── __init__.py
│
├── mem_store/               # FAISS 向量记忆层
│   ├── vector_memory.py     # 语义检索 + 持久化
│   ├── index/               # FAISS 索引存储目录
│   └── __init__.py
│
└── tests/
    ├── test_basic.py        # 基础测试
    └── test_full.py         # 完整端到端测试
```

## 核心特性

### 1. 反思迭代闭环

Reviewer 不通过 → 反馈注入 Planner → 重新分解 → 重新搜索 → 重新撰写 → 再次审查。

实测效果：LangChain vs LlamaIndex 对比报告，第 1 轮评分 6/10 不通过 → 第 2 轮 8/10 通过，报告从 3424 字扩充至 7320 字。

### 2. FAISS 向量记忆

每次研究的（子问题, 摘要, 来源）存入 FAISS 向量库。新问题先语义检索记忆库，相似度 > 0.55 直接复用，跳过搜索。支持持久化，重启不丢失。

### 3. 搜索降级

DuckDuckGo 搜索失败（DNS/网络问题）时自动降级为 LLM 知识兜底，保证系统在任何网络环境下都能产出结果。

### 4. 多 Provider 兼容

同一个接口支持 DeepSeek / OpenAI / 智谱 / Moonshot 等所有 OpenAI 兼容 API。

## 开发历程

| 阶段 | 目标 | 状态 |
|------|------|------|
| Week 1 | Planner + Researcher 最小闭环 | ✅ 完成 |
| Week 2 | Writer + Reviewer 反思迭代闭环 | ✅ 完成 |
| Week 3 | FAISS 记忆层 + Streamlit Demo | ✅ 完成 |

## License

MIT
