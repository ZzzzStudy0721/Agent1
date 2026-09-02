# 面试官模拟 RAG Agent

面向求职者的 AI 面试陪练工具：上传简历 / 项目文档 / 目标岗位 JD，AI 模拟面试官按 JD 提问、追问、点评回答，且所有提问与点评均有原文依据、绝不编造。

纯 Python 单机轻量实现，无 Docker / 数据库依赖。

## 技术栈

`LangChain` · `LangGraph` · `ChromaDB` · `FastAPI` · `Streamlit` · `DeepSeek API` · `BM25 + RRF` · `CrossEncoder 重排` · 本地 `bge-small-zh-v1.5` embedding

## 核心特性

- **混合检索**：BM25 关键词召回 + 向量语义召回，RRF（k=60）融合排序
- **重排精排**：本地 CrossEncoder（bge-reranker-base）对候选精排
- **引用溯源 + 无出处拒答**：检索层向量相似度门限（0.2）硬约束，搜不到相关内容直接拒答，LLM 无编造空间
- **面试状态机**：LangGraph 建模面试流程（开场→出题→追问→点评→复盘），状态流转可控可解释
- **追问链**：回答过短 / 缺量化数据 / 回避问题三条触发规则，每题最多 2 轮追问
- **JD 定制出题**：读取目标岗位 JD，生成 ≥10 道贴合业务场景的定制面试题
- **证据核实**：点评时检索知识库比对回答中的硬事实，与简历/论文矛盾处直接指出
- **Tool Calling**：复盘报告导出封装为 LangChain Tool（JSON Schema 定义），由 LLM 自主决定调用
- **离线评估**：自建 23 题测试集（含口语化问法），Recall@5 / MRR 指标

## 架构

```
┌──────────────────────────────────────────────┐
│                Streamlit UI                 │
│      💬 知识库问答         🎤 模拟面试         │
└──────────────┬──────────────────┬────────────┘
               │ HTTP             │ 直接调用
        ┌──────▼───────┐   ┌──────▼──────────────┐
        │  FastAPI     │   │  LangGraph 面试状态机 │
        │  /chat       │   │  追问判定 → 点评     │
        │  /upload     │   │  → 出题 → 复盘       │
        └──────┬───────┘   └──────┬──────────────┘
               │                  │
        ┌──────▼──────────────────▼─────────────┐
        │           检索管线                     │
        │  BM25 ──┐                             │
        │  Vector ─┴─ RRF 融合 → CrossEncoder 重排│
        └──────┬────────────────────────────────┘
               │
    ┌──────────▼─────────┐    ┌──────────────────┐
    │  ChromaDB 向量库    │    │  DeepSeek LLM    │
    │  (bge-small-zh)    │    │  (生成/点评/核实)  │
    └────────────────────┘    └──────────────────┘
```

面试状态机（LangGraph）：

```
START → 追问判定 ──需要追问(≤2轮)──→ 生成追问 → 等回答 → (回到判定)
              │
              └─通过→ 三维度点评 + 证据核实 ──还有下一题──→ 出题
                                  │
                                  └─无下一题→ 复盘总结 → 调用 export_report 工具 → END
```

## 消融实验（自建 23 题测试集）

| 检索方案 | Recall@5 | MRR |
|---|---|---|
| 纯向量检索（基线） | 0.725 | 0.619 |
| 纯 BM25 关键词检索 | 0.855 | 0.741 |
| 混合检索（BM25+向量，RRF） | 0.855 | 0.656 |
| **混合检索 + CrossEncoder 重排** | **0.986** | **0.920** |

混合检索将 Recall@5 从 0.725 提升至 0.855（+17.9%），叠加重排后达 0.986（较基线 +36.0%），MRR 从 0.619 提升至 0.920（+48.6%）。

> 测试集含书面问法与真实口语化问法两类；指标为关键词命中口径的自建评测，仅供项目内消融对比。

## 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-deepseek langchain-huggingface langchain-chroma \
    langchain-text-splitters langgraph chromadb sentence-transformers \
    rank-bm25 jieba fastapi uvicorn streamlit python-dotenv
```

### 2. 配置

复制 `.env` 并填入 DeepSeek API Key（模型走 hf-mirror 国内镜像，默认已配置）：

```
DEEPSEEK_API_KEY=sk-xxx
HF_ENDPOINT=https://hf-mirror.com
```

### 3. 准备知识库

将简历 / 项目文档 / 目标 JD 的 `.md` 或 `.txt` 文件放入 `data/` 目录（该目录已 gitignore，不会上传 GitHub）。

### 4. 运行方式

**A. 命令行问答**（混合检索 + 拒答）：

```bash
python app.py
```

**B. 命令行模拟面试**（JD 出题 + 追问 + 点评 + 证据核实 + 复盘导出）：

```bash
python agent.py
```

**C. 网页版**（推荐演示用，两个终端）：

```bash
uvicorn api:fastapi_app --host 127.0.0.1 --port 8000
streamlit run ui.py
```

### 5. 测试

```bash
python tests/test_api.py          # DeepSeek API 冒烟
python tests/test_mvp.py          # RAG 端到端
python tests/eval_retrieval.py    # 四模式消融评估
python tests/test_guardrail.py    # 拒答硬约束（含口语问法）
python tests/test_agent.py        # 面试全流程
python tests/test_followup.py     # 追问链
python tests/test_evidence.py     # 证据核实
python tests/test_tool.py         # Tool Calling
python tests/test_fastapi.py      # FastAPI 接口
```

## 项目结构

```
├── app.py            # RAG 问答：加载→切分→向量化→混合检索→重排→拒答→生成
├── retrieval.py      # BM25 索引、RRF 融合、CrossEncoder 重排
├── agent.py          # LangGraph 面试状态机 + JD 出题 + 证据核实 + Tool
├── api.py            # FastAPI 服务层（/chat、/upload）
├── ui.py             # Streamlit 界面（问答 + 模拟面试）
├── data/             # 知识库（gitignore）
├── tests/            # 9 个测试脚本 + 23 题测试集
└── chroma_db/        # 向量库持久化（gitignore）
```

## 设计决策

- **为什么模型层用 DeepSeek**：RAG 架构与模型解耦，模型层可插拔；DeepSeek 是成本/可用性决策，切换其他模型只需改配置。
- **为什么面试用状态机而非自由对话**：面试有固定流程，状态机保证完整走完且追问可解释；自由对话会跑偏。
- **拒答门限为什么用向量相似度而非重排分数**：英文预训练的重排模型对中文口语问法打分不稳定（实测相关问法低至 0.019 与无关问法 0.028 无法区分）；中文 embedding 的相似度信号实测相关 0.283~0.656 vs 无关 0.011~0.127，可稳定分离。
