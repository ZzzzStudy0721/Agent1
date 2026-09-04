# 面试官模拟 RAG Agent

面向求职者的 AI 面试陪练工具：上传简历 / 项目文档 / 目标岗位 JD，AI 模拟面试官自由对话、即兴追问、复盘点评，且所有提问与点评均有原文依据、绝不编造。

纯 Python 单机轻量实现，无 Docker / 数据库依赖。

## 技术栈

`LangChain` · `LangGraph` · `ChromaDB` · `FastAPI` · `Streamlit` · 多后端可切换（DeepSeek 实测默认 + OpenAI 兼容端点接 GLM 等国内模型）· `BM25 + RRF` · `CrossEncoder 重排` · 本地 `bge-small-zh-v1.5` embedding

## 核心特性

- **混合检索**：BM25 关键词召回 + 向量语义召回，RRF（k=60）融合排序
- **重排精排**：本地 CrossEncoder（bge-reranker-base）对候选精排
- **引用溯源 + 无出处拒答**：检索层向量相似度门限（0.2）硬约束，搜不到相关内容直接拒答，LLM 无编造空间
- **自由对话式面试官**：无固定题单，面试官即兴提问 + 深入追问，最接近真人面试
- **LLM 决策路由**：每轮由 LLM 结构化决策（深挖追问 / 切换话题 / 结束面试），LangGraph 状态机保证流程可控
- **话题锚点**：从目标岗位 JD + 候选人简历/项目文档抽取 8 个话题锚点，面试围绕锚点自由发挥、不跑题
- **程序化护栏**：同话题最多追问 3 轮（超限强制切题，代码硬校验）+ 话题覆盖率 + 20 轮上限强制收尾；LLM 决策非法（指向已覆盖话题等）时自动兜底
- **证据核实**：复盘时检索知识库比对回答中的硬事实（只核含数字/指标的声明，宁可漏报不可错报），与简历/论文矛盾处直接指出
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
        │  /chat       │   │  决策路由 → 提问/追问 │
        │  /upload     │   │  → 复盘核实 → 报告    │
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

面试状态机（LangGraph）——每轮对话 invoke 一次图，循环由 Python 驱动：

```
START → 护栏检查（轮数≥20 或话题全覆盖？）──是──→ 复盘总结 + 证据核实
        │                                         → 调用 export_report → END
        └─否→ LLM 决策节点（结构化输出）
               ├─ 深挖追问 ──→ 面试官生成追问 → END（下一轮回到 START）
               ├─ 切换话题 ──→ 面试官挑未覆盖话题提问 → END
               └─ 结束 ─────→ 复盘总结 + 证据核实 → export_report → END
```

## 消融实验（自建 23 题测试集）

| 检索方案 | Recall@5 | MRR |
|---|---|---|
| 纯向量检索（基线） | 0.725 | 0.619 |
| 纯 BM25 关键词检索 | 0.855 | 0.751 |
| 混合检索（BM25+向量，RRF） | 0.812 | 0.656 |
| **混合检索 + CrossEncoder 重排** | **0.964** | **0.928** |

混合检索将 Recall@5 从 0.725 提升至 0.812（+12.0%），叠加重排后达 0.964（较基线 +33.0%），MRR 从 0.619 提升至 0.928（+49.9%）。

**实测延迟**（本地 CPU，2026-09-04）：检索链路（门限+双路召回+重排）< 4s；端到端 7-11s，波动主要来自 LLM 服务端首 token 延迟。

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

**B. 命令行模拟面试**（自由对话式面试官：话题锚点 + 即兴追问 + 复盘 + 证据核实 + 报告导出）：

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
python tests/test_agent.py        # 面试全流程（自由对话）
python tests/test_followup.py     # 决策路由与护栏
python tests/test_topics.py       # 话题锚点抽取
python tests/test_evidence.py     # 证据核实
python tests/test_tool.py         # Tool Calling
python tests/test_fastapi.py      # FastAPI 接口
python tests/test_models.py       # 模型工厂多后端解析
```

## 项目结构

```
├── app.py            # RAG 问答：加载→切分→向量化→混合检索→重排→拒答→生成
├── retrieval.py      # BM25 索引、RRF 融合、CrossEncoder 重排
├── agent.py          # LangGraph 面试状态机 + LLM 决策路由 + 话题锚点 + 证据核实 + Tool
├── models.py         # 模型工厂：DeepSeek / OpenAI 兼容（GLM 等），env 切换
├── api.py            # FastAPI 服务层（/chat、/upload）
├── ui.py             # Streamlit 界面（问答 + 模拟面试）
├── data/             # 知识库（gitignore）
├── tests/            # 9 个测试脚本 + 23 题测试集
└── chroma_db/        # 向量库持久化（gitignore）
```

## 设计决策

- **为什么模型层用 DeepSeek**：RAG 架构与模型解耦，模型层可插拔；DeepSeek 是成本/可用性决策，切换其他模型只需改配置（实测 DeepSeek + OpenAI 兼容端点）。
- **为什么自由对话用状态机实现而非纯 LLM 对话**：纯自由对话会跑偏（话题漂移、忘记面试目的、迟迟不结束）；状态机 + LLM 决策路由让对话自由但有边界——话题锚点限定出题范围，轮数与覆盖率护栏强制收尾，每轮决策结构化、可解释。
- **拒答门限为什么用向量相似度而非重排分数**：英文预训练的重排模型对中文口语问法打分不稳定（实测相关问法低至 0.019 与无关问法 0.028 无法区分）；中文 embedding 的相似度信号实测相关 0.283~0.656 vs 无关 0.011~0.127，可稳定分离。

v1 是反馈式循环：固定题单 + 每答必点评，点评占据主导、提问机械推进，不像真实面试；v2 把点评整体挪到 wrap 环节，中间过程变成面试官主导的连续追问。
