# 面试官模拟 RAG Agent — PRD 与项目规划

> 版本:v2.0 | 日期:2026-08-29 | 状态:开发中 | 负责人:毛骊达

---

## 一、PRD(产品需求文档)

### 1.1 项目定位

面向应届求职者的**AI 面试陪练工具**:上传简历、项目文档、目标岗位 JD,AI 模拟面试官按 JD 提问、追问,并对回答做结构化点评。

**差异化定位**(相对 GitHub 同类项目):
- 同类项目多为 Java 栈、重部署(Docker + MySQL + Redis);本项目**纯 Python、单机轻量**,更贴合 AI 应用岗位
- 检索质量做深:**混合检索 + RRF 融合 + 重排**,而非单一向量检索
- **引用溯源 + 证据核实**双保险防幻觉——面试场景下答错引用比答不出更致命
- 面试流程用 **LangGraph 状态机**建模,可控可解释,而非自由对话

### 1.2 用户故事

> 作为求职者,我想上传简历/项目文档/目标 JD,让 AI 按 JD 模拟面试官提问、追问、点评我的回答(且所有提问和点评都有原文依据),以便在真实面试前反复练习并拿到复盘报告。

### 1.3 功能需求(按优先级)

| 优先级 | 功能 | 说明 |
|---|---|---|
| **P0**(没做完项目不成立) | 知识库上传 | 简历、项目文档、JD 三类材料,支持 md/txt/pdf |
| P0 | 基础 RAG 问答 | 向量检索 + 生成,回答附原文引用 |
| P0 | JD 定制出题 | 按目标岗位 JD 生成定制面试题 |
| **P1**(重要,简历核心亮点) | 混合检索 + RRF 融合 | BM25 关键词召回 + 向量语义召回,RRF 融合排序 |
| P1 | 重排 Rerank | CrossEncoder 对候选精排,提升 top-k 命中率 |
| P1 | 引用溯源 | 每个回答强制标注原文出处,无出处即拒答 |
| P1 | 面试状态机(LangGraph) | 开场→出题→问答→追问→点评→复盘,受控状态流转 |
| P1 | 回答点评 | 结构化点评:STAR 完整性 / 技术深度 / 表达清晰度 |
| P1 | 证据核实 | 简历关键事实核对(关键词定位+LLM 确认),不做开放语义冲突检测 |
| P1 | Tool Calling 封装 | 复盘导出封装为 Tool(JSON Schema 定义),由状态机按需调用 |
| **P2**(时间允许再做) | 离线评估 + 消融实验 | 自建测试集,Recall@5/MRR 指标,对比纯向量 vs 混合检索 |
| P2 | 面试复盘报告 | 每场面试结束导出 markdown 复盘报告 |
| P2 | RESTful API 层 | FastAPI 暴露 /chat、/upload 接口,与 Streamlit 前端解耦 |
| **不做**(防范围蔓延边界) | 语音对话 | 本版不做 |
| 不做 | 自动爬取招聘 JD | 本版不做,JD 手动粘贴 |

### 1.4 非功能需求

- 单机运行,无云端部署要求
- 检索+生成响应时间 < 10s
- 回答点评 ≥ 3 个维度
- 按 JD 生成面试题 ≥ 10 道
- 回答必须携带引用,无引用不输出(硬约束)
- 追问触发条件:回答过短 / 缺少量化细节 / 回避问题,满足其一才追问
- 追问上限:最多 2 轮,防无限深挖

### 1.5 技术栈(已定,见决策记录)

LangChain(init_chat_model 多后端:DeepSeek 默认 / Claude 可切换)+ LangGraph + ChromaDB + FastAPI + Streamlit
Embedding / Reranker 方案:**未定**,M1 前定(见风险表 R1/R2)

---

## 二、竞品调研(2026-08-29,GitHub)

| 项目 | 栈 | 可借鉴点 |
|---|---|---|
| [AURORA / spi-interview-agent](https://github.com/shspic/spi-interview-agent) | Python:FastAPI + LangGraph + Chroma + BGE + RRF | **与本项目最接近**。LangGraph 受控面试状态流、RRF 融合检索、证据核实(识别资料冲突与无依据表达) |
| [Snailclimb/interview-guide](https://github.com/Snailclimb/interview-guide)(~2700★) | Java:Spring AI + pgvector + Redis Stream | 简历分析+模拟面试+RAG 问答闭环;SSE 流式、Redis Stream 异步向量化 |
| [369pro/InterView-Agent](https://github.com/369pro/InterView-Agent) | Python:LangGraph + DeepSeek + Milvus | 混合 RAG 出题(BM25+BGE-M3+reranker)、自适应难度、评估报告 |
| [Zqyuaaan/MirrorAgent](https://github.com/Zqyuaaan/MirrorAgent) | Java:Spring AI Alibaba + Milvus | 三阶段面试节奏(基础→项目→系统设计)、离线 RAG 评估(Recall@10/MRR) |
| [k8nice/smile-boss-ai](https://github.com/k8nice/smile-boss-ai) | Java:Spring Boot 多模块 | 多 Agent 协作、OpenAI 兼容多模型路由(Claude/Qwen/DeepSeek/Kimi 可切换) |
| RAG 工程参考:[avuppal/enterprise-rag](https://github.com/avuppal/enterprise-rag) | Python | 混合 RRF + CrossEncoder + MMR,完整评估套件;实测混合检索 Recall@5 0.71→0.79,加重排→0.84 |
| RAG 工程参考:[tainguyen07/rag-doc-qa](https://github.com/tainguyen07/rag-doc-qa) | Python:LangChain + Chroma | 引用溯源、流式 API、Ragas CI 评估 |

**结论**:检索链路(混合+RRF+重排+评估)照搬成熟模式;差异化落在**面试状态机 + 证据核实 + Tool Calling 封装 + 多模型后端 + FastAPI 服务层 + 纯 Python 轻量栈**。

---

## 三、项目规划

### 3.1 里程碑

| 里程碑 | 截止 | 交付物 |
|---|---|---|
| **M1** | 2026-09-02 | 基础 RAG 闭环跑通(脚本层):上传文档→向量检索→带引用回答 |
| **M2** | 2026-09-06 | 检索质量层:混合检索(BM25+RRF)+ Rerank + 引用溯源硬约束 + 20 条测试集初建 |
| **M3** | 2026-09-13 | 面试状态机(LangGraph)+ JD 定制出题 + 追问 + 点评 + 证据核实 |
| **M4** | 2026-09-18 | FastAPI 接口层 + Streamlit 界面 + 离线评估消融实验 + README + 简历包装 |

投递:9 月第一周并行启动,不等 M4。

### 3.2 WBS(工作分解)

```
1. 环境搭建(8.29-8.30)                       [任务 #1]
   1.1 pip 安装 langchain / langchain-deepseek / langchain-anthropic / chromadb / fastapi / streamlit
   1.2 设置 DEEPSEEK_API_KEY 环境变量
   1.3 最小脚本验证 DeepSeek API 调用
   1.4 模型层多后端配置(init_chat_model:DeepSeek 默认 / Claude 可切换),双端各跑通一次

2. 基础 RAG 闭环(M1)                         [任务 #2]
   2.1 定 embedding 方案(本地 BGE / 硅基流动 API,二选一)
   2.2 准备知识库材料(简历 STAR 文档、毕设、Flask 项目、目标 JD)
   2.3 链路:文档加载 → 切分(recursive 512 token + 10% 重叠)→ 向量化 → ChromaDB
   2.4 检索 + DeepSeek 生成带引用回答
   2.5 脚本层验证闭环

3. 检索质量层(M2)                            [任务 #2]
   3.1 BM25 关键词召回(rank_bm25)
   3.2 RRF 融合排序(k=60)
   3.3 CrossEncoder 重排(BGE-reranker,API 或本地)
   3.4 引用溯源硬约束:prompt 层强制 + 无出处拒答
   3.5 建 20 条测试问答集,每加一层检索立刻测一轮(Recall@5/MRR),消融数据 M2 内成型

4. 面试状态机(M3)                            [任务 #3]
   4.1 LangGraph StateGraph:开场→出题→问答→追问→点评→复盘
   4.2 JD 定制出题(≥10 道)
   4.3 点评(≥3 维度:STAR/技术深度/表达)
   4.4 追问(降级版):3 条触发规则(回答过短/缺量化/回避问题),最多 2 轮
   4.5 证据核实(降级版):简历关键事实核对(关键词定位 + LLM 确认),不做开放语义冲突检测
   4.6 Tool Calling:仅复盘导出 1 个 Tool(JSON Schema 定义),由状态机按需调用

5. API + UI + 评估 + 包装(M4)                [任务 #4]
   5.1 FastAPI 后端(/chat、/upload 接口)+ Streamlit 界面(聊天窗口、引用展示、JD 切换)
   5.2 评估补测:测试集 M2 已建,M4 补测 + 整理消融对比表
   5.3 消融实验:纯向量 vs 混合检索 vs 混合+重排 对比表
   5.4 复盘报告导出(markdown)
   5.5 README(架构图、使用说明、技术亮点)
   5.6 简历 STAR 条目包装
   5.7 投递并行:9 月第一周开始,不等 M4
```

### 3.3 风险与预案

| # | 风险 | 预案 |
|---|---|---|
| R1 | Embedding 方案未定,阻塞 M1 | 二选一:本地 BGE 中文模型(免费无网络,吃内存)/ 硅基流动免费 embedding API(需注册) |
| R2 | Reranker 方案未定,阻塞 M2 | 二选一:硅基流动 BGE-reranker API / 本地 CrossEncoder 小模型;本地优先,API 兜底 |
| R3 | LangGraph 学习成本拖慢 M3 | 只学 StateGraph 核心(节点/边/状态),不碰高级特性;先画状态图再写代码 |
| R4 | 范围蔓延(想加语音、爬虫) | P2 清单已白纸黑字"本版不做" |
| R5 | 项目挤压投递时间 | 投递 9 月第一周启动,与开发并行 |
| R6 | DeepSeek API 限流/费用 | 控制单次调用 token;检索结果精简后拼接 |
| R7 | Claude API 注册/计费门槛 | 仅用 Haiku 做最小验证(几毛钱),生产仍走 DeepSeek;双后端只求跑通不求压测 |

### 3.4 验收标准(M4 时逐条核对)

1. 上传简历+项目+JD 三类材料后,就简历内容提问,能得到**带原文引用**的回答;检索无命中时明确拒答而非编造
2. 按 JD 生成 ≥10 道定制面试题
3. 对任意回答给出 ≥3 个维度的点评,且能指出与简历冲突/无依据之处
4. 混合检索 + Rerank 在自建测试集上 Recall@5/MRR **优于**纯向量基线(消融实验表为证)
5. 面试全流程由 LangGraph 状态机驱动,可正常走完开场→复盘
6. Streamlit 界面可用,核心功能可通过 FastAPI 接口(curl)调用,全流程无命令行操作

---

## 四、决策记录(同步自 MEMORY.md)

| 日期 | 决策 | 要点 |
|---|---|---|
| 2026-08-28 | 学习路线:项目驱动冲刺 | 直接做项目边做边学,2-3 周出成品;否决"先系统学再动手"和"原生 Agent 路线" |
| 2026-08-28 | 项目选题:面试官模拟 RAG Agent | 一个项目覆盖 RAG+Agent 双关键词;否决视觉问答(时间风险)、求职情报(杂活多) |
| 2026-08-28 | 模型:纯 DeepSeek | 现成 key、零成本、国内网络稳;代价是简历缺 Claude API 关键词 |
| 2026-09-01 | 模型层:多后端可插拔 | 补 Claude API 经验缺口(招聘需求 P0 隐含要求);DeepSeek 仍为默认(成本/国内网络),Claude 用 Haiku 验证即可 |

**面试话术备忘**:"为什么用 DeepSeek?"→ RAG 架构与模型解耦,模型层可插拔——项目实际跑通 DeepSeek 与 Claude 双后端,默认 DeepSeek 是成本/可用性决策,切换只改一行配置。

---

## 五、简历包装要点

**技术亮点 6 条(简历 STAR 条目素材)**:

1. **混合检索**:BM25 关键词 + 向量语义双路召回,RRF 融合——解决 JD 术语等精确词召回差的问题
2. **重排**:CrossEncoder 精排候选,提升 top-k 命中(有消融数据可讲)
3. **引用溯源**:回答强制携带原文出处,无命中即拒答——防幻觉的硬约束设计
4. **LangGraph 状态机**:面试流程建模为受控状态图,追问策略可解释、流程可回溯
5. **证据核实**:点评时检测回答与简历的冲突和无依据表述
6. **离线评估**:自建测试集,Recall@5/MRR 指标,纯向量 vs 混合 vs 混合+重排的消融对比
7. **Tool Calling**:证据核实、复盘导出封装为 Tool(JSON Schema 定义),状态机按需调用
8. **多模型可插拔**:LangChain init_chat_model 双后端,DeepSeek 默认 / Claude 可切换
9. **FastAPI 服务层**:/chat、/upload 接口化交付,前端与核心逻辑解耦

**高频面试问题预答**:
- "为什么混合检索?" → 单向量检索对术语/精确词召回差;BM25 补关键词,RRF 融合两者
- "怎么防幻觉?" → 三层:检索层(无命中拒答)、生成层(prompt 强制引用)、点评层(证据核实)
- "为什么用状态机?" → 面试有固定流程,状态机保证完整走完且追问可解释;自由对话会跑偏
- "LangChain Agent 和直接调 API 有什么区别?" → 直接调 API 是固定流程,每次都要自己编排;Agent 是 LLM 在循环里自主决定调用哪个工具、何时结束。我的项目两者都有:检索问答是确定性链路(不套 Agent),追问深挖与证据核实是受控 Agent 循环(状态机限定状态,不自由发散)
- "什么时候该用 Agent,什么时候不该用?" → 判断三件事:任务是否开放、是否需要多步动态决策、错误成本是否可容忍。检索问答流程固定 → 不该用 Agent;面试追问要根据回答动态深入 → 该用,但用状态机约束边界
- "你怎么让 AI 生成正确的 Tool 定义?生成的代码有 bug 怎么定位?" → 先给接口契约(JSON Schema + 输入输出示例)再让 AI 生成,生成后立刻写最小用例验证;有 bug 就把报错回贴给 AI 并加约束重生成,复杂问题自己读代码定位到行再让它修
- "平时怎么用 AI 写代码?" → 结构化需求先行(先写 PRD/接口契约)→ 让 AI 生成 → 审查 → 出错时把报错上下文回贴、加约束重生成 → 我来做架构设计和集成验证,AI 做实现。讲 2-3 个真实纠错案例(素材来自开发过程记录,每天 3 行)

---

## 六、进度追踪

| 任务 | 状态 | 备注 |
|---|---|---|
| #1 环境搭建 | ⬜ 待开始 | 依赖未安装;含模型层双后端验证(Claude 需注册 key,R7) |
| #2 基础 RAG + 检索质量层 | ⬜ 待开始 | 阻塞项:embedding 选型(R1)、reranker 选型(R2) |
| #3 面试状态机 | ⬜ 待开始 | LangGraph 需学习(R3) |
| #4 API+UI+评估+包装+投递 | ⬜ 待开始 | 含 FastAPI 层;投递部分 9 月第一周独立启动 |
