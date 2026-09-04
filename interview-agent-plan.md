# 面试官模拟 RAG Agent — PRD 与项目规划

> 版本:v3.0 | 最近更新:2026-09-03 | 状态:开发中 | 负责人:毛骊达

---

## 一、PRD(产品需求文档)

### 1.1 项目定位

面向应届求职者的**AI 面试陪练工具**:上传简历、项目文档、目标岗位 JD,AI 模拟面试官按 JD 提问、追问,并对回答做结构化点评。

**差异化定位**(相对 GitHub 同类项目):
- 同类项目多为 Java 栈、重部署(Docker + MySQL + Redis);本项目**纯 Python、单机轻量**,更贴合 AI 应用岗位
- 检索质量做深:**混合检索 + RRF 融合 + 重排**,而非单一向量检索
- **引用溯源 + 证据核实**双保险防幻觉——面试场景下答错引用比答不出更致命
- 面试对话是**受控的自由对话**:LangGraph 状态机 + LLM 决策路由,而非固定题单的机械问答

### 1.2 用户故事

> 作为求职者,我想上传简历/项目文档/目标 JD,让 AI 按 JD 模拟面试官提问、追问、点评我的回答(且所有提问和点评都有原文依据),以便在真实面试前反复练习并拿到复盘报告。

### 1.3 功能需求(按优先级)

| 优先级 | 功能 | 说明 |
|---|---|---|
| **P0**(没做完项目不成立) | 知识库上传 | 简历、项目文档、JD 三类材料,支持 md/txt/pdf |
| P0 | 基础 RAG 问答 | 向量检索 + 生成,回答附原文引用 |
| P0 | JD+简历定制话题 | 从 JD 技术要求 + 简历项目经历抽取 8 个面试话题锚点 |
| **P1**(重要,简历核心亮点) | 混合检索 + RRF 融合 | BM25 关键词召回 + 向量语义召回,RRF 融合排序 |
| P1 | 重排 Rerank | CrossEncoder 对候选精排,提升 top-k 命中率 |
| P1 | 引用溯源 | 每个回答强制标注原文出处,无出处即拒答 |
| P1 | 面试状态机(LangGraph) | 每轮一次图:护栏检查→LLM 决策路由(深挖/换题/结束)→提问,自由对话不跑偏 |
| P1 | 复盘点评 | wrap 节点集中点评:整体表现 / 最突出优点 / 最需改进,不再每答打断 |
| P1 | 证据核实 | 复盘时核对回答中带硬数字/指标的内容(混合检索定位 + LLM 确认),不做开放语义冲突检测 |
| P1 | Tool Calling 封装 | 复盘导出封装为 Tool(JSON Schema 定义),wrap 节点由 LLM 自主调用 |
| **P2**(时间允许再做) | 离线评估 + 消融实验 | 自建测试集,Recall@5/MRR 指标,对比纯向量 vs 混合检索 |
| P2 | 面试复盘报告 | 每场面试结束导出 markdown 复盘报告 |
| P2 | RESTful API 层 | FastAPI 暴露 /chat、/upload 接口,与 Streamlit 前端解耦 |
| **不做**(防范围蔓延边界) | 语音对话 | 本版不做 |
| 不做 | 自动爬取招聘 JD | 本版不做,JD 手动粘贴 |

### 1.4 非功能需求

- 单机运行,无云端部署要求
- 检索链路(门限+双路召回+重排)< 4s(2026-09-04 实测 3.1-3.3s);端到端 7-11s,波动来自 LLM 服务端首 token 延迟
- 复盘点评 3 部分:整体表现总结 / 最突出优点 / 最需改进
- 按 JD+简历抽取 8 个话题锚点,覆盖 JD 技术要求与项目经历
- 回答必须携带引用,无引用不输出(硬约束)
- 追问时机由 LLM 判断(回答过短 / 缺量化 / 回避 / 有值得深挖的细节),prompt 软引导
- 同一话题追问上限 3 轮(MAX_FOLLOWUPS 程序硬护栏,超限强制切题),防无限深挖

### 1.5 技术栈(已定,见决策记录)

LangChain(`models.get_chat_model` 工厂:init_chat_model 推断 + OpenAI 兼容端点兜底,DeepSeek 默认实测 / GLM 等兼容厂商可接 / Claude 分支代码就绪,锁区不实测)+ LangGraph + ChromaDB + FastAPI + Streamlit
Embedding:本地 `bge-small-zh-v1.5`(HuggingFace,已落地);Reranker:本地 `BGE-reranker-base`(CrossEncoder,已落地)

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

| 里程碑 | 截止 | 实际 | 交付物 |
|---|---|---|---|
| **M1** | 2026-09-02 | ✅ 9.1 | 基础 RAG 闭环跑通(脚本层):上传文档→向量检索→带引用回答 |
| **M2** | 2026-09-06 | ✅ 9.2 | 检索质量层:混合检索(BM25+RRF)+ Rerank + 引用溯源硬约束 + 23 条测试集 |
| **M3** | 2026-09-13 | ✅ 9.3 | 面试状态机(LangGraph)+ 话题锚点 + LLM 决策路由 + 复盘点评 + 证据核实 + Tool |
| **M4** | 2026-09-18 | ✅ 9.4 | API+UI+消融评估+README+简历 STAR 包装(5.6)+验收 6 条核对,全部完成 |

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
   3.5 建 23 条测试问答集(含口语化问法),每加一层检索立刻测一轮(Recall@5/MRR),消融数据 M2 内成型

4. 面试状态机(M3,已按 v2 完成)                 [任务 #3]
   4.1 LangGraph StateGraph:每轮一次图——START→护栏检查→LLM 决策节点→提问/复盘→END;对话循环由 Python 驱动
   4.2 JD+简历抽取话题锚点(8 个):JD 技术要求与项目经历双覆盖,面试围绕锚点自由发挥
   4.3 复盘点评(wrap 集中):整体表现总结 + 最突出优点 + 最需改进,不再每答打断
   4.4 追问(LLM 判断 + 程序硬护栏):深挖时机交给 LLM(过短/缺量化/回避/值得挖的细节),同话题最多 3 轮(MAX_FOLLOWUPS),超限 _sanitize_decision 强制切题
   4.5 证据核实(降级版):wrap 一次性核对回答中含硬数字/指标的内容(混合检索定位 + LLM 确认),不做开放语义冲突检测
   4.6 Tool Calling:复盘报告导出 1 个 Tool(JSON Schema 定义),wrap 节点 bind_tools 由 LLM 自主调用

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
| R1 | Embedding 方案未定,阻塞 M1 | ~~二选一:本地 BGE 中文模型 / 硅基流动 API~~ ✅ 已解除(9.2):本地 bge-small-zh-v1.5 落地,离线免费,机器可跑 |
| R2 | Reranker 方案未定,阻塞 M2 | ~~二选一:硅基流动 API / 本地 CrossEncoder~~ ✅ 已解除(9.2):本地 BGE-reranker-base 落地(注意:对中文口语问法打分不稳,拒答门限改用向量相似度,见 9dcf6de) |
| R3 | LangGraph 学习成本拖慢 M3 | 只学 StateGraph 核心(节点/边/状态),不碰高级特性;先画状态图再写代码 |
| R4 | 范围蔓延(想加语音、爬虫) | P2 清单已白纸黑字"本版不做" |
| R5 | 项目挤压投递时间 | 投递 9 月第一周启动,与开发并行 |
| R6 | DeepSeek API 限流/费用 | 控制单次调用 token;检索结果精简后拼接 |
| R7 | Claude API 锁区无法注册(大陆账号有封号风险) | 第二实测后端改用 OpenAI 兼容国内模型(智谱 GLM 等,国内可注册);Claude 分支代码就绪不实测,README/文档如实标注 |

### 3.4 验收标准(M4 时逐条核对)

> ✅ 验收核对(2026-09-04):10 个测试脚本全部通过(agent/api/evidence/fastapi/followup/guardrail/models/mvp/tool/topics),6 条验收标准逐条满足。M4 完成。

1. 上传简历+项目+JD 三类材料后,就简历内容提问,能得到**带原文引用**的回答;检索无命中时明确拒答而非编造
2. 按 JD+简历抽取 8 个话题锚点,面试覆盖 JD 技术要求与候选人项目经历
3. 整场面试结束给出复盘点评(整体表现/优点/改进),证据核实能指出回答与简历原文硬事实冲突
4. 混合检索 + Rerank 在自建测试集上 Recall@5/MRR **优于**纯向量基线(消融实验表为证)
5. 面试由 LangGraph + LLM 决策路由驱动,自由对话全程有护栏(追问上限/覆盖率/轮数),可推进到复盘收尾
6. Streamlit 界面可用,核心功能可通过 FastAPI 接口(curl)调用,全流程无命令行操作

---

## 四、决策记录(同步自 MEMORY.md)

| 日期 | 决策 | 要点 |
|---|---|---|
| 2026-08-28 | 学习路线:项目驱动冲刺 | 直接做项目边做边学,2-3 周出成品;否决"先系统学再动手"和"原生 Agent 路线" |
| 2026-08-28 | 项目选题:面试官模拟 RAG Agent | 一个项目覆盖 RAG+Agent 双关键词;否决视觉问答(时间风险)、求职情报(杂活多) |
| 2026-08-28 | 模型:纯 DeepSeek | 现成 key、零成本、国内网络稳;代价是简历缺 Claude API 关键词 |
| 2026-09-01 | 模型层:多后端可插拔 | 补 Claude API 经验缺口(招聘需求 P0 隐含要求);DeepSeek 仍为默认(成本/国内网络),Claude 用 Haiku 验证即可 |
| 2026-09-03 | 对话模式:v1 固定流程 → v2 自由决策路由 | 刻意重构(9.3 commit 7b524db):v1 是 9.2 的反馈式循环——固定题单不感知回答内容、每答必点评打断节奏、追问规则穷举有天花板;v2 话题锚点 + LLM 决策路由(followup/switch_topic/end),点评集中到 wrap,程序护栏(追问上限/覆盖率/轮数)防跑偏 |
| 2026-09-03 | 多后端落地:models.get_chat_model 工厂 | 封装 init_chat_model + OpenAI 兼容端点兜底(commit c9f3bca/c44ad77):DeepSeek 默认实测;GLM 等国内模型走 CHAT_BASE_URL 可实测;Claude 分支代码就绪——后发现 Anthropic 锁国区、注册有封号风险,放弃实测,原「Haiku 验证几毛钱」预案作废(见 R7) |
| 2026-09-03 | 追问上限硬化:3 轮程序护栏 | 同一话题追问上限原为 prompt 软约束(2 轮,LLM 可能违反);改为 MAX_FOLLOWUPS=3 + state 计数,_sanitize_decision 超限强制切题(commit 10ce3ae) |

**面试话术备忘**:"为什么用 DeepSeek?"→ RAG 架构与模型解耦,模型层可插拔——项目实际跑通 DeepSeek 与 OpenAI 兼容双后端(可接智谱 GLM 等国内模型);Claude 分支代码就绪(LangChain init_chat_model 原生支持),因 Anthropic 不对大陆开放未实测;默认 DeepSeek 是成本/可用性决策,切换只改 .env 三行。

---

## 五、简历包装要点

**技术亮点 6 条(简历 STAR 条目素材)**:

1. **混合检索**:BM25 关键词 + 向量语义双路召回,RRF 融合——解决 JD 术语等精确词召回差的问题
2. **重排**:CrossEncoder 精排候选,提升 top-k 命中(有消融数据可讲)
3. **引用溯源**:回答强制携带原文出处,无命中即拒答——防幻觉的硬约束设计
4. **LangGraph + LLM 决策路由**:自由对话式面试,每轮动作结构化(深挖/换题/结束)可解释,程序护栏防跑偏
5. **证据核实**:点评时检测回答与简历的冲突和无依据表述
6. **离线评估**:自建测试集,Recall@5/MRR 指标,纯向量 vs 混合 vs 混合+重排的消融对比
7. **Tool Calling**:证据核实、复盘导出封装为 Tool(JSON Schema 定义),状态机按需调用
8. **多模型可插拔**:LangChain init_chat_model 双后端,DeepSeek 默认 / Claude 可切换
9. **FastAPI 服务层**:/chat、/upload 接口化交付,前端与核心逻辑解耦

**高频面试问题预答**:
- "为什么混合检索?" → 单向量检索对术语/精确词召回差;BM25 补关键词,RRF 融合两者
- "怎么防幻觉?" → 三层:检索层(无命中拒答)、生成层(prompt 强制引用)、点评层(证据核实)
- "自由对话为什么还要用状态机?" → 纯自由对话会跑偏(话题漂移、忘面试目的、迟迟不结束);LangGraph 每轮走「护栏→决策→提问」:LLM 决定"说什么"(深挖/换题/结束均为结构化输出),程序保证"不能跑偏"——话题锚点限范围、同话题追问 3 轮硬上限、覆盖率+轮数强制收尾、非法决策自动兜底。追问策略因答而异,又结构化可解释
- "LangChain Agent 和直接调 API 有什么区别?" → 直接调 API 是固定流程,每次都要自己编排;Agent 是 LLM 在循环里自主决定调用哪个工具、何时结束。我的项目两者都有:检索问答是确定性链路(不套 Agent),追问深挖与证据核实是受控 Agent 循环(状态机限定状态,不自由发散)
- "什么时候该用 Agent,什么时候不该用?" → 判断三件事:任务是否开放、是否需要多步动态决策、错误成本是否可容忍。检索问答流程固定 → 不该用 Agent;面试追问要根据回答动态深入 → 该用,但用状态机约束边界
- "你怎么让 AI 生成正确的 Tool 定义?生成的代码有 bug 怎么定位?" → 先给接口契约(JSON Schema + 输入输出示例)再让 AI 生成,生成后立刻写最小用例验证;有 bug 就把报错回贴给 AI 并加约束重生成,复杂问题自己读代码定位到行再让它修
- "平时怎么用 AI 写代码?" → 结构化需求先行(先写 PRD/接口契约)→ 让 AI 生成 → 审查 → 出错时把报错上下文回贴、加约束重生成 → 我来做架构设计和集成验证,AI 做实现。讲 2-3 个真实纠错案例(素材来自开发过程记录,每天 3 行)

---

## 六、进度追踪

| 任务 | 状态 | 备注 |
|---|---|---|
| #1 环境搭建 | ✅ 9.1-9.3 | 依赖已装;模型工厂 models.py(9.3,见决策记录);Claude 分支代码就绪不实测(锁区,R7) |
| #2 基础 RAG + 检索质量层 | ✅ 9.1-9.2 | M1 9.1 / M2 9.2 完成;embedding=本地 bge-small-zh-v1.5,reranker=本地 BGE-reranker-base |
| #3 面试状态机 | ✅ 9.3 | v2 自由决策路由版(7b524db 起)+ 追问 3 轮硬护栏(10ce3ae);9.2 的 v1 固定流程已被刻意推翻,见决策记录 |
| #4 API+UI+评估+包装+投递 | 🟡 进行中 | API/UI/README/消融表已提交(9.2);剩:简历 STAR 条目更新(5.6)+ 验收标准 6 条逐条核对;投递为并行线,见实施统筹 |
