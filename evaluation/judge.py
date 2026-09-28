"""面试官提问质量的 LLM-as-judge 评判器（三维度 rubric）。

judge 运行在与生成模型不同的后端上（见
models.get_judge_model），以保证打分公正。每个维度是
1-5 的整数并配有锚点描述；评判结果还为每个维度附带一句理由，
供人工抽查。
"""
from pydantic import BaseModel, Field

from models import STRUCTURED_METHOD, get_judge_model


class JudgeVerdict(BaseModel):
    """judge 的结构化输出：三个维度的分数 + 理由。"""

    relevance: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    depth: int = Field(ge=1, le=5)
    reason: str


JUDGE_PROMPT = (
    "你是面试质量评委。对面试官提出的一个问题从三个维度打分（1-5 整数）。\n\n"
    "维度与锚点：\n"
    "1. relevance 相关性：问题是否贴合当前话题与对话语境\n"
    "   1=完全跑题或与上文无关；3=沾边但偏离核心；5=精准围绕话题，切中上下文\n"
    "2. specificity 具体性：问题是否具体、可回答、可展开\n"
    "   1=空泛无法回答（如「你有什么亮点」）；3=可回答但范围模糊；5=指向明确、有具体抓手\n"
    "3. depth 深挖引导力：问题是否能引出量化数据、技术取舍或困难反思\n"
    "   1=只能引出是/否或表面回答；3=能引出一般性展开；5=逼出量化细节、取舍依据或复盘反思\n\n"
    "输出：三个整数分 + reason（每维一句理由，共三句）。\n\n"
    "话题锚点：{topic}\n"
    "最近对话：\n{history}\n"
    "待评问题：{question}"
)


def judge_question(topic: str, history: str, question: str, llm=None) -> JudgeVerdict:
    """给面试官的一个问题打分；返回一个 JudgeVerdict。"""
    llm = llm or get_judge_model()
    prompt = JUDGE_PROMPT.format(topic=topic, history=history, question=question)
    return llm.with_structured_output(JudgeVerdict, method=STRUCTURED_METHOD).invoke(prompt)


def summarize(verdicts: list[JudgeVerdict]) -> dict:
    """把多条评判结果聚合成均分和通过率（每个维度 >=4 算通过）。"""
    if not verdicts:
        return {"n": 0}
    dims = ("relevance", "specificity", "depth")
    out = {"n": len(verdicts)}
    for d in dims:
        scores = [getattr(v, d) for v in verdicts]
        out[f"{d}_mean"] = round(sum(scores) / len(scores), 2)
        out[f"{d}_pass"] = round(sum(1 for s in scores if s >= 4) / len(scores), 3)
    return out
