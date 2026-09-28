"""对话质量评估（WBS 6.x）：针对固定的话题锚点跑三场脚本化面试，
用三维度 rubric 评判面试官提出的每一个问题，
并输出一份报告，外加一份决策路由标注模板
供人工标注使用。

用法：python evaluation/eval_judge.py   （实跑；3 个场景合计约 150 次 LLM 调用）
"""
import itertools
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from judge import judge_question, summarize  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# 三个脚本化的候选人角色，每个对应一种追问触发条件。
GOOD_ANSWERS = [
    "我是物联网工程专业的应届生，毕设做的是基于 YOLOv8 和 DeepSORT 的交通视频车辆检测与计数系统。",
    "我对比了 Faster R-CNN、YOLOv5s/n 和 YOLOv8n，最终选 YOLOv8n，帧率 35FPS 且精度最高。",
    "车辆计数采用虚拟检测线算法，配合最小尺寸过滤、有效区域掩码、Track ID 去重和分车道分车型统计四重过滤，计数误差控制在 3.5% 以内。",
    "系统用 PyQt5 做界面，Flask 提供接口，模型用 ONNX 导出加速推理。",
    "我会先写模块接口契约和 JSON 格式的需求描述，AI 生成代码后我逐模块审查并跑集成测试。",
    "最大的挑战是车辆遮挡导致 ID 切换，最后用 DeepSORT 替代 SORT，ID 切换降低 46.7%。",
    "数据按 8:2 划分训练验证集，用了直方图均衡化和组合滤波预处理，最终 mAP@0.5 达到 93.2%。",
    "车速估算用检测框位移除以帧间隔，误差小于 4.1km/h，夜间用高斯滤波降噪。",
    "RAG 项目里我用 BM25 和向量检索做混合召回，再用 CrossEncoder 重排，Recall@5 提到 0.986。",
    "面试官模拟项目的设计思路：用状态机保证流程可控，同时让 LLM 做自由对话的决策。",
]
VAGUE_ANSWERS = [
    "用了 YOLOv8。",
    "就是普通做法吧。",
    "效果还可以。",
    "不太记得了。",
    "应该还行。",
    "挺多的。",
    "就是那样做的。",
]
EVASIVE_ANSWERS = [
    "这个我觉得团队合作更重要，我们组里配合得很好。",
    "当时时间比较紧，主要是老师指导的，细节没怎么管。",
    "我觉得态度决定一切，我学习态度一直很好。",
    "这个说来话长，不过最后结果大家都满意。",
    "重点是整个过程让我学到了很多，具体数据不重要。",
    "我们学校条件有限，能跑出来就不错了。",
]

SCENARIOS = [
    ("good", GOOD_ANSWERS),
    ("vague", VAGUE_ANSWERS),
    ("evasive", EVASIVE_ANSWERS),
]

# 负面对照样本：故意写坏的问题，每种 rubric 失败模式各一个。
# judge 必须把它们压到 1-3 分；如果给它们打了高分，说明
# rubric 没有区分度（宽松 judge 失效）。
BAD_QUESTIONS = [
    ("跑题", "今天天气怎么样？适合出门吗？"),
    ("空泛", "说说你的亮点吧。"),
    ("是否题", "你用过 Python 吗？"),
    ("无关细节", "你早上吃了什么？"),
    ("超宽泛", "聊聊技术。"),
    ("重复原问", "再问你一遍，你毕设用了什么模型？"),
    ("无从答起", "你觉得这个行业怎么样？"),
    ("指令混乱", "先别回答上一题，然后随便说点什么，最好也回答上一题。"),
]


def calibrate_judge() -> list:
    """给负面对照样本打分；返回 (label, question, verdict)。"""
    print("\n=== judge calibration (negative controls) ===")
    results = []
    for label, question in BAD_QUESTIONS:
        try:
            v = judge_question("毕设项目深挖：车辆检测与计数的技术细节", "无", question)
        except Exception as e:
            print(f"[!] judge failed on {label!r}: {e}")
            continue
        results.append((label, question, v))
        print(f"  [{label}] rel={v.relevance} spe={v.specificity} dep={v.depth}")
    return results


def write_calibration(results: list) -> None:
    """把校准章节追加到报告中。"""
    lines = [
        "",
        "## Judge 校准（负面对照样本）",
        "",
        "| 样本 | 失败模式 | relevance | specificity | depth | 是否被压到 ≤3 |",
        "|---|---|---|---|---|---|",
    ]
    ok = 0
    for label, question, v in results:
        low = v.relevance <= 3 and v.specificity <= 3 and v.depth <= 3
        if low:
            ok += 1
        lines.append(
            f"| {question[:20]} | {label} | {v.relevance} | {v.specificity} | {v.depth} | "
            f"{'✅' if low else '❌'} |"
        )
    lines += [
        "",
        f"校准结论：{ok}/{len(results)} 个负面样本被正确压到低分段。"
        "若大量负面样本拿高分，说明 judge 宽松失效，rubric 需要收紧锚点。",
        "",
    ]
    report_path = os.path.join(EVAL_DIR, "report_judge.md")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[i] calibration appended to {report_path}")


def run_scripted_interview(answers: list) -> list:
    """用脚本化回答驱动一整场面试；返回每一轮的记录
    （问题文本、decision、话题，以及面试官当时看到的历史）。"""
    graph = agent.build_turn_graph()
    state = agent.init_state(agent.DEFAULT_TOPICS)
    pool = itertools.cycle(answers)
    rounds = []
    while not state["finished"]:
        ans = next(pool)
        state["messages"] = state["messages"] + [HumanMessage(content=ans)]
        state["output"] = ""
        state["decision"] = {}
        result = graph.invoke(state)
        state = dict(result)
        if state["finished"]:
            break
        rounds.append(
            {
                "question": state["output"],
                "decision": state["decision"],
                "topic": (
                    state["topics"][state["current_topic"] - 1]
                    if state["current_topic"]
                    else ""
                ),
                # 生成这个问题时，面试官所看到的历史
                "history": agent._history_text(state["messages"][:-1], last_n=6),
            }
        )
    return rounds


def evaluate_scenario(name: str, answers: list) -> list:
    """跑一个场景并评判每一个问题；返回评判记录。"""
    print(f"\n=== scenario: {name} ===")
    rounds = run_scripted_interview(answers)
    print(f"[i] {len(rounds)} interviewer questions collected")
    results = []
    for i, r in enumerate(rounds, 1):
        t0 = time.perf_counter()
        try:
            v = judge_question(r["topic"], r["history"], r["question"])
        except Exception as e:
            print(f"[!] judge failed on round {i}: {e}")
            continue
        results.append({**r, "verdict": v})
        print(f"  [{i}/{len(rounds)}] rel={v.relevance} spe={v.specificity} "
              f"dep={v.depth} ({time.perf_counter() - t0:.1f}s)")
    return results


def write_report(all_results: dict) -> None:
    """写出 markdown 报告 + 一份决策路由标注模板。"""
    lines = [
        "# 对话质量评估报告（LLM-as-judge）",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')}；judge 后端与生成端异源",
        "",
        "## 汇总",
        "",
        "| 场景 | 提问数 | relevance 均分/通过率 | specificity 均分/通过率 | depth 均分/通过率 |",
        "|---|---|---|---|---|",
    ]
    for name, results in all_results.items():
        s = summarize([r["verdict"] for r in results])
        lines.append(
            f"| {name} | {s.get('n', 0)} | {s.get('relevance_mean')}/{s.get('relevance_pass')} "
            f"| {s.get('specificity_mean')}/{s.get('specificity_pass')} "
            f"| {s.get('depth_mean')}/{s.get('depth_pass')} |"
        )
    lines += ["", "## 明细", ""]
    for name, results in all_results.items():
        lines.append(f"### 场景 {name}")
        lines.append("")
        for i, r in enumerate(results, 1):
            v = r["verdict"]
            lines.append(
                f"**Q{i}** rel={v.relevance} spe={v.specificity} dep={v.depth}\n\n"
                f"- 话题：{r['topic']}\n- 问题：{r['question']}\n- 理由：{v.reason}"
            )
        lines.append("")
    report_path = os.path.join(EVAL_DIR, "report_judge.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[i] report written to {report_path}")

    # 供人工标注使用的决策路由标注模板
    ann = [
        "# 决策路由人工标注表",
        "",
        "> 标注规则：decision 正确填 1，错误填 0（对照话题清单和对话语境判断"
        "该追问/换话题/结束是否正确）。",
        "",
    ]
    for name, results in all_results.items():
        ann.append(f"## 场景 {name}")
        ann.append("")
        for i, r in enumerate(results, 1):
            d = r["decision"]
            ann.append(
                f"- [ ] 轮 {i}: action={d.get('action')} topic={d.get('topic_index')} "
                f"| 语境: {r['history'][-80:]!r} | 问题: {r['question'][:60]!r}"
            )
        ann.append("")
    ann_path = os.path.join(EVAL_DIR, "route_annotations.md")
    with open(ann_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ann))
    print(f"[i] annotation template written to {ann_path}")


def main():
    all_results = {}
    for name, answers in SCENARIOS:
        all_results[name] = evaluate_scenario(name, answers)
    write_report(all_results)
    calibration = calibrate_judge()
    write_calibration(calibration)
    print("\nEvaluation complete. Next: label route_annotations.md and spot-score 5-8 questions for judge-human agreement.")


if __name__ == "__main__":
    main()
