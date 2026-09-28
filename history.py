"""面试历史记录（reports/history.json）的读写。

有意做成独立模块、且只依赖标准库：Streamlit 页面一打开就要读它，
如果把它挂在 agent.py 上，首屏又会被 torch / sentence-transformers
的加载阻塞 1-2 分钟（那是 H1 修过的老毛病）。

每条记录形如：
    {
        "time": "2026-09-29 20:15",   # 面试结束时间
        "job": "默认",                 # 岗位标签，为多岗位切换预留
        "rounds": 6,                  # 面试官轮数
        "topics": [...],              # 本场抽取的话题锚点
        "report": "interview_report_20260929_2015.md",  # reports/ 下的报告文件
        "technical_depth": 7,         # 以下五项来自复盘的 InterviewScore
        "communication": 6,
        "project_experience": 8,
        "job_fit": 7,
        "star_completeness": 5,
        "summary": "...",
        "strength": "...",
        "improvement": "..."
    }
"""
import json
import logging
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
HISTORY_FILE = os.path.join(REPORTS_DIR, 'history.json')
DEFAULT_JOB = '默认'

# 复盘的五个评分维度：(字段名, 中文标签)。
# 放在这里而不是 agent.py，是因为历史记录存的就是这些字段，
# 标签跟着数据走，UI 才能只 import 本模块就画出表格
# ——不必为了六个中文字加载 torch。
SCORE_DIMENSIONS = [
    ('technical_depth', '技术深度'),
    ('communication', '表达逻辑'),
    ('project_experience', '项目经验'),
    ('job_fit', '岗位匹配'),
    ('star_completeness', '回答完整度'),
]

logger = logging.getLogger(__name__)


def load_history() -> list:
    """读取全部历史记录，按写入顺序返回；文件缺失或损坏时返回空列表。

    读取失败不能让页面崩掉——历史记录是附加价值，
    最坏情况也就是看起来「还没练过」。
    """
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning('history file unreadable (%s), treating as empty', e)
        return []
    if not isinstance(data, list):
        logger.warning('history file is not a list, treating as empty')
        return []
    return [e for e in data if isinstance(e, dict)]


def record_history(entry: dict) -> None:
    """把一场面试的成绩追加到 reports/history.json。

    写入失败只记日志、不抛异常：面试已经跑完了，不能因为存历史失败
    把收尾流程弄崩（调用方在 graph 节点里）。
    """
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        history = load_history()
        history.append(entry)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning('failed to record interview history (%s)', e)
