"""历史面试成绩的图表（能力雷达图）。

和 history.py 一样，本模块不碰 torch / langchain：它随页面渲染，
 heavyweight 的导入会把首屏拖慢（那是 H1 修过的老毛病）。
plotly 也是在函数内部才导入——没有历史记录时页面根本不该为它付出代价。
"""
from history import SCORE_DIMENSIONS


def _scores(entry: dict) -> list:
    """按 SCORE_DIMENSIONS 的顺序取出五维分数；缺失或非数值记 0。"""
    out = []
    for key, _ in SCORE_DIMENSIONS:
        value = entry.get(key)
        out.append(value if isinstance(value, (int, float)) else 0)
    return out


def ability_radar(entries: list, index: int):
    """把第 `index` 场面试的五维分数画成雷达图。

    有其他场次时额外叠一条虚线：**除本场之外**的历史平均。
    不含本场是刻意的——只跑过两场时，把本场算进平均值会让参照线被
    本场自己拉偏，"这场比平时如何"就看不出来了。
    """
    import plotly.graph_objects as go  # 见模块 docstring：只在真要画图时导入

    labels = [label for _, label in SCORE_DIMENSIONS]
    # 雷达图需要闭合，所以首尾各补一个点
    theta = labels + [labels[0]]
    current = _scores(entries[index])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=current + [current[0]],
            theta=theta,
            name=f'本场（{entries[index].get("time", "未知时间")}）',
            fill='toself',
        )
    )

    others = [_scores(e) for i, e in enumerate(entries) if i != index]
    if others:
        average = [round(sum(col) / len(others), 2) for col in zip(*others, strict=True)]
        fig.add_trace(
            go.Scatterpolar(
                r=average + [average[0]],
                theta=theta,
                name=f'历史平均（其他 {len(others)} 场）',
                line={'dash': 'dash'},
            )
        )

    fig.update_layout(
        polar={'radialaxis': {'range': [0, 10], 'dtick': 2}},
        showlegend=True,
        height=420,
        margin={'l': 60, 'r': 60, 't': 40, 'b': 40},
    )
    return fig
