"""Render two concise supplements from saved deterministic results only."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]
LABELS = {"kronos_base": "Kronos", "no_propagation": "不传播", "btc_beta": "BTC-beta", "historical_30": "过去30分钟"}


def num(x):
    return "null" if x is None else f"{x:.2f}"


def pct(x):
    return "null" if x is None else f"{x:.2%}"


def generate(r):
    direction_groups = r["trigger_direction"]["groups"]
    first = ["# B 线增补一：方向分组、下行波动、期末方向", "",
             "同一批 84 事件、十资产、冻结单路径；Kronos 重跑 **0 次**。误差单位 bps，越低越好；Top3 为期望召回，越高越好。", "",
             "**1．涨冲击与跌冲击**（表内依次为：波动 Top3 / 波动 MAE / 回撤 MAE）", "",
             f"| 方法 | 涨冲击 n={direction_groups['up']['event_count']} | 跌冲击 n={direction_groups['down']['event_count']} |",
             "|---|---:|---:|"]
    for method in METHODS:
        cells = []
        for name in ("up", "down"):
            s = direction_groups[name]["methods"][method]
            cells.append(f"{pct(s['volatility_top3_recall_expected'])} / {num(s['volatility_mae_bps'])} / {num(s['max_drawdown_mae_bps'])}")
        first.append(f"| {LABELS[method]} | {' | '.join(cells)} |")
    contributions = r["trigger_direction"]["mdd_advantage_decomposition"]
    lines = []
    for method in ("historical_30", "btc_beta", "no_propagation"):
        g = contributions[method]["groups"]
        lines.append(f"相对{LABELS[method]} {num(g['up']['contribution_to_all_event_advantage_bps'])} / {num(g['down']['contribution_to_all_event_advantage_bps'])}")
    first += ["", "混合／未知方向 n=0，指标 null；84 次全部保留。回撤优势对总体的贡献（涨 / 跌，bps）：" + "；".join(lines) + "。因此，相对历史基线约95%的改善来自跌冲击，相对 BTC-beta 则约61%来自涨冲击，不能脱离比较对象概括。", "",
              "**2．下行波动**（全部 n=84）", "", "| 方法 | 下行 Top3 | 下行 MAE |", "|---|---:|---:|"]
    for method in METHODS:
        s = r["downside"]["aggregate"]["methods"][method]
        first.append(f"| {LABELS[method]} | {pct(s['top3_recall_expected'])} | {num(s['mae_bps'])} |")
    first += ["", "口径：下行半方差 = Σmin(r,0)²/30，下行波动 = √半方差 × √30；r 为分钟对数收益，以零为界、不去均值、不只按负收益数量除。四方法有效 n=84，null=0。", "",
              "**3．30分钟期末方向**（全十资产，不排除起源资产）", "",
              "| 方法 | 正确 / 非零真值 | 命中率 |", "|---|---:|---:|"]
    d = r["terminal_direction"]["aggregate"]
    for name, label in (("kronos_base", "Kronos"), ("random", "50% 随机（理论期望）"),
                        ("always_continue", "始终延续冲击方向"), ("always_reverse", "始终反转冲击方向")):
        if name == "random":
            first.append(f"| {label} | 不做随机抽样 | {pct(d['random_binary_expected_accuracy'])} |")
        else:
            s = d["methods"][name]
            first.append(f"| {label} | {s['common_correct']} / {s['common_pairs']} | {pct(s['common_binary_accuracy'])} |")
    model = d["methods"]["kronos_base"]
    first += ["", f"共 {d['all_pairs']} 个资产—事件对；真值不涨不跌 {d['true_flat_pairs']} 对，单列后共同二分类分母 {d['common_pairs']}。Kronos 预测平盘 {model['common_predicted_flat']} 对；全三符号准确率 {model['all_three_sign_correct']}/{model['all_three_sign_pairs']}={pct(model['all_three_sign_correct']/model['all_three_sign_pairs'])}。这些相关数据不是840次独立试验。", "",
              "**一句话结论：回撤优势来自哪组取决于比较基线；下行波动仍未胜历史基线，期末方向五五开，结论为“无方向预测能力（本样本未证实）”，不进 pitch。**", "",
              "风险指标沿用原评分资产与事件等权口径；单月探索，无统计显著性声明。[完整数值与来源](results-v1.json) · [冻结口径](CONTRACT.md)"]

    second = ["# B 线增补二：预警半径与分组", "",
              "使用同一批冻结预测；无重跑、无删样本。误差为 MAE（bps↓），Top3 为期望召回（↑）。", "",
              "**1．三个10分钟区间**（各 n=84；H = 过去30分钟重放基线）", "",
              "| 预测区间 | 回撤：Kronos | 回撤：H | 波动：Kronos | 波动：H |", "|---|---:|---:|---:|---:|"]
    for s in r["radius"]["aggregate"]["segments"]:
        k, h = s["methods"]["kronos_base"], s["methods"]["historical_30"]
        second.append(f"| {s['start_minute']}–{s['end_minute']} 分钟 | {num(k['max_drawdown']['mae_bps'])} | {num(h['max_drawdown']['mae_bps'])} | {num(k['volatility']['mae_bps'])} | {num(h['volatility']['mae_bps'])} |")
    radii = r["radius"]["aggregate"]["advantage_radius_minutes"]
    assert radii == {"max_drawdown": None, "volatility": 10, "joint": None}, "Update interpretation rather than reusing an obsolete conclusion"
    second += ["", "**半径结论：按本轮比较，波动误差的连续优势半径是10分钟；回撤半径=null，回撤与波动的共同优势半径=null。** 20–30分钟虽两项均胜，但前段并未连续胜出，因此不能称为30分钟连续优势。", "",
               "每段使用自身路径的段首价并重置回撤峰值，波动为该段10个收益的总体标准差×√10；基线同样切既有30分钟重放路径。半径要求从第0分钟起连续严格胜出，因此10分钟只适用于本样本的波动误差比较，受基线重放对齐影响，不能称为已验证的稳定预警能力。", "",
               "**2．完整分组**（每格：回撤 MAE / 波动 Top3；n = MAE有效事件 / Top3有效事件）", "",
               "| 组别 | n | Kronos | 不传播 | BTC-beta | 过去30分钟 |", "|---|---:|---:|---:|---:|---:|"]
    for axis, group, label in (("tier", "big", "大币"), ("tier", "small", "小币"),
                               ("strength", "6_to_8", "6≤强度<8σ"), ("strength", "ge_8", "强度≥8σ"),
                               ("systemic", "systemic", "系统性"), ("systemic", "non_systemic", "非系统性")):
        g = r["groups"]["summaries"][axis][group]
        cells = [f"{num(g['methods'][m]['max_drawdown_mae_bps'])} / {pct(g['methods'][m]['volatility_top3_recall_expected'])}" for m in METHODS]
        second.append(f"| {label} | {g['mae_event_count']} / {g['top3_defined_count']} | {' | '.join(cells)} |")
    big, small = [r["groups"]["summaries"]["tier"][x] for x in ("big", "small")]
    second += ["", f"大币=BTC/ETH/SOL/BNB，小币=其余六币。沿用系统性全资产、非系统性排除并列起源资产的口径；大币有 {big['top3_null_count']} 事件不足3个资产，Top3=null，MAE仍保留。其余组无空值。大／小币Top3随机名额期望分别为 {pct(big['chance_recall_mean'])}/{pct(small['chance_recall_mean'])}，不能直接比较两组召回率高低。", "",
               "非系统性保留同时出现的多个最早源头，不捏造唯一源头；系统性标签为事后分类。三个分组维度重叠，事件数不相加。", "",
               "**一句话结论：原版的回撤幅度误差在多个组较低，但所有列出的组均未胜过历史基线的波动Top3；demo最多陈述“本样本前10分钟波动误差占优”，不能宣传通用优势半径。**", "",
               "单月、单路径探索；全部组与null已保留。[完整数值与来源](results-v1.json) · [冻结口径](CONTRACT.md)"]
    return {"SUPPLEMENT_1.md": "\n".join(first) + "\n", "SUPPLEMENT_2.md": "\n".join(second) + "\n"}


if __name__ == "__main__":
    result = json.loads((HERE / "results-v1.json").read_text())
    output = generate(result)
    if any((HERE / name).exists() for name in output):
        raise FileExistsError("Do not overwrite existing reports")
    for name, content in output.items():
        with (HERE / name).open("x") as f:
            f.write(content)
    print(json.dumps({"reports": list(output), "model_calls": 0}, ensure_ascii=False))
