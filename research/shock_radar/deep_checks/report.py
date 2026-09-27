"""Render the compact honesty report from completed frozen arithmetic results."""
import json
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    loaded = {name: json.loads((HERE / (name + '.json')).read_text())
              for name in ('coverage', 'permutation', 'failures', 'clustering')}
    c, p, f, k = (loaded[n] for n in ('coverage', 'permutation', 'failures', 'clustering'))
    assert c['pooled']['expected'] == 30 and c['pooled']['invalid'] == 0
    assert p['replicate_count'] == 200 and f['population']['validated_events'] == 140
    assert k['summary']['rerun_count'] == 84
    cov = c['pooled']; tier = c['by_tier']; kt = k['summary']; ks = kt['singleton']
    e = p['tests']['nonzero_directed_pair_count']; delay = p['tests']['observation_mean_delay_minutes']
    unchanged = sum(r['ari'] == 1 for r in k['leave_one_out'])
    text = f'''# B 线深挖：四个诚实检查

范围：01月84事件＋02月56事件的冻结结果；新增推理0、下载0，未读取03，未改A线或线上页面。

| 检查 | 核心数字 | 一句话诚实结论 |
|---|---|---|
| 不确定性带覆盖率 | **{cov['covered']}/{cov['expected']}＝{cov['coverage_valid']:.1%}**；大币{tier['major']['covered']}/{tier['major']['expected']}、小币{tier['small']['covered']}/{tier['small']['expected']}；{cov['above']}次高于上界、{cov['below']}次低于下界 | **区间过度自信，demo必须标注“未校准的模型采样区间”**，不能称90%置信区间。 |
| 图谱随机化 | 真实/零分布中位数：**68/23条边，3.01/15.63分钟**；两项校正p均**{e['holm_adjusted_p']:.5f}** | 时间聚集超过本次零模型；**不证明因果传染、单条边稳定或预测价值**。 |
| 最差10次回撤预测 | 平均误差**{f['top10']['mean_event_mae_bps']:.2f} bps**；跌冲击9/10、系统性8/10；**98/98**被评分资产均低估回撤 | 这批最差失败集中在跌冲击、系统性事件，并低估损失；是事后描述，不能直接充当预警规则。 |
| n=1类别删一检查 | 单例仍孤立**{ks['isolated_retained_count']}/{ks['retention_trials']}＝{ks['persistence']:.1%}**；{unchanged}/84分组不变；ARI中位{kt['ari']['median']:.2f}、最低{kt['ari']['min']:.4f} | 保留冻结四类，单例标为**“n=1孤立观察，缺少重复样本支持”**；局部分组仍会变化。 |

最差10次的背景对照：全140事件中跌冲击47.9%、系统性26.4%，平均误差42.38 bps。8σ+占最差10次的40%，背景24.3%；不能单独断言强度决定失败。小币/大币平均误差228.71/171.73 bps；但最差资产来自小币的比例70%，与全样本70.7%相近，**没有额外的小币集中证据**。

| 最差序位／UTC时间（2026） | σ | 方向 | 系统性 | 回撤MAE | 大币MAE | 小币MAE | 最差资产 |
|---|---:|---|---|---:|---:|---:|---|
'''
    for number, row in enumerate(f['top10_rows'], 1):
        stamp = row['timestamp'][5:16].replace('T', ' ')
        text += (f"| {number}. {stamp} | {row['strength_sigma']:.2f} | {'跌' if row['direction']=='down' else '涨'} | "
                 f"{'是' if row['systemic'] else '否'} | {row['mae_bps']:.2f} | {row['major_mae_bps']:.2f} | "
                 f"{row['small_mae_bps']:.2f} | {row['worst_asset']} |\n")
    text += '''
口径与限制：误差单位bps，100 bps＝1个百分点；大币为BTC/ETH/SOL/BNB，按各档资产均值比较。覆盖率为闭区间[p05,p95]，三个预选事件各2/10、4/10、4/10，30对不是独立样本；每资产仅10次采样，未拟合新校准。

图谱零模型固定200次、seed20260927：各资产保留触发序列独立循环平移，再按原规则合并事件/建图；两项单侧原始p均1/201，Holm校正。零模型未保留共同市场时段，合格事件数190–212（真实84），因此只检验跨资产时间对齐。删一仍固定原距离/average-linkage/k=4；删除单例自身不计入83次保留检验。

证据：[逐项结果与核验](VERIFICATION.md) · [最差10次CSV](top10.csv)。完成后停止。
'''
    with (HERE / 'REPORT.md').open('x') as stream:
        stream.write(text)
    hashes = {name + '.json': sha256((HERE / (name + '.json')).read_bytes()).hexdigest() for name in loaded}
    with (HERE / 'report_sources.json').open('x') as stream:
        json.dump({'input_sha256': hashes, 'renderer_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                   'report_sha256': sha256(text.encode()).hexdigest(), 'model_calls': 0}, stream, indent=2)
        stream.write('\n')
    print('REPORT.md written from four completed result files')


if __name__ == '__main__':
    main()
