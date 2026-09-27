import { el } from '../format.mjs?v=ui-r2';
import { tierForAsset, durationStatus } from '../duration-state.mjs?v=duration-r1';

// Reference times describe historical tier median curves; they never expire attention.
export function renderDuration(host, data, { asset, eventId, minute }) {
  const wasOpen = host.querySelector('details')?.open || false;
  const hadFocus = host.contains(document.activeElement);
  host.replaceChildren();
  const tier = tierForAsset(data, asset);
  const groups = data?.groups;
  const validGroup = (g) => g && typeof g.label === 'string' && Number.isInteger(g.reference_minute)
    && g.reference_minute >= 0 && g.reference_minute <= 120 && Number.isInteger(g.n) && g.n > 0
    && Number.isInteger(g.n_expected) && g.n_expected >= g.n && Number.isInteger(g.unconfirmed_after_120)
    && g.unconfirmed_after_120 >= 0 && g.unconfirmed_after_120 <= g.n
    && Number.isInteger(g.event_median_minute) && g.event_median_minute >= 0 && g.event_median_minute <= 120
    && ['2026-01','2026-02'].every((m) => Number.isInteger(g.monthly_reference?.[m]) && g.monthly_reference[m] >= 0 && g.monthly_reference[m] <= 120);
  if (!tier || !validGroup(groups?.major) || !validGroup(groups?.small) || data.threshold !== 1.2 || data.confirmation_points !== 10) {
    host.append(el('p', 'minor', '历史观察时长暂不可用；价格回放仍可继续。')); return false;
  }
  const current = groups[tier];
  const top = el('div', 'duration-heading');
  const title = el('h3', null, '还要看多久？');
  top.append(title, el('p', 'minor', '参考历史节奏，再检查这次是否回落。'));
  host.append(top);
  const refs = el('div', 'duration-references');
  for (const key of ['small', 'major']) {
    const g = groups[key];
    const row = el('div', `duration-reference${key === tier ? ' selected' : ''}`);
    row.append(el('span', 'duration-group', `${g.label}${key === tier ? ' · 当前资产所属' : ''}`));
    const track = el('div', 'duration-track'); track.setAttribute('aria-hidden', 'true');
    const mark = el('i', 'duration-mark'); mark.style.left = `${g.reference_minute / 120 * 100}%`; track.append(mark);
    if (key === tier) { const now = el('i', 'duration-now'); now.style.left = `${minute / 120 * 100}%`; track.append(now); }
    row.append(track, el('span', 'duration-value', `${g.reference_minute} 分钟`)); refs.append(row);
  }
  const axis = el('div', 'duration-axis'); axis.setAttribute('aria-hidden', 'true');
  const scale = el('div'); for (const n of [0,30,60,90,120]) scale.append(el('span', null, `${n}`));
  axis.append(el('span'), scale, el('span', null, '分钟')); refs.append(axis); host.append(refs);
  host.append(el('p', 'minor duration-key', '圆点：历史中位曲线确认回落时点；短线：本次回放已揭示时间。'));
  const status = durationStatus(data, eventId, tier, minute);
  const observation = el('div', 'duration-observation'); observation.dataset.tier = tier;
  if (status.available) {
    const condition = status.confirmed ? '已连续 10 个分钟点在基线附近'
      : status.ratio > data.threshold ? (status.hadConfirmed ? '再次高于参考线' : '仍高于参考线') : '已到基线附近，继续确认';
    const head = el('div', 'duration-statusline');
    head.append(el('span', null, `${current.label}本次波动 · ${minute ? `+${minute} 分钟` : '冲击时刻'}`), el('strong', null, `${status.ratio.toFixed(2)} × 基线`));
    observation.append(head, el('p', 'duration-condition', condition));
    const progress = el('div', 'duration-confirmation');
    const marks = el('span', 'duration-points'); marks.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 10; i++) marks.append(el('i', i < Math.min(status.streak, 10) ? 'met' : ''));
    progress.append(marks, el('span', null, `连续满足 ${Math.min(status.streak, 10)} / 10 点`)); observation.append(progress);
    observation.dataset.streak = String(status.streak); observation.dataset.confirmed = String(status.confirmed);
  } else observation.append(el('p', null, '本次分组波动资料不足，暂不能判断回落状态。'));
  observation.append(el('p', 'minor', '回落条件：分组波动不高于基线的 1.2 倍，连续 10 个分钟点。达到参考时间不会自动收起资产。'));
  host.append(observation);
  const details = el('details', 'duration-details');
  details.open = wasOpen;
  details.append(el('summary', null, '这些时长从哪里来？'));
  details.append(el('p', 'minor', `2026 年 1–2 月，${current.n}/${current.n_expected} 次冲击有完整衰减窗口。${groups.small.reference_minute}/${groups.major.reference_minute} 分钟对应先按事件内取分组中位数、再跨事件取中位数的曲线；不表示某个资产到时恢复，也不是单次事件恢复时间的中位数。`));
  const table = el('table'); const head = el('thead'); const hr = el('tr');
  for (const t of ['历史口径','小币组','大币组']) { const th = el('th', null, t); th.scope = 'col'; hr.append(th); } head.append(hr); table.append(head);
  const body = el('tbody');
  const rows = [
    ['合并中位曲线确认点', ...['small','major'].map((k) => `${groups[k].reference_minute} 分钟`)],
    ['1 月 / 2 月曲线确认点', ...['small','major'].map((k) => `${groups[k].monthly_reference['2026-01']} / ${groups[k].monthly_reference['2026-02']} 分钟`)],
    ['单事件恢复时间中位数', ...['small','major'].map((k) => `${groups[k].event_median_minute} 分钟`)],
    ['超过 120 分钟仍未确认', ...['small','major'].map((k) => `${groups[k].unconfirmed_after_120} / ${groups[k].n}`)],
  ];
  for (const cells of rows) { const tr = el('tr'); cells.forEach((v,i) => { const cell=el(i===0?'th':'td',null,v); if(i===0) cell.scope='row'; tr.append(cell); }); body.append(tr); }
  table.append(body); details.append(table);
  details.append(el('p', 'minor', '每资产用 10 分钟滚动波动除以冲击前基线，再取组内中位数。基线使用截至冲击前 5 分钟的 120 个收益；后续新冲击保留，超过 120 分钟的样本没有当作 120 分钟恢复。'));
  details.append(el('p', 'minor', '这份已知历史统计包含当前回放案例，属于回顾性参考。价格回放仍只到 +30 分钟；波动回落不等于价格反弹、损失恢复或其他风险消失。'));
  host.append(details);
  if (hadFocus) details.querySelector('summary').focus({ preventScroll: true });
  return true;
}
