import {el} from '../format.mjs?v=en-r1';
import {tr} from '../i18n.mjs?v=en-r1';

export function renderAttentionTime(host, duration, asset) {
  host.replaceChildren();
  host.append(el('h3', null, tr('历史回落参考', 'Historical recovery reference')), el('p', 'minor', tr('历史统计，不是预测', 'Historical statistics, not a forecast')));
  if (!duration?.groups?.small || !duration?.groups?.major) {
    host.append(el('p', 'minor', tr('历史时间参考暂不可用。', 'Historical timing unavailable.'))); return false;
  }
  const groups = ['small', 'major'].map(key => ({key, ...duration.groups[key]}));
  if (groups.some(group => !Number.isFinite(group.reference_minute) || group.reference_minute < 0)) return false;
  const scale = Math.max(60, ...groups.map(group => group.reference_minute));
  for (const group of groups) {
    const selected = group.assets.includes(asset);
    const row = el('div', `attention-time-row${selected ? ' selected' : ''}`);
    row.append(el('span', 'attention-time-label', `${group.key === 'small' ? tr('小币组', 'Smaller assets') : tr('大币组', 'Major assets')}${selected ? tr(' · 当前资产所属', ' · selected asset’s group') : ''}`));
    const track = el('span', 'attention-time-track'); track.setAttribute('aria-hidden', 'true');
    const fill = el('i'); fill.style.width = `${group.reference_minute / scale * 100}%`; track.append(fill);
    row.append(track, el('span', 'attention-time-number', tr(`${group.reference_minute} 分钟`, `${group.reference_minute} min`))); host.append(row);
  }
  const counts = [...new Set(groups.map(group => group.n))].join(' / ');
  host.append(el('p', 'minor historical-reference-method', tr(
    `${counts} 次历史冲击中位曲线 · 连续 ${duration.confirmation_points} 分钟 ≤${duration.threshold}× 基线`,
    `Median curve across ${counts} historical shocks · ${duration.confirmation_points} consecutive minutes ≤${duration.threshold}× baseline`
  )));
  return true;
}
