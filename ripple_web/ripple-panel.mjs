// Restore the original interactive graph on the integrated homepage.
// Historical association is independent of the live quote/model state.
import {el, ASSETS, hideTip} from './format.mjs?v=en-r1';
import {tr} from './i18n.mjs?v=en-r1';
import {contagionTitle, chainSentences, renderContagion} from './blocks/contagion.mjs?v=en-r1';

const $ = id => document.getElementById(id);
let graph = null, source = 'BTC', entered = false, loading = false;
function draw(asset = source) {
  if (!graph || !ASSETS.includes(asset)) return;
  source = asset; hideTip();
  $('shock-ripple-source').value = source;
  $('shock-ripple-summary').textContent = contagionTitle(graph, source);
  renderContagion($('shock-ripple-map'), graph, source, draw);
}
async function load() {
  if (loading) return;
  loading = true;
  try {
    const response = await fetch('data/graph.json', {cache:'no-cache'});
    if (!response.ok) throw new Error('graph unavailable');
    const next = await response.json();
    if (!Array.isArray(next.edges) || !Array.isArray(next.strongest)) throw new Error('graph invalid');
    graph = next;
    $('shock-ripple').querySelector('.eyebrow').textContent = tr(`历史冲击 · 2026年1月 · ${graph.event_count}次`, `Historical shocks · January 2026 · ${graph.event_count} events`);
    $('shock-ripple-source').replaceChildren(...ASSETS.map(asset => {
      const option = el('option', null, asset); option.value = asset; return option;
    }));
    $('shock-ripple-chains').replaceChildren(...chainSentences(graph).map(text => el('li', null, text)));
    $('shock-ripple-source').disabled = false;
    $('shock-ripple-replay').disabled = false;
    $('shock-ripple').dataset.loadState = 'ready';
    draw();
  } catch (_) {
    $('shock-ripple').dataset.loadState = 'error';
    $('shock-ripple-summary').textContent = tr('冲击图暂时无法读取。', 'The shock map is unavailable.');
    const button = el('button', 'text-button', tr('重新读取冲击图', 'Reload shock map')); button.type = 'button';
    button.addEventListener('click', load); $('shock-ripple-map').replaceChildren(button);
  } finally { loading = false; }
}
$('shock-ripple-source').addEventListener('change', event => draw(event.target.value));
$('shock-ripple-replay').addEventListener('click', () => draw());
new IntersectionObserver(entries => {
  if (!entered && entries[0].isIntersecting && graph) { entered = true; draw(); }
}, {threshold:.25}).observe($('shock-ripple-map'));
load();
