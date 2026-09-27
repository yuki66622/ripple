import test from 'node:test';
import assert from 'node:assert/strict';
import {createLiveClient} from './live-client.mjs';

const epoch = Date.parse('2026-09-27T13:00:00Z');
const market = (minute = 0, extra = {}) => ({window_id:`window-${minute}`,
  as_of:new Date(epoch + minute * 60_000).toISOString(), age_seconds:0, stale:false, error:null, ...extra});
const state = (status, minute = 0) => ({status, window_id:`window-${minute}`, as_of:market(minute).as_of, error:status === 'error' ? 'Rejected sample' : null});
const response = (body, status = 200) => ({ok:status < 400, status, json:async () => body});
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; };
function harness(overrides = {}) {
  const timers = new Map(), events = {markets:[], predictions:[], errors:[], calls:[]};
  let nextTimer = 0;
  const env = {time:epoch, hidden:false, market:market(), status:state('running'),
    post:async id => response({...state('running'), window_id:id}), ...overrides};
  const client = createLiveClient({
    onMarket:value => events.markets.push(value),
    onPrediction:value => events.predictions.push(value),
    onError:(message, detail) => events.errors.push({message, ...detail}),
  }, {
    fetch:async (url, options) => {
      events.calls.push({url, options});
      if (url === '/api/live') return env.getMarket ? env.getMarket() : response(env.market);
      if (url === '/api/prediction') return env.getStatus ? env.getStatus() : response(env.status);
      if (url === '/api/predict') return env.post(JSON.parse(options.body).window_id);
      throw new Error(`Unexpected URL ${url}`);
    },
    setInterval:(callback, ms) => { const id = ++nextTimer; timers.set(id, {callback, ms}); return id; },
    clearInterval:id => timers.delete(id), now:() => env.time, isHidden:() => env.hidden,
    timeoutSignal:() => undefined,
  });
  return {client, env, events, timers,
    posts:() => events.calls.filter(call => call.url === '/api/predict').map(call => JSON.parse(call.options.body).window_id),
    tick:async ms => { for (const [id, timer] of [...timers]) if (timer.ms === ms && timers.has(id)) await timer.callback(); },
    next:(minute, extra) => { env.time = epoch + minute * 60_000; env.market = market(minute, extra); },
  };
}

test('automatically submits one fresh window; asset/manual/repeated refreshes never resample it', async () => {
  const h = harness();
  await h.client.start();
  assert.deepEqual(h.posts(), ['window-0']);
  h.env.status = state('ready');
  await h.tick(1_500);
  await h.client.refresh();
  await h.client.predict('window-0');
  await h.client.start();
  assert.deepEqual(h.posts(), ['window-0']);
  assert.equal(h.events.predictions.at(-1).status, 'ready');
  assert.equal([...h.timers.values()].filter(t => t.ms === 300_000).length, 1);
  assert.equal([...h.timers.values()].filter(t => t.ms === 1_500).length, 0);
});

test('serializes jobs and coalesces all intermediate windows into the newest window', async () => {
  const h = harness(); await h.client.start();
  h.next(1); await h.client.refresh();
  h.next(2); await h.client.refresh();
  assert.deepEqual(h.posts(), ['window-0']);
  h.env.status = state('ready'); await h.tick(1_500);
  assert.deepEqual(h.posts(), ['window-0', 'window-2']);
  assert.equal(h.events.predictions.at(-1).window_id, 'window-2');
  h.env.status = state('error', 2); await h.tick(1_500);
  await h.client.refresh(); await h.client.predict('window-2');
  assert.deepEqual(h.posts(), ['window-0', 'window-2'], 'A rejected model sample is frozen too.');
});

test('new market arriving during POST cannot create a concurrent submission', async () => {
  const pending = deferred();
  const h = harness({post:async () => pending.promise});
  const starting = h.client.start();
  await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
  h.next(1); await h.client.refresh();
  assert.deepEqual(h.posts(), ['window-0']);
  pending.resolve(response(state('running'))); await starting;
  h.env.post = async id => response({...state('running', 1), window_id:id});
  h.env.status = state('ready'); await h.tick(1_500);
  assert.deepEqual(h.posts(), ['window-0', 'window-1']);
});

test('stop/history suspends pending jobs but still receives already running completion', async () => {
  const h = harness(); await h.client.start();
  h.next(1); await h.client.refresh(); h.client.stop();
  h.env.status = state('ready'); await h.tick(1_500);
  await h.client.predict('window-1');
  assert.deepEqual(h.posts(), ['window-0']);
  assert.equal(h.events.predictions.at(-1).status, 'ready');
  assert.equal(h.timers.size, 0);
  h.next(2); await h.client.start();
  assert.deepEqual(h.posts(), ['window-0', 'window-2']);
});

test('hidden pages neither fetch automatically nor start another job; visible refresh resumes latest', async () => {
  const h = harness({hidden:true}); await h.client.start(); await h.tick(300_000);
  assert.equal(h.events.calls.length, 0);
  h.env.hidden = false; await h.tick(300_000);
  h.next(1); await h.client.refresh(); h.env.hidden = true;
  h.env.status = state('ready'); await h.tick(1_500); await h.tick(300_000);
  assert.deepEqual(h.posts(), ['window-0']);
  h.env.hidden = false; h.next(2); await h.tick(300_000);
  assert.deepEqual(h.posts(), ['window-0', 'window-2']);
});

test('stale/error/aged snapshots never auto-submit, including a queued window that ages during inference', async () => {
  for (const extra of [{stale:true}, {error:'feed failed'}, {age_seconds:181}, {as_of:'invalid'}]) {
    const h = harness({market:market(0, extra)}); await h.client.start();
    assert.deepEqual(h.posts(), []);
  }
  const h = harness(); await h.client.start();
  h.next(1); await h.client.refresh(); h.env.time += 181_000;
  h.env.status = state('ready'); await h.tick(1_500);
  assert.deepEqual(h.posts(), ['window-0']);
});

test('failed feed clears the queued candidate and reports market scope, without corrupting running prediction', async () => {
  const h = harness(); await h.client.start();
  h.next(1); await h.client.refresh();
  h.env.getMarket = async () => { throw new Error('Feed unavailable'); };
  await h.client.refresh();
  h.env.status = state('ready'); await h.tick(1_500);
  assert.deepEqual(h.posts(), ['window-0']);
  assert.deepEqual(h.events.errors.at(-1), {message:'Feed unavailable', scope:'market'});
  delete h.env.getMarket; await h.client.refresh();
  assert.deepEqual(h.posts(), ['window-0', 'window-1']);
});

test('uncertain POST timeout reconciles running result with GET, never repeating the submission', async () => {
  const h = harness({post:async () => { throw new Error('Timeout'); }});
  await h.client.start();
  assert.deepEqual(h.events.predictions.map(p => p.status), ['running', 'error', 'running']);
  assert.equal(h.events.predictions[1].window_id, 'window-0');
  assert.equal(h.events.errors.at(-1).scope, 'prediction');
  assert.equal(h.events.calls.filter(c => c.url === '/api/prediction').length, 1);
  h.env.status = state('ready'); await h.tick(1_500); await h.client.refresh();
  assert.deepEqual(h.posts(), ['window-0']);
});

test('unknown POST and failed reconciliation reserve execution slot until server status is known', async () => {
  const h = harness({post:async () => { throw new Error('Timeout'); },
    getStatus:async () => { throw new Error('Status unavailable'); }});
  await h.client.start();
  h.next(1); await h.client.refresh(); await h.tick(1_500);
  assert.deepEqual(h.posts(), ['window-0']);
  assert.equal(h.events.predictions.at(-1).status, 'error', 'UI receives explicit failure instead of a forever-running placeholder.');
  assert.ok(h.events.errors.every(error => error.scope === 'prediction'));
  delete h.env.getStatus; h.env.status = {status:'idle', error:null};
  h.env.post = async id => response({...state('running', 1), window_id:id});
  await h.tick(1_500);
  assert.deepEqual(h.posts(), ['window-0', 'window-1']);
});

test('server conflicts are reconciled; the failed target is not retried after another running job ends', async () => {
  const h = harness({market:market(1), time:epoch + 60_000,
    post:async () => response({error:'Already running'}, 409), status:state('running', 0)});
  await h.client.start();
  h.env.status = state('ready', 0); await h.tick(1_500); await h.client.refresh();
  assert.deepEqual(h.posts(), ['window-1']);
  assert.equal(h.events.predictions.at(-1).window_id, 'window-1');
  assert.equal(h.events.predictions.at(-1).status, 'error');
  assert.deepEqual(h.events.predictions.map(p => p.status), ['running', 'error'], 'An unrelated running job and its completion must not replace the current failure.');
  assert.equal([...h.timers.values()].filter(timer => timer.ms === 1_500).length, 0);
});

test('idle or old terminal reconciliation preserves current-window submit failure and releases the execution slot', async () => {
  for (const result of [{status:'idle', error:null}, state('ready', 0), state('error', 0)]) {
    const h = harness({market:market(1), time:epoch + 60_000,
      post:async () => { throw new Error('Submission unavailable'); }, status:result});
    await h.client.start(); await h.client.refresh();
    assert.deepEqual(h.events.predictions.map(p => p.status), ['running', 'error']);
    assert.equal(h.events.predictions.at(-1).window_id, 'window-1');
    assert.equal(h.events.predictions.at(-1).error, '本窗口预测连接失败：Submission unavailable');
    assert.deepEqual(h.posts(), ['window-1']);
    h.env.post = async () => response(state('running', 2));
    h.next(2); await h.client.refresh();
    assert.deepEqual(h.posts(), ['window-1', 'window-2']);
    assert.equal(h.events.predictions.at(-1).window_id, 'window-2');
    assert.equal(h.events.predictions.at(-1).status, 'running');
  }
});

test('a matching terminal result can recover an uncertain submission without resampling', async () => {
  const h = harness({post:async () => { throw new Error('Timeout'); }, status:state('ready')});
  await h.client.start(); await h.client.refresh();
  assert.deepEqual(h.events.predictions.map(p => p.status), ['running', 'error', 'ready']);
  assert.equal(h.events.predictions.at(-1).window_id, 'window-0');
  assert.deepEqual(h.posts(), ['window-0']);
});

test('refresh cadence defaults to five minutes; changing it schedules the next refresh without an immediate request', async () => {
  const h = harness(); await h.client.start();
  h.env.status = state('ready'); await h.tick(1_500);
  const before = h.events.calls.length;
  for (const milliseconds of [60_000, 300_000, 600_000, 1_800_000]) {
    h.client.setRefreshInterval(milliseconds);
    assert.deepEqual([...h.timers.values()].map(timer => timer.ms), [milliseconds]);
    assert.equal(h.events.calls.length, before, 'Choosing an interval must not fetch or submit.');
  }
  await h.tick(300_000);
  assert.equal(h.events.calls.length, before, 'The old timer was removed.');
  await h.tick(1_800_000);
  assert.deepEqual(h.posts(), ['window-0'], 'An unchanged window still cannot be resampled.');
  h.next(30); await h.tick(1_800_000);
  assert.deepEqual(h.posts(), ['window-0', 'window-30']);
});

test('refresh selection persists across stop/start and can be set while stopped', async () => {
  const h = harness();
  h.client.setRefreshInterval(600_000);
  assert.equal(h.timers.size, 0); assert.equal(h.events.calls.length, 0);
  await h.client.start(); h.env.status = state('ready'); await h.tick(1_500);
  assert.deepEqual([...h.timers.values()].map(timer => timer.ms), [600_000]);
  h.client.stop(); h.client.setRefreshInterval(1_800_000);
  assert.equal(h.timers.size, 0);
  await h.client.start();
  assert.deepEqual([...h.timers.values()].map(timer => timer.ms), [1_800_000]);
  assert.deepEqual(h.posts(), ['window-0']);
});

test('invalid refresh intervals fail without altering the active timer or making requests', async () => {
  const h = harness(); await h.client.start();
  const timers = [...h.timers], calls = h.events.calls.length;
  for (const invalid of [undefined, null, '60000', 0, -1, 59_999, 60_000.1, NaN, Infinity, 2_147_483_648]) {
    assert.throws(() => h.client.setRefreshInterval(invalid), RangeError);
  }
  assert.deepEqual([...h.timers], timers);
  assert.equal(h.events.calls.length, calls);
});
