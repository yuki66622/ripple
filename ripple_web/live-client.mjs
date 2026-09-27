import { isEnglish, tr } from './i18n.mjs?v=en-r1';

// Translate public errors without exposing server exception details in English.
export function publicLiveError(message, fallback = 'Market data is unavailable. Please try again.') {
  if (!isEnglish) return message;
  const text = String(message || '');
  if (/行情读取失败|公开行情暂时连接失败/.test(text)) return 'Market connection failed. Please try again.';
  if (/行情正在连接/.test(text)) return 'Connecting to the market feed. Please try again.';
  if (/窗口已过期|行情已过期或连接异常/.test(text)) return 'This market window is stale. Refresh the market data.';
  if (/上一次分析仍在运行/.test(text)) return 'The previous analysis is still running. Please wait.';
  if (/模型分析未能通过校验/.test(text)) return 'The forecast failed validation. Market prices are still available.';
  if (/结果未能保存/.test(text)) return 'The forecast could not be saved and was not published.';
  if (/仅允许本机/.test(text)) return 'This action is available only on the local server.';
  if (/未知接口|未找到该页面/.test(text)) return 'The requested resource was not found.';
  if (/分析请求无效/.test(text)) return 'The analysis request is invalid.';
  if (/十资产行情不完整|Market data for the ten assets is incomplete/.test(text)) return 'Market data for the ten assets is incomplete.';
  if (/预测状态格式无效|Invalid forecast status/.test(text)) return 'The forecast status is invalid.';
  if (/timeout|timed out/i.test(text)) return 'The request timed out. Please try again.';
  return fallback;
}

// One frozen prediction per closed-minute window. Newer windows replace queued work.
export function createLiveClient({onMarket, onPrediction, onError}, runtime = {}) {
  const fetcher = runtime.fetch || ((...args) => globalThis.fetch(...args));
  const every = runtime.setInterval || ((...args) => globalThis.setInterval(...args));
  const clear = runtime.clearInterval || (id => globalThis.clearInterval(id));
  const now = runtime.now || (() => Date.now());
  const hidden = runtime.isHidden || (() => Boolean(globalThis.document?.hidden));
  const signal = runtime.timeoutSignal || (() => AbortSignal.timeout(25_000));
  let enabled = false, fetching = false, polling = false;
  let interval = null, predictionTimer = null, latest = null, job = null;
  let refreshInterval = 300_000;
  let unreconciledWindow = null;
  const attempted = new Set();

  const request = async (url, options = {}) => {
    const response = await fetcher(url, {cache:'no-store', signal:signal(), ...options});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    return body;
  };
  const fresh = snapshot => {
    const origin = Date.parse(snapshot?.as_of);
    return Boolean(snapshot && typeof snapshot.window_id === 'string' && snapshot.window_id
      && Number.isFinite(origin) && !snapshot.stale && !snapshot.error
      && now() - origin <= 180_000
      && (snapshot.age_seconds == null || (Number.isFinite(snapshot.age_seconds) && snapshot.age_seconds <= 180)));
  };
  const validateState = result => {
    if (!result || !['idle', 'running', 'ready', 'error'].includes(result.status)
      || (result.status !== 'idle' && (typeof result.window_id !== 'string' || !result.window_id))) {
      throw new Error(tr('预测状态格式无效', 'Invalid forecast status'));
    }
    return result;
  };
  function stopPolling() {
    if (predictionTimer !== null) clear(predictionTimer);
    predictionTimer = null;
  }
  function keepPolling() {
    if (predictionTimer === null) predictionTimer = every(pollPrediction, 1_500);
  }
  function receive(result) {
    validateState(result);
    // Reconciliation may return an older job or idle. Neither supersedes the
    // current window's submit failure; only its own state can recover that UI.
    if (unreconciledWindow === null || result.window_id === unreconciledWindow) {
      unreconciledWindow = null;
      onPrediction(result);
    }
    if (result.status === 'running') {
      // A previous tab/request may own the running job. It still blocks new work.
      attempted.add(result.window_id);
      job = {window_id:result.window_id, as_of:result.as_of};
      keepPolling();
    } else {
      if (result.window_id) attempted.add(result.window_id);
      job = null;
      stopPolling();
    }
  }
  async function pollPrediction() {
    if (polling) return;
    polling = true;
    try { receive(await request('/api/prediction')); }
    catch (error) {
      // Unknown state keeps the execution slot reserved until the server answers.
      onError(tr(`预测状态暂不可用：${error.message}`, 'Forecast status is temporarily unavailable. Reconnecting…'), {scope:'prediction'});
      keepPolling();
    } finally { polling = false; }
    await maybePredict();
  }
  async function maybePredict() {
    if (!enabled || hidden() || job || polling || !fresh(latest) || attempted.has(latest.window_id)) return;
    const snapshot = latest;
    job = {window_id:snapshot.window_id, as_of:snapshot.as_of};
    attempted.add(snapshot.window_id);
    unreconciledWindow = null;
    onPrediction({status:'running', ...job, error:null});
    try {
      receive(await request('/api/predict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({window_id:snapshot.window_id}),
      }));
    } catch (error) {
      unreconciledWindow = snapshot.window_id;
      onPrediction({status:'error', window_id:snapshot.window_id, as_of:snapshot.as_of,
        error:tr(`本窗口预测连接失败：${error.message}`, 'The forecast connection failed for this window.')});
      onError(publicLiveError(error.message, 'The forecast connection failed. Checking its status…'), {scope:'prediction'});
      // A timeout can occur after the server accepted the request. Read its state;
      // never submit this window again, even if the status read also fails.
      await pollPrediction();
      return;
    }
    await maybePredict();
  }
  async function refresh() {
    if (fetching) return;
    fetching = true;
    try {
      const snapshot = await request('/api/live');
      onMarket(snapshot);
      latest = snapshot;
    } catch (error) {
      latest = null;
      onError(publicLiveError(error.message), {scope:'market'});
    } finally { fetching = false; }
    await maybePredict();
  }
  function scheduleRefresh() {
    if (enabled && interval === null) interval = every(() => { if (!hidden()) return refresh(); }, refreshInterval);
  }
  return {
    refresh,
    setRefreshInterval(milliseconds) {
      // Native timers clamp overflowing delays to near-zero; reject them too.
      if (!Number.isInteger(milliseconds) || milliseconds < 60_000 || milliseconds > 2_147_483_647) {
        throw new RangeError('Refresh interval must be an integer between 60000 and 2147483647 milliseconds.');
      }
      if (milliseconds === refreshInterval) return;
      refreshInterval = milliseconds;
      if (interval !== null) clear(interval);
      interval = null;
      scheduleRefresh();
    },
    // Compatibility only: this obeys the same active/fresh/once-only constraints.
    async predict(windowId) { if (latest?.window_id === windowId) await maybePredict(); },
    start() {
      enabled = true;
      latest = null;
      scheduleRefresh();
      if (!hidden()) return refresh();
    },
    stop() {
      enabled = false;
      latest = null;
      if (interval !== null) clear(interval);
      interval = null;
      // Running inference may finish; polling cannot schedule work while stopped.
    },
  };
}
