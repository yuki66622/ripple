export const DATASETS = ["radar", "graph", "replays", "nearest", "evaluation", "method", "acceptance", "explorer"];

export async function loadDatasets(fetcher = fetch) {
  const settled = await Promise.allSettled(DATASETS.map(async (name) => {
    const response = await fetcher(`data/${name}.json`, { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }));
  const data = {}, errors = {};
  settled.forEach((result, i) => {
    const name = DATASETS[i];
    if (result.status === "fulfilled") data[name] = result.value;
    else errors[name] = String(result.reason?.message || "数据不可用");
  });
  return { data, errors };
}
