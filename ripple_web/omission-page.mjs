import { renderOmissions } from "./blocks/omission.mjs?v=omission-r1";

async function load() {
  const host = document.getElementById("omission-content");
  try {
    const response = await fetch("data/omission.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.metrics?.length !== 2 || !data.cases?.length) throw new Error("数据不完整");
    renderOmissions(host, data);
    document.body.dataset.loadState = "ready";
  } catch {
    host.replaceChildren();
    const message = document.createElement("p");
    message.textContent = "暂时无法读取冻结的遗漏分析。";
    const retry = document.createElement("button");
    retry.type = "button"; retry.textContent = "重新读取";
    retry.addEventListener("click", load);
    host.append(message, retry);
    document.body.dataset.loadState = "error";
  }
}
load();
