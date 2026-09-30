// app.js — all frontend logic for the TGI LLM Serving Dashboard.
// No build step required: open index.html via a simple static server
// (see README.md) and it talks to the FastAPI backend below.

const API_BASE = "http://localhost:8000";

// ---------- Tab switching ----------
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// ---------- Dashboard ----------
async function refreshDashboard() {
  try {
    const healthResp = await fetch(`${API_BASE}/health`);
    const health = await healthResp.json();
    document.getElementById("statServerStatus").textContent = health.status === "ok" ? "🟢 Online" : "🟡 Loading";
    document.getElementById("statModel").textContent = health.model_name;

    const metricsResp = await fetch(`${API_BASE}/metrics`);
    const m = await metricsResp.json();
    document.getElementById("statRequests").textContent = m.request_count;
    document.getElementById("statLatency").textContent = m.average_latency_seconds > 0 ? `${m.average_latency_seconds}s` : "—";
    document.getElementById("statTps").textContent = m.average_tokens_per_second > 0 ? m.average_tokens_per_second : "—";
    document.getElementById("statUptime").textContent = `${Math.round(m.uptime_seconds)}s`;
  } catch (e) {
    document.getElementById("statServerStatus").textContent = "🔴 Offline";
  }
}
document.getElementById("refreshMetricsBtn").addEventListener("click", refreshDashboard);
refreshDashboard();
setInterval(refreshDashboard, 8000);

// ---------- Text Generation ----------
document.getElementById("generateBtn").addEventListener("click", async () => {
  const prompt = document.getElementById("promptInput").value.trim();
  const maxTokens = parseInt(document.getElementById("maxTokensInput").value, 10);
  const stream = document.getElementById("streamToggle").checked;
  const outputBox = document.getElementById("generateOutput");
  const statsBox = document.getElementById("generateStats");
  if (!prompt) { alert("Please enter a prompt."); return; }

  outputBox.textContent = "";
  statsBox.textContent = "Generating…";

  const start = performance.now();

  if (stream) {
    const resp = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_new_tokens: maxTokens, stream: true }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.startsWith("data:")) continue;
        const jsonStr = part.replace("data:", "").trim();
        try {
          const data = JSON.parse(jsonStr);
          if (data.token) outputBox.textContent += data.token;
          if (data.done) {
            statsBox.textContent = `Latency: ${data.latency_seconds}s (streamed)`;
          }
        } catch (_) {}
      }
    }
  } else {
    const resp = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_new_tokens: maxTokens, stream: false }),
    });
    const data = await resp.json();
    outputBox.textContent = data.generated_text;
    statsBox.textContent = `Latency: ${data.latency_seconds}s · Tokens: ${data.new_tokens} · Tokens/sec: ${data.tokens_per_second}`;
  }
  refreshDashboard();
});

// ---------- Batching Demo ----------
function renderBatchResults(container, data) {
  let html = `<div class="batch-request"><strong>${data.mode.toUpperCase()}</strong> — ${data.explanation}<br/>`;
  if (data.total_latency_seconds !== undefined) {
    html += `<div class="meta">Total latency: ${data.total_latency_seconds}s</div>`;
  }
  html += `</div>`;

  const items = data.requests || data.timeline || [];
  items.forEach((item, i) => {
    html += `<div class="batch-request">
      <div class="meta">Request ${i + 1}${item.arrival_time_seconds !== undefined ? " · arrived at " + item.arrival_time_seconds + "s" : ""}${item.latency_seconds ? " · latency " + item.latency_seconds.toFixed(3) + "s" : ""}</div>
      <div><em>${item.prompt}</em></div>
      <div>${item.generated_text || ""}</div>
    </div>`;
  });
  container.innerHTML = html;
}

document.getElementById("runSequentialBtn").addEventListener("click", async () => {
  const prompts = document.getElementById("batchPrompts").value.split("\n").map((p) => p.trim()).filter(Boolean);
  const container = document.getElementById("batchResults");
  container.innerHTML = "Running sequential requests…";
  const resp = await fetch(`${API_BASE}/batch/demo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompts, mode: "sequential", max_new_tokens: 25 }),
  });
  const data = await resp.json();
  renderBatchResults(container, data);
  refreshDashboard();
});

document.getElementById("runBatchedBtn").addEventListener("click", async () => {
  const prompts = document.getElementById("batchPrompts").value.split("\n").map((p) => p.trim()).filter(Boolean);
  const container = document.getElementById("batchResults");
  container.innerHTML = "Running batched requests…";
  const resp = await fetch(`${API_BASE}/batch/demo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompts, mode: "batched", max_new_tokens: 25 }),
  });
  const data = await resp.json();
  renderBatchResults(container, data);
  refreshDashboard();
});

// ---------- TGI Concepts ----------
const concepts = [
  { title: "Continuous Batching", text: "New requests are merged into the active batch as soon as capacity allows, instead of waiting for the whole batch to finish — keeps the GPU busy under mixed request lengths." },
  { title: "KV Cache", text: "Previously computed attention key/value tensors are reused instead of recomputed for every new token, which is what makes long generations fast. Try the /demo/kv-cache endpoint to see the measured difference." },
  { title: "Quantization", text: "Lower-precision weights (INT8, 4-bit, GPTQ, AWQ, FP8) shrink a model's memory footprint so larger models fit on smaller/fewer GPUs. This demo uses CPU dynamic INT8 quantization as a concept-equivalent." },
  { title: "Tensor Parallelism", text: "A model too large for one GPU is sharded across several GPUs, each computing part of every layer, synchronised via NCCL. Not runnable on a single-GPU-less laptop — see tgi/docker-compose.yml for the real flags." },
  { title: "Token Streaming", text: "Generated tokens are sent to the client as soon as they're produced (Server-Sent Events), so the first words appear almost immediately instead of after the full response completes." },
  { title: "OpenAI-Compatible API", text: "Exposing /v1/chat/completions in the same shape as OpenAI's API lets existing client code point at a self-hosted server with minimal changes." },
];
const conceptGrid = document.getElementById("conceptGrid");
concepts.forEach((c) => {
  const div = document.createElement("div");
  div.className = "concept-card";
  div.innerHTML = `<h4>${c.title}</h4><p>${c.text}</p>`;
  conceptGrid.appendChild(div);
});

// ---------- API Demo ----------
const exampleRequest = {
  model: "tgi-demo",
  messages: [{ role: "user", content: "What is Text Generation Inference?" }],
  max_tokens: 40,
  stream: false,
};
document.getElementById("apiRequestExample").textContent =
  `curl -X POST ${API_BASE}/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(exampleRequest)}'`;

document.getElementById("tryApiBtn").addEventListener("click", async () => {
  const out = document.getElementById("apiResponseExample");
  out.textContent = "Calling backend…";
  const resp = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(exampleRequest),
  });
  const data = await resp.json();
  out.textContent = JSON.stringify(data, null, 2);
  refreshDashboard();
});

// ---------- Architecture diagram (inline SVG) ----------
document.getElementById("archDiagram").innerHTML = `
<svg viewBox="0 0 900 260" xmlns="http://www.w3.org/2000/svg" width="100%">
  <style>
    .box { fill:#1c2438; stroke:#6c8cff; stroke-width:1.5; rx:8; }
    .label { fill:#e7ebf5; font-size:13px; font-family:Segoe UI, sans-serif; text-anchor:middle; }
    .sub { fill:#9aa4bf; font-size:10px; font-family:Segoe UI, sans-serif; text-anchor:middle; }
    .arrow { stroke:#37d6a7; stroke-width:2; marker-end:url(#arrowhead); fill:none; }
  </style>
  <defs>
    <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <polygon points="0 0, 6 3, 0 6" fill="#37d6a7" />
    </marker>
  </defs>

  <rect class="box" x="10" y="100" width="110" height="60"/>
  <text class="label" x="65" y="128">User</text>
  <text class="sub" x="65" y="144">Client / Browser</text>

  <rect class="box" x="150" y="100" width="120" height="60"/>
  <text class="label" x="210" y="128">API / Router</text>
  <text class="sub" x="210" y="144">FastAPI (Rust in real TGI)</text>

  <rect class="box" x="300" y="100" width="140" height="60"/>
  <text class="label" x="370" y="122">Scheduler</text>
  <text class="sub" x="370" y="138">Continuous batching</text>
  <text class="sub" x="370" y="150">(queue + batch window)</text>

  <rect class="box" x="470" y="100" width="140" height="60"/>
  <text class="label" x="540" y="122">Model Server</text>
  <text class="sub" x="540" y="138">model_service.py</text>
  <text class="sub" x="540" y="150">(Python model server in TGI)</text>

  <rect class="box" x="640" y="100" width="110" height="60"/>
  <text class="label" x="695" y="128">LLM</text>
  <text class="sub" x="695" y="144">Qwen2.5 Instruct / real model</text>

  <rect class="box" x="300" y="10" width="140" height="50"/>
  <text class="label" x="370" y="32">Launcher</text>
  <text class="sub" x="370" y="46">spawns router + shards</text>

  <rect class="box" x="470" y="10" width="220" height="50"/>
  <text class="label" x="580" y="32">Multi-GPU (real TGI)</text>
  <text class="sub" x="580" y="46">Tensor-parallel shards via NCCL</text>

  <path class="arrow" d="M120,130 L150,130"/>
  <path class="arrow" d="M270,130 L300,130"/>
  <path class="arrow" d="M440,130 L470,130"/>
  <path class="arrow" d="M610,130 L640,130"/>
  <path class="arrow" d="M370,60 L370,100"/>
  <path class="arrow" d="M580,60 L540,100"/>
</svg>
`;
