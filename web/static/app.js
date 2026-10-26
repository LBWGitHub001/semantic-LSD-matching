/**
 * Semantic LSD Matching Web Management Interface
 * Frontend controller: WebSocket terminal streaming, Chart rendering, API integrations.
 */

// Global State
let currentTab = "dashboard";
let socket = null;
let autoScroll = true;
let allRuns = [];
let allGalleryImages = [];
let currentGalleryIdx = 0;
let liveLossHistory = [];
let liveF1History = [];

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initWebSocket();
  initStatusPolling();
  loadDashboard();
  loadDatasets();
  loadCheckpoints();
  loadGallery();
});

// -------------------------------------------------------------
// Tabs Navigation
// -------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const target = tab.dataset.tab;
      currentTab = target;
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      document.getElementById(`tab-${target}`).classList.add("active");

      // Tab specific refresh
      if (target === "dashboard") loadDashboard();
      if (target === "gallery") loadGallery();
      if (target === "eval") loadCheckpoints();
    });
  });
}

// -------------------------------------------------------------
// WebSocket Terminal Log Streaming
// -------------------------------------------------------------
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/logs`;

  const wsBadge = document.getElementById("ws-badge");
  if (wsBadge) {
    wsBadge.innerText = "○ WS 连接中...";
    wsBadge.className = "badge badge-warning";
  }

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log("[WS] Connected to log stream");
    if (wsBadge) {
      wsBadge.innerText = "● WS 已连接";
      wsBadge.className = "badge badge-success";
    }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleSocketMessage(data);
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };

  socket.onclose = () => {
    console.log("[WS] Disconnected, reconnecting in 3s...");
    if (wsBadge) {
      wsBadge.innerText = "○ WS 重连中...";
      wsBadge.className = "badge badge-warning";
    }
    setTimeout(initWebSocket, 3000);
  };
}

function handleSocketMessage(data) {
  if (data.type === "init") {
    if (data.backlog && data.backlog.length > 0) {
      clearTerminal();
      data.backlog.forEach(line => appendTerminalLine(line));
    }
    updateStatusUI(data.status);
  } else if (data.type === "log") {
    appendTerminalLine(data.line);
    parseMetricsFromLine(data.line);
  } else if (data.type === "task_started") {
    updateStatusUI({ is_busy: true, active_task: data.task });
    appendTerminalLine(`\n[+] ====== Task Started: ${data.task.task_type} (${data.task.task_id}) ======\n`);
    const select = document.getElementById("log-source-select");
    if (select) select.value = "live";
  } else if (data.type === "task_stopped" || data.type === "task_finished") {
    updateStatusUI({ is_busy: false, active_task: null });
    appendTerminalLine(`\n[*] ====== Task Finished (${data.type}) ======\n`);
    loadDashboard();
    loadGallery();
  }
}

async function switchLogSource() {
  const select = document.getElementById("log-source-select");
  if (!select) return;
  const val = select.value;

  if (val === "live") {
    clearTerminal();
    appendTerminalLine("[*] 已切换至实时任务控制台。\n");
    try {
      const res = await fetch("/api/logs/active");
      const data = await res.json();
      if (data.logs && data.logs.length > 0) {
        data.logs.forEach(line => appendTerminalLine(line));
      } else {
        appendTerminalLine("[i] 当前暂无正在运行的任务。点击“开始训练”或“运行基准评测”即可在此处查看实时输出。");
      }
    } catch (e) {
      console.error("Failed to load active logs:", e);
    }
  } else {
    // Load historical run log
    clearTerminal();
    appendTerminalLine(`[*] 正在读取实验 [${val}] 的 train.log ...\n`);
    try {
      const res = await fetch(`/api/logs/history?run=${val}`);
      const data = await res.json();
      clearTerminal();
      if (data.lines && data.lines.length > 0) {
        data.lines.forEach(line => appendTerminalLine(line));
      } else {
        appendTerminalLine(`[!] 实验 ${val} 的 train.log 为空或不存在。`);
      }
    } catch (e) {
      appendTerminalLine(`[!] 读取日志失败: ${e}`);
    }
  }
}

function appendTerminalLine(rawText) {
  const terminal = document.getElementById("terminal-logs");
  if (!terminal) return;

  const colored = ansiToHtml(rawText);
  // Detect tqdm progress bar updates (e.g. 50%|##### | 150/300)
  const isProgressBar = rawText.includes("%|") || rawText.includes("pair/s") || rawText.includes("it/s");

  if (isProgressBar && terminal.lastElementChild && terminal.lastElementChild.dataset.isProgress === "true") {
    // In-place update to mimic terminal \r behavior (smooth progress bar)
    terminal.lastElementChild.innerHTML = colored;
  } else {
    const div = document.createElement("div");
    div.innerHTML = colored;
    if (isProgressBar) {
      div.dataset.isProgress = "true";
    }
    terminal.appendChild(div);
  }

  // Keep terminal buffer clean (max 1200 elements in DOM)
  while (terminal.childNodes.length > 1200) {
    terminal.removeChild(terminal.firstChild);
  }

  if (autoScroll) {
    terminal.scrollTop = terminal.scrollHeight;
  }
}

function toggleAutoScroll() {
  autoScroll = !autoScroll;
  const btn = document.getElementById("btn-autoscroll");
  if (btn) btn.innerText = autoScroll ? "滚动: 开" : "滚动: 关";
}

function clearTerminal() {
  const terminal = document.getElementById("terminal-logs");
  if (terminal) terminal.innerHTML = "";
}

function ansiToHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\033\[([0-9;]+)m/g, (match, p1) => {
      if (p1 === "0") return '</span>';
      if (p1.includes("31")) return '<span style="color:#EF4444;">';
      if (p1.includes("32")) return '<span style="color:#10B981;">';
      if (p1.includes("33")) return '<span style="color:#F59E0B;">';
      if (p1.includes("34")) return '<span style="color:#3B82F6;">';
      if (p1.includes("35")) return '<span style="color:#8B5CF6;">';
      if (p1.includes("36")) return '<span style="color:#06B6D4;">';
      return '<span>';
    });
}

function parseMetricsFromLine(line) {
  // Regex to extract Loss and Val F1 during training:
  // e.g.: "Epoch [01/15] (152.8s) | Loss: 1.5467 | Val Prec:  67.1% | Rec:  58.1% | F1:  62.3%"
  const match = line.match(/Epoch\s+\[(\d+)\/\d+\].*?Loss:\s+([\d\.]+).*?F1:\s+([\d\.]+)%/);
  if (match) {
    const ep = parseInt(match[1]);
    const loss = parseFloat(match[2]);
    const f1 = parseFloat(match[3]);

    liveLossHistory.push({ epoch: ep, val: loss });
    liveF1History.push({ epoch: ep, val: f1 });
    renderLiveChart();
  }
}

// -------------------------------------------------------------
// Live Mini Chart (HTML5 Canvas)
// -------------------------------------------------------------
function renderLiveChart() {
  const canvas = document.getElementById("live-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  if (liveLossHistory.length < 2) {
    ctx.fillStyle = "#6B7280";
    ctx.font = "12px sans-serif";
    ctx.fillText("等待训练生成多轮指标曲线...", 20, h / 2);
    return;
  }

  // Draw grid
  ctx.strokeStyle = "rgba(55, 65, 81, 0.4)";
  ctx.lineWidth = 1;
  for (let y = 30; y < h - 20; y += 40) {
    ctx.beginPath();
    ctx.moveTo(30, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }

  // Plot Loss (Red line, normalized to left axis)
  const maxLoss = Math.max(...liveLossHistory.map(d => d.val), 1.6);
  ctx.strokeStyle = "#EF4444";
  ctx.lineWidth = 2;
  ctx.beginPath();
  liveLossHistory.forEach((pt, idx) => {
    const x = 40 + (idx / (liveLossHistory.length - 1)) * (w - 70);
    const y = (h - 30) - (pt.val / maxLoss) * (h - 60);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Plot Val F1 (Blue line, 0 - 100%)
  ctx.strokeStyle = "#3B82F6";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  liveF1History.forEach((pt, idx) => {
    const x = 40 + (idx / (liveF1History.length - 1)) * (w - 70);
    const y = (h - 30) - (pt.val / 100.0) * (h - 60);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Legend
  ctx.fillStyle = "#EF4444";
  ctx.font = "11px sans-serif";
  ctx.fillText("— Train Loss", w - 160, 20);
  ctx.fillStyle = "#3B82F6";
  ctx.fillText("— Val F1 (%)", w - 80, 20);
}

// -------------------------------------------------------------
// Status Polling & Indicators
// -------------------------------------------------------------
function initStatusPolling() {
  fetchStatus();
  setInterval(fetchStatus, 2000);
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    updateStatusUI(data);
  } catch (e) {
    console.error("Status fetch failed:", e);
  }
}

function updateStatusUI(status) {
  if (!status) return;
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  const gpuBadge = document.getElementById("gpu-badge");

  if (status.is_busy && status.active_task) {
    dot.className = "dot dot-busy";
    text.innerText = `运行中: ${status.active_task.task_type} (${status.active_task.task_id})`;
  } else {
    dot.className = "dot dot-idle";
    text.innerText = "系统空闲 (Ready)";
  }

  const res = status.resources || {};
  const gpu = (res.gpu && res.gpu.available) ? res.gpu : status.gpu;

  if (gpu && gpu.available) {
    gpuBadge.innerText = `GPU: ${gpu.memory_used_mb}MB / ${gpu.memory_total_mb}MB (${gpu.memory_pct}%) | ${gpu.temperature_c}°C`;
  } else {
    gpuBadge.innerText = "CPU 模式";
  }

  // Update Right-side Hardware Resource Monitor
  if (res.cpu) {
    const cpuPctEl = document.getElementById("res-cpu-pct");
    const cpuBarEl = document.getElementById("res-cpu-bar");
    const cpuCoresEl = document.getElementById("res-cpu-cores");
    const cpuLoadEl = document.getElementById("res-cpu-load");

    if (cpuPctEl) cpuPctEl.innerText = `${res.cpu.pct.toFixed(1)}%`;
    if (cpuBarEl) cpuBarEl.style.width = `${Math.min(100, Math.max(0, res.cpu.pct))}%`;
    if (cpuCoresEl) cpuCoresEl.innerText = `${res.cpu.count} 核心 (Threads)`;
    if (cpuLoadEl) cpuLoadEl.innerText = `负载: ${res.cpu.load_1m} / ${res.cpu.load_5m}`;
  }

  if (res.ram) {
    const ramPctEl = document.getElementById("res-ram-pct");
    const ramBarEl = document.getElementById("res-ram-bar");
    const ramUsedEl = document.getElementById("res-ram-used");
    const swapUsedEl = document.getElementById("res-swap-used");

    if (ramPctEl) ramPctEl.innerText = `${res.ram.pct.toFixed(1)}%`;
    if (ramBarEl) ramBarEl.style.width = `${Math.min(100, Math.max(0, res.ram.pct))}%`;
    if (ramUsedEl) ramUsedEl.innerText = `已用: ${res.ram.used_gb} / ${res.ram.total_gb} GB`;
    if (swapUsedEl) swapUsedEl.innerText = `交换区: ${res.ram.swap_used_gb} / ${res.ram.swap_total_gb} GB`;
  }

  if (gpu && gpu.available) {
    const gpuPctEl = document.getElementById("res-gpu-pct");
    const gpuBarEl = document.getElementById("res-gpu-bar");
    const gpuNameEl = document.getElementById("res-gpu-name");
    const gpuTempEl = document.getElementById("res-gpu-temp");

    if (gpuPctEl) gpuPctEl.innerText = `${gpu.utilization_pct.toFixed(1)}%`;
    if (gpuBarEl) gpuBarEl.style.width = `${Math.min(100, Math.max(0, gpu.utilization_pct))}%`;
    if (gpuNameEl) gpuNameEl.innerText = gpu.name || "NVIDIA RTX GPU";
    if (gpuTempEl) gpuTempEl.innerText = `温度: ${gpu.temperature_c}°C`;

    const vramPctEl = document.getElementById("res-vram-pct");
    const vramBarEl = document.getElementById("res-vram-bar");
    const vramUsedEl = document.getElementById("res-vram-used");
    const gpuPowerEl = document.getElementById("res-gpu-power");

    if (vramPctEl) vramPctEl.innerText = `${gpu.memory_pct.toFixed(1)}%`;
    if (vramBarEl) vramBarEl.style.width = `${Math.min(100, Math.max(0, gpu.memory_pct))}%`;
    if (vramUsedEl) vramUsedEl.innerText = `已用: ${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB`;
    if (gpuPowerEl) gpuPowerEl.innerText = `功耗: ${gpu.power_w ? gpu.power_w + " W" : "-"}`;
  }

  // Diagnostic helper text
  const helperEl = document.getElementById("res-helper-text");
  if (helperEl) {
    if (status.is_busy) {
      if (gpu && gpu.utilization_pct >= 60) {
        helperEl.innerText = "🟢 GPU满载计算中，数据流水线畅通！";
      } else if (gpu && gpu.utilization_pct < 15) {
        helperEl.innerText = "🟡 GPU占用较低，可能受限于CPU数据加载";
      } else {
        helperEl.innerText = "🔵 模型训练 / 评测正在进行中";
      }
    } else {
      helperEl.innerText = "⚪ 系统空闲，硬件就绪，随时可启动训练";
    }
  }
}

// -------------------------------------------------------------
// Tab 1: Dashboard & Runs
// -------------------------------------------------------------
async function loadDashboard() {
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    allRuns = data.runs || [];

    // Summary Cards
    document.getElementById("stat-total-runs").innerText = allRuns.length;

    let bestRaw = 0;
    let bestSem = 0;
    allRuns.forEach(r => {
      if (r.test_raw_f1 && r.test_raw_f1 > bestRaw) bestRaw = r.test_raw_f1;
      if (r.test_sem_f1 && r.test_sem_f1 > bestSem) bestSem = r.test_sem_f1;
    });
    document.getElementById("stat-best-raw-f1").innerText = bestRaw ? `${bestRaw.toFixed(1)}%` : "-";
    document.getElementById("stat-best-sem-f1").innerText = bestSem ? `${bestSem.toFixed(1)}%` : "-";

    // Table rows
    const tbody = document.getElementById("runs-table-body");
    tbody.innerHTML = "";

    // Update log source select dropdown in terminal
    const logSelect = document.getElementById("log-source-select");
    if (logSelect) {
      const cur = logSelect.value;
      logSelect.innerHTML = `<option value="live">实时任务输出 (Live)</option>`;
      allRuns.forEach(r => {
        logSelect.innerHTML += `<option value="${r.run_name}">历史日志: ${r.run_name} (${r.model_type})</option>`;
      });
      if (cur) logSelect.value = cur;
    }

    allRuns.forEach((r) => {
      const tr = document.createElement("tr");
      const bestF1Str = r.val_f1 ? `${r.val_f1.toFixed(1)}% (Ep ${r.best_epoch})` : "-";
      const rawF1Str = r.test_raw_f1 ? `<b style="color:#60A5FA;">${r.test_raw_f1.toFixed(1)}%</b>` : "-";
      const semF1Str = r.test_sem_f1 ? `<b style="color:#34D399;">${r.test_sem_f1.toFixed(1)}%</b>` : "-";

      tr.innerHTML = `
        <td><input type="checkbox" class="run-select-box" value="${r.run_name}" onchange="updateCompareBtn()"></td>
        <td><b>${r.run_name}</b></td>
        <td><span class="badge badge-purple">${r.model_type}</span></td>
        <td>${r.start_time}</td>
        <td>${r.epochs}</td>
        <td>${bestF1Str}</td>
        <td>${rawF1Str}</td>
        <td>${semF1Str}</td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="viewRunDetails('${r.run_name}')">复盘报告</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
    updateCompareBtn();
  } catch (e) {
    console.error("Failed to load runs:", e);
  }
}

function updateCompareBtn() {
  const selected = Array.from(document.querySelectorAll(".run-select-box:checked")).map(cb => cb.value);
  const btn = document.getElementById("btn-compare-runs");
  if (btn) {
    btn.disabled = selected.length !== 2;
    btn.innerText = selected.length === 2 ? `横向对比选中的 2 个实验` : `请勾选 2 个实验进行对比 (${selected.length}/2)`;
  }
}

async function compareSelectedRuns() {
  const selected = Array.from(document.querySelectorAll(".run-select-box:checked")).map(cb => cb.value);
  if (selected.length !== 2) return;

  try {
    const res = await fetch(`/api/runs/compare?run1=${selected[0]}&run2=${selected[1]}`);
    const data = await res.json();
    if (data.error) {
      alert(data.message);
      return;
    }

    const modal = document.getElementById("compare-modal");
    document.getElementById("compare-modal-title").innerText = `实验对比: ${data.run1_name} vs ${data.run2_name}`;

    const tbody = document.getElementById("compare-table-body");
    tbody.innerHTML = "";
    data.table.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight:600; color:#9CA3AF;">${row.feature}</td>
        <td><b>${row.run1}</b></td>
        <td><b>${row.run2}</b></td>
      `;
      tbody.appendChild(tr);
    });

    const curveImg = document.getElementById("compare-curve-img");
    if (data.curve_url) {
      curveImg.src = data.curve_url;
      curveImg.style.display = "block";
    } else {
      curveImg.style.display = "none";
    }

    modal.style.display = "flex";
  } catch (e) {
    console.error("Comparison error:", e);
  }
}

async function viewRunDetails(runName) {
  try {
    const res = await fetch(`/api/runs/${runName}`);
    const data = await res.json();
    if (data.error) {
      alert(data.message);
      return;
    }

    const modal = document.getElementById("run-modal");
    document.getElementById("run-modal-title").innerText = `实验复盘详情: ${data.run_name} (${data.model_type})`;

    // Render report markdown (simple parser)
    const reportContainer = document.getElementById("run-report-container");
    if (data.report_markdown) {
      reportContainer.innerHTML = simpleMarkdownRender(data.report_markdown);
    } else {
      reportContainer.innerHTML = "<p style='color:#9CA3AF;'>该次实验尚未生成 report.md 复盘报告。</p>";
    }

    // Curves image
    const curveImg = document.getElementById("run-curves-img");
    if (data.curves_url) {
      curveImg.src = data.curves_url;
      curveImg.style.display = "block";
    } else {
      curveImg.style.display = "none";
    }

    modal.style.display = "flex";
  } catch (e) {
    console.error("Fetch run detail error:", e);
  }
}

function simpleMarkdownRender(md) {
  return md
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/gim, '<b>$1</b>')
    .replace(/`([^`]+)`/gim, '<code style="background:#1F2937; padding:2px 6px; border-radius:4px; color:#F59E0B;">$1</code>')
    .replace(/^> \[\!WARNING\](.*$)/gim, '<blockquote style="border-left:4px solid #EF4444; background:rgba(239,68,68,0.1); padding:8px 12px; margin:8px 0;"><b>[WARNING]</b>$1')
    .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>')
    .replace(/\n/gim, '<br>');
}

function closeModal(id) {
  document.getElementById(id).style.display = "none";
}

// -------------------------------------------------------------
// Tab 2: Training Studio Controls
// -------------------------------------------------------------
async function loadDatasets() {
  try {
    const res = await fetch("/api/datasets");
    const data = await res.json();
    const trainSelect = document.getElementById("train-data-dir");
    const evalSelect = document.getElementById("eval-data-dir");

    if (trainSelect) trainSelect.innerHTML = "";
    if (evalSelect) evalSelect.innerHTML = "";

    (data.datasets || []).forEach(d => {
      const opt = `<option value="${d.rel_path}">${d.name} (${d.total_pairs} 对样本)</option>`;
      if (trainSelect) trainSelect.innerHTML += opt;
      if (evalSelect) evalSelect.innerHTML += opt;
    });

    // Populate dataset cards in Tab 5
    const datasetCardsContainer = document.getElementById("dataset-cards-container");
    if (datasetCardsContainer) {
      datasetCardsContainer.innerHTML = "";
      (data.datasets || []).forEach(d => {
        datasetCardsContainer.innerHTML += `
          <div class="card" style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div>
                <b style="font-size:15px; color:#60A5FA;">${d.name}</b>
                <div style="color:#9CA3AF; font-size:12px; margin-top:4px;">路径: ${d.rel_path}</div>
              </div>
              <div style="text-align:right;">
                <span class="badge badge-success">Train: ${d.train_pairs}</span>
                <span class="badge badge-blue">Val: ${d.val_pairs}</span>
                <span class="badge badge-purple">Test: ${d.test_pairs}</span>
              </div>
            </div>
          </div>
        `;
      });
    }
  } catch (e) {
    console.error("Failed to load datasets:", e);
  }
}

async function startTraining() {
  const model = document.getElementById("train-model").value;
  const data_dir = document.getElementById("train-data-dir").value;
  const epochs = parseInt(document.getElementById("train-epochs").value);
  const lr = parseFloat(document.getElementById("train-lr").value);
  const dustbin_weight = parseFloat(document.getElementById("train-dustbin").value);
  const motion_weight = parseFloat(document.getElementById("train-motion").value);
  const min_area_ratio = parseFloat(document.getElementById("train-area-ratio").value);
  const pyramid = document.getElementById("train-pyramid").checked;
  const amp = document.getElementById("train-amp").checked;

  const btnStart = document.querySelector("#tab-train .btn-success");
  if (btnStart) {
    btnStart.innerText = "⏳ 启动中...";
    btnStart.disabled = true;
  }

  const logSelect = document.getElementById("log-source-select");
  if (logSelect) logSelect.value = "live";

  clearTerminal();
  appendTerminalLine(`[+] 正在请求启动训练: model=${model}, data=${data_dir}, epochs=${epochs}, lr=${lr}...`);

  liveLossHistory = [];
  liveF1History = [];
  renderLiveChart();

  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model, data_dir, epochs, lr, dustbin_weight, motion_weight, min_area_ratio, pyramid, amp
      }),
    });
    const result = await res.json();
    if (result.error) {
      alert(`启动失败: ${result.message}`);
      appendTerminalLine(`\n[!] 启动失败: ${result.message}\n`);
    } else {
      appendTerminalLine(`\n[✓] 训练子进程成功启动! PID: ${result.task.pid}\n[*] 实时输出监听中...\n`);
    }
  } catch (e) {
    alert(`网络请求异常: ${e}`);
    appendTerminalLine(`\n[!] 请求发生网络异常: ${e}\n`);
  } finally {
    if (btnStart) {
      btnStart.innerText = "▶ 开始训练";
      btnStart.disabled = false;
    }
  }
}

async function stopTraining() {
  if (!confirm("确定要平滑停止当前训练吗？模型将捕获中断信号并安全保存当前权重与配置。")) return;
  try {
    const res = await fetch("/api/train?action=stop", { method: "POST" });
    const result = await res.json();
    appendTerminalLine(`\n[*] ${result.message}\n`);
  } catch (e) {
    alert(`停止请求失败: ${e}`);
  }
}

// -------------------------------------------------------------
// Tab 3: Evaluation Benchmark Controls
// -------------------------------------------------------------
async function loadCheckpoints() {
  try {
    const res = await fetch("/api/checkpoints");
    const data = await res.json();
    const select = document.getElementById("eval-checkpoint");
    if (!select) return;

    select.innerHTML = "";
    (data.checkpoints || []).forEach(c => {
      const star = c.is_best ? "★ " : "";
      select.innerHTML += `<option value="${c.rel_path}">${star}${c.folder}/${c.name} (${c.size_mb} MB)</option>`;
    });
  } catch (e) {
    console.error("Failed to load checkpoints:", e);
  }
}

async function startEvaluation() {
  const checkpoint = document.getElementById("eval-checkpoint").value;
  const data_dir = document.getElementById("eval-data-dir").value;
  const split = document.getElementById("eval-split").value;
  const min_area_ratio = parseFloat(document.getElementById("eval-area-ratio").value);
  const num_vis = parseInt(document.getElementById("eval-num-vis").value);

  const btnEval = document.querySelector("#tab-eval .btn-primary");
  if (btnEval) {
    btnEval.innerText = "⏳ 评测中...";
    btnEval.disabled = true;
  }

  // Switch to train tab so user can watch the live terminal output
  const tabTrainBtn = document.querySelector('.tab-btn[data-tab="train"]');
  if (tabTrainBtn) tabTrainBtn.click();

  const logSelect = document.getElementById("log-source-select");
  if (logSelect) logSelect.value = "live";

  clearTerminal();
  appendTerminalLine(`[+] 正在启动评测任务: checkpoint=${checkpoint}, split=${split}, min_area_ratio=${min_area_ratio}...\n`);

  try {
    const res = await fetch("/api/eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint, data_dir, split, min_area_ratio, num_vis }),
    });
    const result = await res.json();
    if (result.error) {
      alert(`评测启动失败: ${result.message}`);
      appendTerminalLine(`\n[!] 评测启动失败: ${result.message}\n`);
    } else {
      appendTerminalLine(`\n[✓] 评测任务启动成功! PID: ${result.task.pid}\n[*] 实时进度监听中...\n`);
    }
  } catch (e) {
    alert(`评测请求异常: ${e}`);
    appendTerminalLine(`\n[!] 评测请求异常: ${e}\n`);
  } finally {
    if (btnEval) {
      btnEval.innerText = "🧪 运行基准评测 (Run Benchmark)";
      btnEval.disabled = false;
    }
  }
}

// -------------------------------------------------------------
// Tab 4: Results Visualization Gallery
// -------------------------------------------------------------
async function loadGallery() {
  try {
    const res = await fetch("/api/gallery");
    const data = await res.json();
    allGalleryImages = data.images || [];

    document.getElementById("gallery-count").innerText = `${allGalleryImages.length} 张样本图`;

    renderGalleryGrid(allGalleryImages);
  } catch (e) {
    console.error("Failed to load gallery:", e);
  }
}

function renderGalleryGrid(images) {
  const grid = document.getElementById("gallery-grid");
  if (!grid) return;
  grid.innerHTML = "";

  images.forEach((img, idx) => {
    const card = document.createElement("div");
    card.className = "gallery-card";
    card.onclick = () => openGalleryModal(idx);
    card.innerHTML = `
      <img class="gallery-thumb" src="${img.url}" loading="lazy">
      <div class="gallery-info">
        <div class="gallery-name" title="${img.name}">${img.name}</div>
        <div class="gallery-meta">
          <span>${img.size_kb} KB</span>
          <span style="color:#60A5FA;">点击查看高清大图</span>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

function filterGallery() {
  const q = document.getElementById("gallery-search").value.toLowerCase();
  const filtered = allGalleryImages.filter(img => img.name.toLowerCase().includes(q));
  renderGalleryGrid(filtered);
}

function openGalleryModal(idx) {
  currentGalleryIdx = idx;
  const img = allGalleryImages[idx];
  if (!img) return;

  const modal = document.getElementById("gallery-modal");
  document.getElementById("gallery-modal-img").src = img.url;
  document.getElementById("gallery-modal-title").innerText = `(${idx + 1}/${allGalleryImages.length}) ${img.name}`;
  modal.style.display = "flex";
}

function galleryNavigate(delta) {
  const newIdx = currentGalleryIdx + delta;
  if (newIdx >= 0 && newIdx < allGalleryImages.length) {
    openGalleryModal(newIdx);
  }
}

// Keyboard arrow navigation for gallery
document.addEventListener("keydown", (e) => {
  const modal = document.getElementById("gallery-modal");
  if (modal && modal.style.display === "flex") {
    if (e.key === "ArrowLeft") galleryNavigate(-1);
    if (e.key === "ArrowRight") galleryNavigate(1);
    if (e.key === "Escape") closeModal("gallery-modal");
  }
});

// -------------------------------------------------------------
// Tab 5: Dataset Generation Controls
// -------------------------------------------------------------
async function startDatasetGeneration() {
  const task = document.getElementById("gen-task").value;
  const s3d_dir = document.getElementById("gen-s3d-dir").value;
  const out_dir = document.getElementById("gen-out-dir").value;
  const max_scenes = document.getElementById("gen-max-scenes").value;
  const num_workers = parseInt(document.getElementById("gen-workers").value);

  try {
    const res = await fetch("/api/dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task, s3d_dir, out_dir, max_scenes: max_scenes ? parseInt(max_scenes) : null, num_workers
      }),
    });
    const result = await res.json();
    if (result.error) {
      alert(`生成启动失败: ${result.message}`);
    } else {
      appendTerminalLine(`\n[+] 数据集生成任务启动: PID ${result.task.pid}\n`);
    }
  } catch (e) {
    alert(`生成请求异常: ${e}`);
  }
}
