/**
 * Semantic LSD Matching Web Management Interface
 * Frontend controller: WebSocket terminal streaming, Chart rendering, API integrations.
 */

// Global State
let currentTab = "history";
let currentSelectedRun = null;
let currentRunDetails = null;
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
  initModelDocs();
});

// -------------------------------------------------------------
// Tabs Navigation (Top Level)
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
      const targetPane = document.getElementById(`tab-${target}`);
      if (targetPane) targetPane.classList.add("active");

      // Tab specific refresh
      if (target === "history") {
        const detailView = document.getElementById("history-detail-view");
        const listView = document.getElementById("history-list-view");
        if (detailView) detailView.style.display = "none";
        if (listView) listView.style.display = "block";
        loadDashboard();
      } else if (target === "train") {
        loadDatasets();
      }
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

  socket.onerror = (err) => {
    console.warn("[WS] Socket error:", err);
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
    if (dot) dot.className = "dot dot-busy";
    if (text) text.innerText = `运行中: ${status.active_task.task_type} (${status.active_task.task_id})`;
  } else {
    if (dot) dot.className = "dot dot-idle";
    if (text) text.innerText = "系统空闲 (Ready)";
  }

  const res = status.resources || {};
  const gpu = (res.gpu && res.gpu.available) ? res.gpu : status.gpu;

  if (gpu && gpu.available) {
    if (gpuBadge) gpuBadge.innerText = `GPU: ${gpu.memory_used_mb}MB / ${gpu.memory_total_mb}MB (${gpu.memory_pct}%) | ${gpu.temperature_c}°C`;
  } else {
    if (gpuBadge) gpuBadge.innerText = "CPU 模式";
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
// Tab 1: 实验历史数据 & 实验复盘看板
// -------------------------------------------------------------
async function loadDashboard() {
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    allRuns = data.runs || [];

    // Summary Cards
    const statTotalEl = document.getElementById("stat-total-runs");
    if (statTotalEl) statTotalEl.innerText = allRuns.length;

    let bestRaw = 0;
    let bestSem = 0;
    allRuns.forEach(r => {
      if (r.test_raw_f1 && r.test_raw_f1 > bestRaw) bestRaw = r.test_raw_f1;
      if (r.test_sem_f1 && r.test_sem_f1 > bestSem) bestSem = r.test_sem_f1;
    });
    const bestRawEl = document.getElementById("stat-best-raw-f1");
    const bestSemEl = document.getElementById("stat-best-sem-f1");
    if (bestRawEl) bestRawEl.innerText = bestRaw ? `${bestRaw.toFixed(1)}%` : "-";
    if (bestSemEl) bestSemEl.innerText = bestSem ? `${bestSem.toFixed(1)}%` : "-";

    // Table rows
    const tbody = document.getElementById("runs-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    // Update log source select dropdown in terminal
    const logSelect = document.getElementById("log-source-select");
    if (logSelect) {
      const cur = logSelect.value;
      logSelect.innerHTML = `<option value="live">实时任务输出 (Live)</option>`;
      allRuns.forEach(r => {
        const tagText = r.run_tag || r.run_name;
        logSelect.innerHTML += `<option value="${r.run_name}">历史日志: ${tagText}</option>`;
      });
      if (cur) logSelect.value = cur;
    }

    allRuns.forEach((r) => {
      const tr = document.createElement("tr");
      const bestF1Str = r.val_f1 ? `${r.val_f1.toFixed(1)}% (Ep ${r.best_epoch})` : "-";
      const rawF1Str = r.test_raw_f1 ? `<b style="color:#60A5FA;">${r.test_raw_f1.toFixed(1)}%</b>` : "-";
      const semF1Str = r.test_sem_f1 ? `<b style="color:#34D399;">${r.test_sem_f1.toFixed(1)}%</b>` : "-";
      const displayTag = r.run_tag || r.run_name;

      tr.innerHTML = `
        <td><input type="checkbox" class="run-select-box" value="${r.run_name}" onchange="updateCompareBtn()"></td>
        <td style="max-width: 440px;">
          <div class="run-tag-pill" title="${displayTag}">${displayTag}</div>
        </td>
        <td><span class="badge badge-purple">${r.model_type}</span></td>
        <td style="color: #9CA3AF; font-size: 13px;">${r.start_time}</td>
        <td>${r.epochs}</td>
        <td>${bestF1Str}</td>
        <td>${rawF1Str}</td>
        <td>${semF1Str}</td>
        <td>
          <button class="btn btn-primary btn-sm" onclick="openReviewTab('${r.run_name}')">新标签页打开复盘 ↗</button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    if (allRuns.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:#9CA3AF; padding: 24px;">暂无历史实验记录，请前往【启动新的训练】开始第一次实验！</td></tr>`;
    }
    updateCompareBtn();
  } catch (e) {
    console.error("Failed to load runs:", e);
    const tbody = document.getElementById("runs-table-body");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:#EF4444; padding: 24px;">加载历史实验失败: ${e.message || e}</td></tr>`;
    }
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
    const res = await fetch(`/api/runs/compare?run1=${encodeURIComponent(selected[0])}&run2=${encodeURIComponent(selected[1])}`);
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

// -------------------------------------------------------------
// Selected Run Review Workspace (二级复盘与评测工作台)
// -------------------------------------------------------------
async function openRunDetailWorkspace(runName) {
  currentSelectedRun = runName;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runName)}`);
    const data = await res.json();
    if (data.error) {
      alert(`无法加载实验详情: ${data.message}`);
      return;
    }
    currentRunDetails = data;

    // Header & Tag
    const tagTitle = document.getElementById("detail-run-tag");
    if (tagTitle) tagTitle.innerText = data.run_tag || data.run_name;

    const statusPill = document.getElementById("detail-status-pill");
    const isDone = data.config && data.config.test_benchmark != null;
    if (statusPill) {
      statusPill.innerText = isDone ? "Done ★ 已完成" : "Recorded 已记录";
      statusPill.className = isDone ? "badge badge-success" : "badge badge-blue";
    }

    // Badges
    const modelBadge = document.getElementById("detail-badge-model");
    if (modelBadge) modelBadge.innerText = `架构: ${data.model_type}`;

    const epochsBadge = document.getElementById("detail-badge-epochs");
    const hparams = data.config?.hyperparameters || {};
    if (epochsBadge) epochsBadge.innerText = `轮数: ${hparams.epochs || data.history?.length || "-"} 轮`;

    const bestVal = data.config?.best_validation || {};
    const valBadge = document.getElementById("detail-badge-val");
    if (valBadge) valBadge.innerText = `最佳验证 F1: ${bestVal.f1 ? bestVal.f1.toFixed(1) + "%" : "-"}`;

    const testBench = data.config?.test_benchmark || {};
    const rawF1 = testBench.raw_visual?.f1;
    const semF1 = testBench.semantic_constrained?.f1;
    const rawBadge = document.getElementById("detail-badge-raw");
    if (rawBadge) rawBadge.innerText = `盲测 Raw F1: ${rawF1 ? rawF1.toFixed(1) + "%" : "-"}`;
    const semBadge = document.getElementById("detail-badge-sem");
    if (semBadge) semBadge.innerText = `语义约束 F1: ${semF1 ? semF1.toFixed(1) + "%" : "-"}`;

    const dirBadge = document.getElementById("detail-badge-dir");
    if (dirBadge) {
      const relDir = data.run_dir.includes("checkpoints/") ? data.run_dir.split("checkpoints/")[1] : data.run_dir;
      dirBadge.innerText = `目录: ${relDir}`;
    }

    // Subpane 1: Curves & Report
    const curvesImg = document.getElementById("detail-curves-img");
    const noCurves = document.getElementById("detail-no-curves");
    if (data.curves_url) {
      if (curvesImg) {
        curvesImg.src = data.curves_url;
        curvesImg.style.display = "block";
      }
      if (noCurves) noCurves.style.display = "none";
    } else {
      if (curvesImg) curvesImg.style.display = "none";
      if (noCurves) noCurves.style.display = "block";
    }

    const reportContainer = document.getElementById("detail-report-container");
    if (reportContainer) {
      if (data.report_markdown) {
        reportContainer.innerHTML = simpleMarkdownRender(data.report_markdown);
      } else {
        reportContainer.innerHTML = "<p style='color:#9CA3AF;'>该实验目录尚未生成 report.md 分析报告文件。</p>";
      }
    }

    // Checkpoints in this run
    const ckptList = document.getElementById("detail-checkpoints-list");
    if (ckptList) {
      ckptList.innerHTML = "";
      if (data.checkpoints && data.checkpoints.length > 0) {
        data.checkpoints.forEach(c => {
          const star = c.is_best ? "★ 最佳模型 " : "";
          const badgeClass = c.is_best ? "badge badge-success" : "badge badge-blue";
          ckptList.innerHTML += `
            <div style="display:flex; justify-content:space-between; align-items:center; background:#1F2937; padding:8px 12px; border-radius:6px; border:1px solid #374151;">
              <div>
                <b style="color:#F3F4F6; font-size:13px;">${c.name}</b>
                <span class="${badgeClass}" style="margin-left:8px; font-size:11px;">${star}${c.size_mb} MB</span>
              </div>
              <button class="btn btn-secondary btn-sm" onclick="setEvalCheckpoint('${c.rel_path}')">选中评测</button>
            </div>
          `;
        });
      } else {
        ckptList.innerHTML = "<span style='color:#9CA3AF;'>暂无权重文件</span>";
      }
    }

    // Subpane 2: Setup evaluation with current run's best_model
    const bestCkpt = data.checkpoints?.find(c => c.is_best) || data.checkpoints?.[0];
    const bestPath = bestCkpt ? bestCkpt.rel_path : (data.config?.checkpoints?.best_checkpoint || "");
    await loadCheckpoints(bestPath);

    // Set dataset in eval if available
    const dsDir = data.config?.dataset?.dataset_dir || data.config?.hyperparameters?.all_cli_args?.data_dir;
    if (dsDir) {
      const evalDataSelect = document.getElementById("eval-data-dir");
      if (evalDataSelect) {
        const dsName = dsDir.split("/").pop();
        for (let i = 0; i < evalDataSelect.options.length; i++) {
          if (evalDataSelect.options[i].value.includes(dsName)) {
            evalDataSelect.selectedIndex = i;
            break;
          }
        }
      }
    }

    // Subpane 3: Gallery
    await loadGallery(runName);

    // Switch view
    document.getElementById("history-list-view").style.display = "none";
    document.getElementById("history-detail-view").style.display = "block";
    switchDetailSubTab("report");

    // Scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (e) {
    console.error("Open run workspace error:", e);
    alert(`加载实验详情异常: ${e}`);
  }
}

function backToHistoryList() {
  document.getElementById("history-detail-view").style.display = "none";
  document.getElementById("history-list-view").style.display = "block";
  loadDashboard();
}

function switchDetailSubTab(subtab) {
  document.querySelectorAll(".sub-tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.subtab === subtab);
  });
  document.querySelectorAll(".sub-tab-pane").forEach(pane => {
    pane.classList.toggle("active", pane.id === `subpane-${subtab}`);
  });
  if (subtab === "gallery" && currentSelectedRun) {
    loadGallery(currentSelectedRun);
  }
}

function reloadCurrentRunGallery() {
  if (currentSelectedRun) {
    loadGallery(currentSelectedRun);
  } else {
    loadGallery();
  }
}

function openCurvesModal() {
  const modal = document.getElementById("curves-modal");
  const curvesImg = document.getElementById("detail-curves-img");
  const modalImg = document.getElementById("curves-modal-img");
  if (modal && curvesImg && modalImg) {
    modalImg.src = curvesImg.src;
    modal.style.display = "flex";
  }
}

function setEvalCheckpoint(relPath) {
  const select = document.getElementById("eval-checkpoint");
  if (select) {
    for (let i = 0; i < select.options.length; i++) {
      if (select.options[i].value === relPath || select.options[i].value.endsWith(relPath)) {
        select.selectedIndex = i;
        break;
      }
    }
  }
  switchDetailSubTab("eval");
}

function toggleDatasetFactory() {
  const body = document.getElementById("dataset-factory-body");
  if (body) {
    body.style.display = body.style.display === "none" ? "block" : "none";
  }
}

async function viewRunDetails(runName) {
  openRunDetailWorkspace(runName);
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

function openReviewTab(runName) {
  const url = "/review?run=" + encodeURIComponent(runName);
  window.open(url, "_blank");
}

async function previewCurrentModel() {
  const modelEl = document.getElementById("train-model");
  const pyramidEl = document.getElementById("train-pyramid");
  const model = modelEl ? modelEl.value : "xfeat_line_matcher";
  const pyramid = pyramidEl ? pyramidEl.checked : true;
  const modal = document.getElementById("model-preview-modal");
  if (!modal) return;

  try {
    const res = await fetch(`/api/model/preview?model=${encodeURIComponent(model)}&pyramid=${pyramid}&d_model=128`);
    const data = await res.json();
    if (data.error) {
      alert("获取模型预览失败: " + data.message);
      return;
    }

    const titleEl = document.getElementById("preview-model-name");
    if (titleEl) titleEl.innerText = `📐 ${data.name} (${data.model_type})`;

    const taglineEl = document.getElementById("preview-model-tagline");
    if (taglineEl) taglineEl.innerText = data.tagline || "";

    const totalParamsEl = document.getElementById("preview-total-params");
    if (totalParamsEl) totalParamsEl.innerText = data.params_total || "-";

    const trainableParamsEl = document.getElementById("preview-trainable-params");
    if (trainableParamsEl) trainableParamsEl.innerText = data.trainable_params || "-";

    // Shapes
    const shapesTbody = document.getElementById("preview-shapes-tbody");
    if (shapesTbody) {
      shapesTbody.innerHTML = "";
      (data.input_shapes || []).forEach(s => {
        shapesTbody.innerHTML += `
          <tr>
            <td><b style="color:#F3F4F6;">${s.name}</b></td>
            <td><code style="color:#60A5FA; background:#1F2937; padding:2px 6px; border-radius:4px; font-weight:600;">${s.shape}</code></td>
            <td style="color:#9CA3AF;">${s.desc}</td>
          </tr>
        `;
      });
    }

    // Stages
    const stagesTbody = document.getElementById("preview-stages-tbody");
    if (stagesTbody) {
      stagesTbody.innerHTML = "";
      (data.pipeline_stages || []).forEach(st => {
        stagesTbody.innerHTML += `
          <tr>
            <td><b style="color:#A78BFA;">${st.stage}</b></td>
            <td><span class="badge badge-blue">${st.dim}</span></td>
            <td style="color:#D1D5DB;">${st.desc}</td>
          </tr>
        `;
      });
    }

    // Code
    const codeEl = document.getElementById("preview-pytorch-code");
    if (codeEl) codeEl.innerText = data.pytorch_repr || "// 暂无代码表示";

    modal.style.display = "flex";
  } catch (e) {
    alert("请求模型预览发生异常: " + e);
  }
}

function copyModelPytorchRepr() {
  const codeEl = document.getElementById("preview-pytorch-code");
  if (!codeEl) return;
  navigator.clipboard.writeText(codeEl.innerText).then(() => {
    alert("已成功复制 PyTorch 模型结构文本到剪贴板！");
  }).catch(() => {
    alert("复制失败，请手动选取复制。");
  });
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

// -------------------------------------------------------------
// Data Ratio Slider & Input Two-Way Sync
// -------------------------------------------------------------
function syncDataRatioFromSlider(val) {
  const fVal = Math.max(0.01, Math.min(1.0, parseFloat(val) || 1.0));
  const input = document.getElementById("train-data-ratio-input");
  if (input) input.value = fVal.toFixed(2);
}

function syncDataRatioFromInput(val) {
  let fVal = parseFloat(val);
  if (isNaN(fVal)) return;
  fVal = Math.max(0.01, Math.min(1.0, fVal));
  const slider = document.getElementById("train-data-ratio-slider");
  if (slider) slider.value = fVal;
}

async function startTraining() {
  const model = document.getElementById("train-model").value;
  const data_dir = document.getElementById("train-data-dir").value;
  const epochs = parseInt(document.getElementById("train-epochs").value);
  const lr = parseFloat(document.getElementById("train-lr").value);
  const dustbin_weight = parseFloat(document.getElementById("train-dustbin").value);
  const motion_weight = parseFloat(document.getElementById("train-motion").value);
  const min_area_ratio = parseFloat(document.getElementById("train-area-ratio").value);
  const data_ratio_el = document.getElementById("train-data-ratio-input");
  const data_ratio = data_ratio_el ? Math.max(0.01, Math.min(1.0, parseFloat(data_ratio_el.value) || 1.0)) : 1.0;
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
  const ratioLabel = data_ratio < 1.0 ? `, ratio=${Math.round(data_ratio * 100)}%` : "";
  appendTerminalLine(`[+] 正在请求启动训练: model=${model}, data=${data_dir}${ratioLabel}, epochs=${epochs}, lr=${lr}...`);

  liveLossHistory = [];
  liveF1History = [];
  renderLiveChart();

  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model, data_dir, epochs, lr, dustbin_weight, motion_weight, min_area_ratio, data_ratio, pyramid, amp
      }),
    });
    const result = await res.json();
    if (result.error) {
      alert(`启动失败: ${result.message}`);
      appendTerminalLine(`\n[!] 启动失败: ${result.message}\n`);
    } else {
      appendTerminalLine(`\n[✓] 训练子进程成功启动! PID: ${result.task.pid}\n[*] 实时输出监听中...\n[*] 正在为您在新标签页中打开实时监控工作台...\n`);
      // Open standalone live training monitor in a new browser tab
      window.open("/train_live", "_blank");
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
async function loadCheckpoints(selectedRunCkpt = null) {
  try {
    const res = await fetch("/api/checkpoints");
    const data = await res.json();
    const select = document.getElementById("eval-checkpoint");
    if (!select) return;

    select.innerHTML = "";
    (data.checkpoints || []).forEach(c => {
      const star = c.is_best ? "★ " : "";
      const isSel = selectedRunCkpt && (c.rel_path === selectedRunCkpt || c.path === selectedRunCkpt || c.name === selectedRunCkpt);
      select.innerHTML += `<option value="${c.rel_path}" ${isSel ? "selected" : ""}>${star}${c.folder}/${c.name} (${c.size_mb} MB)</option>`;
    });

    if (selectedRunCkpt) {
      for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].value.includes(selectedRunCkpt) || select.options[i].text.includes(selectedRunCkpt)) {
          select.selectedIndex = i;
          break;
        }
      }
    }
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

  const btnEval = document.querySelector("#subpane-eval .btn-primary");
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
// Subpane 3: Results Visualization Gallery
// -------------------------------------------------------------
async function loadGallery(runName = null) {
  try {
    const url = runName ? `/api/gallery?run=${encodeURIComponent(runName)}` : "/api/gallery";
    const res = await fetch(url);
    const data = await res.json();
    allGalleryImages = data.images || [];

    const countEl = document.getElementById("gallery-count");
    if (countEl) countEl.innerText = `${allGalleryImages.length} 张样本图`;

    renderGalleryGrid(allGalleryImages);
  } catch (e) {
    console.error("Failed to load gallery:", e);
  }
}

function renderGalleryGrid(images) {
  const grid = document.getElementById("gallery-grid");
  if (!grid) return;
  grid.innerHTML = "";

  if (images.length === 0) {
    grid.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 48px; text-align: center; background: rgba(31, 41, 55, 0.4); border-radius: 8px; border: 1px dashed #374151;">
        <div style="font-size: 32px; margin-bottom: 12px;">🖼️</div>
        <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 16px;">当前实验尚未生成匹配样张</div>
        <button class="btn btn-primary btn-sm" onclick="switchDetailSubTab('eval')">前往【评测基准】一键运行评测并生成样张 ▶</button>
      </div>
    `;
    return;
  }

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
          <span style="color:#60A5FA;">点击查看大图</span>
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

// -------------------------------------------------------------
// Tab 6: Interactive Models Guide & Architecture Docs
// -------------------------------------------------------------
let currentDocModel = "xfeat_line_matcher";
let currentArchStages = [];
let currentCodeSnippetKey = "python_api";

const MODEL_DOCS_DATA = {
  xfeat_line_matcher: {
    badge: "多模态直线+目标区域混合网络 (SOTA)",
    badgeClass: "badge-purple",
    title: "XFeatLineMatcher: Multi-Scale Pyramid & 5D Line Geometry Matcher",
    subtitle: "结合 XFeat FPN 多尺度视觉特征金字塔与 5D 亚像素直线几何编码，通过两层图交叉注意力机制与 Sinkhorn 双随机最优传输实现鲁棒配对。",
    metric: "F1: 79.22%",
    metricLabel: "Structured3D 真实室内盲测基准",
    stages: [
      {
        num: "STAGE 01",
        title: "多模态输入解析",
        desc: "RGB 图像切片与 2D/3D LSD 直线段",
        shapes: "Patches: (B, M, 3, 112, 112) | BBoxes: (B, M, 4) | Lines: (B, M, K, 5)",
        code: "datasets/semantic_region_dataset.py",
        detail: `<strong>多模态输入构造机制：</strong>
<ul style="margin: 8px 0 0 20px; line-height: 1.6;">
  <li><strong>候选目标切片 (Region Patches)</strong>：通过实例掩膜最小外接矩形自适应扩展裁剪并双线性插值到固定 112x112 分辨率。</li>
  <li><strong>关联直线提取 (Line Association)</strong>：利用空间投票法检测落在各个目标 Patch 内的 LSD 连续直线段，每 Patch 截取至多 K=8 根最显著直线。</li>
  <li><strong>5D 几何参数归一化</strong>：将直线端点转换为尺度不变向量: <code>[x_c/W, y_c/H, L/max(W,H), sin(2θ), cos(2θ)]</code>。</li>
</ul>`
      },
      {
        num: "STAGE 02",
        title: "XFeat FPN 金字塔",
        desc: "跨尺度多级视觉特征抽取与池化",
        shapes: "输入 (B, M, 3, 112, 112) ➔ P1(64D, 1/4) + P2(64D, 1/8) + P3(64D, 1/16) ➔ (B, M, 192)",
        code: "models/object_matcher/xfeat_line_matcher.py#L90",
        detail: `<strong>Multi-Scale Feature Pyramid Network (FPN)：</strong>
<ul style="margin: 8px 0 0 20px; line-height: 1.6;">
  <li>基础骨干采用轻量级全卷积网络，输出 1/4, 1/8, 1/16 三级空间特征图。</li>
  <li>通过 1x1 侧边卷积与最近邻上采样逐级融合高层语义与底层几何边缘纹理。</li>
  <li>对各层级执行 Masked Global Average Pooling，拼接形成 <strong>192 维多尺度视觉描述子</strong>。</li>
</ul>`
      },
      {
        num: "STAGE 03",
        title: "5D 线段几何编码",
        desc: "PatchLineGeometryEncoder 与跨模态对齐",
        shapes: "Lines: (B, M, K, 5) ➔ MLP(64D) ➔ Masked Mean Pooling ➔ Line Token (B, M, 64)",
        code: "models/object_matcher/xfeat_line_matcher.py#L35",
        detail: `<strong>线段几何编码与 Null Token 机制：</strong>
<ul style="margin: 8px 0 0 20px; line-height: 1.6;">
  <li><strong>线段特征投影</strong>：对 5D 几何向量施加两层非线性 MLP 映射至 64 维空间。</li>
  <li><strong>掩膜均值池化 (Masked Pooling)</strong>：对各 Patch 内的有效直线特征取均值，消除直线排列顺序影响。</li>
  <li><strong>Null Token 退化保护</strong>：若 Patch 内部未检测到任何直线，自动回退到可学习的 <code>null_line_token</code>，确保梯度平滑不断裂。</li>
  <li><strong>多模态融合层</strong>：<code>Linear(192 + 64, 128) + LayerNorm + GELU</code> 将视觉与直线联合映射为 128 维 Joint Token。</li>
</ul>`
      },
      {
        num: "STAGE 04",
        title: "图注意力网络 (GNN)",
        desc: "自注意力结构内聚 + 跨视角注意力语义传递",
        shapes: "Tokens0, Tokens1: (B, M, 128) ➔ 2-Layer Cross & Self Multi-Head Attention",
        code: "models/object_matcher/xfeat_object_matcher.py#L140",
        detail: `<strong>双视角注意力信息流动图：</strong>
<ul style="margin: 8px 0 0 20px; line-height: 1.6;">
  <li><strong>自注意力 (Self-Attention)</strong>：同视角内部建模空间布局几何拓扑关系，让邻近物体感知相对位置。</li>
  <li><strong>跨注意力 (Cross-Attention)</strong>：跨视角在 View 0 与 View 1 间进行相似度寻址，解决视角旋转、形变与遮挡下的语义关联。</li>
  <li>采用 4-Head 缩放点积注意力，配备残差连接与 LayerNorm 维持数值稳定性。</li>
</ul>`
      },
      {
        num: "STAGE 05",
        title: "双随机最优传输",
        desc: "Sinkhorn 迭代求解 + 扩展垃圾桶拒识",
        shapes: "得分矩阵 (M0+1, M1+1) ➔ 20次 Sinkhorn 迭代 ➔ 双随机匹配概率矩阵 P",
        code: "models/object_matcher/loss.py",
        detail: `<strong>最优传输与出画落单拒识：</strong>
<ul style="margin: 8px 0 0 20px; line-height: 1.6;">
  <li>在相似度矩阵末尾拼接可学习的<strong>垃圾桶参数行/列 (Dustbin)</strong>，吸收出画、遮挡或无对应的目标。</li>
  <li>基于对数域 Log-Sinkhorn 算法执行 20 次交替行列归一化，逼近双随机投影矩阵。</li>
  <li><strong>位移场共识损失 (Motion Consensus Loss)</strong>：对几何位移偏离全局中位数的假阳性误匹配施加铰链惩罚（Hinge Margin 0.15）。</li>
</ul>`
      }
    ]
  },
  mask_line: {
    badge: "6通道连续几何场引导模型",
    badgeClass: "badge-blue",
    title: "MaskLineFeatMatchNet: Continuous EDT + AFM Geometry Prompted Line Matcher",
    subtitle: "将线段离散端点转化为 2 通道欧氏距离场 (EDT) 与 4 通道方向角度场 (AFM)，以 6 通道稠密张量输入 CNN 提取全局拓扑特征。",
    metric: "F1: 68.50%",
    metricLabel: "无纹理纯几何场景直线基准",
    stages: [
      {
        num: "STAGE 01",
        title: "直线段几何栅格化",
        desc: "OpenCV LSD 直线提取与多边形掩膜生成",
        shapes: "RGB 图像 (3, H, W) + 线段集合 {(x1, y1, x2, y2)}",
        code: "models/mask_line_matcher/geometric_field.py",
        detail: "将离散线段转为空间布尔掩膜，作为距离变换的初始轮廓种子点。"
      },
      {
        num: "STAGE 02",
        title: "6通道几何场编码",
        desc: "欧氏距离场 (EDT) + 朝向向量场 (AFM)",
        shapes: "输出 6 通道张量 (6, H, W) = RGB(3) + EDT(1) + AFM_cos(1) + AFM_sin(1)",
        code: "models/mask_line_matcher/geometric_field.py#L40",
        detail: "通过快速距离变换计算图像每个像素到最近直线的精确距离与线段朝向正余弦，构建平滑连续的几何引导场。"
      },
      {
        num: "STAGE 03",
        title: "几何感知卷积骨干",
        desc: "首层 6 通道卷积核直接提取几何与纹理混合特征",
        shapes: "输入 (B, 6, H, W) ➔ 浅层 ResNet ➔ 特征图 (B, 64, H/4, W/4)",
        code: "models/mask_line_matcher/mask_line_matcher.py#L45",
        detail: "修改首层卷积权重适配 6 通道，让网络低层同时感知颜色梯度与直线几何边缘分布。"
      },
      {
        num: "STAGE 04",
        title: "规范最优传输矩阵",
        desc: "固定规范尺寸 (255, 255) 置换等变矩阵映射",
        shapes: "输出统一矩阵 (B, 255, 255)，未激活行/列严格置零",
        code: "models/mask_line_matcher/mask_line_matcher.py#L120",
        detail: "严格满足置换等变性（Permutation Equivariance），即交换输入线段顺序时，输出矩阵对应行/列严格同构置换。"
      }
    ]
  },
  line_transformer: {
    badge: "基线线段图注意力模型",
    badgeClass: "badge-warning",
    title: "LineMatcherNet: Point Sampling & Graph Transformer Line Matcher",
    subtitle: "在检测出的线段上均匀插值离散采样点，抽取局部视觉特征并拼接端点 4D 几何坐标，经 Transformer 自注意力配对。",
    metric: "F1: 62.30%",
    metricLabel: "基础线特征匹配基线",
    stages: [
      {
        num: "STAGE 01",
        title: "线段端点与均匀采样",
        desc: "沿线等间距采样 5 个特征点",
        shapes: "线段输入 (N, 4) ➔ 采样坐标点 (N, 5, 2)",
        code: "models/line_transformer/line_transformer.py",
        detail: "沿线段中心及两端等距采集局部像素点，规避单点特征漂移。"
      },
      {
        num: "STAGE 02",
        title: "视觉特征网格采样",
        desc: "Grid Sample 双线性插值视觉特征",
        shapes: "Feature Map (B, C, H, W) ➔ LineVisualSampler ➔ (B, N, C)",
        code: "models/line_transformer/line_transformer.py#L80",
        detail: "在 CNN 特征图上通过双线性插值直接提取采样点特征，并通过 1D 卷积聚合为单线视觉向量。"
      },
      {
        num: "STAGE 03",
        title: "自注意力图推理",
        desc: "全图拓扑关联与交叉注意力匹配",
        shapes: "Tokens (B, N, 128) ➔ Transformer Layers ➔ 相似度矩阵 (B, N0, N1)",
        code: "models/line_transformer/line_transformer.py#L160",
        detail: "利用标准 Transformer 建模线段间的平行、共线与正交关系，输出互为最近邻的双向配对。"
      }
    ]
  }
};

const CODE_SNIPPETS = {
  python_api: {
    title: "Python API: 5行代码执行多模态匹配推理",
    code: `# 导入统一模型与评估工厂
from evaluation import ModelFactory
from datasets import predict_bipartite_matches

# 1. 自动探查并加载模型检查点 (自动识别金字塔与几何编码)
model, info, ckpt = ModelFactory.create_from_checkpoint(
    "checkpoints/xfeat_line_matcher/latest",
    device="cuda"
)

# 2. 前向推理双视角候选切片与直线几何特征
# patches: (1, M, 3, 112, 112), patch_lines: (1, M, K, 5)
out = model(patches0=patches0, patches1=patches1, patch_lines0=lines0, patch_lines1=lines1)

# 3. 提取双向最佳匹配与出画落单垃圾桶
matches, unmatched_0, unmatched_1 = predict_bipartite_matches(out["prob_matrix"][0], thresh=0.2)
print(f"匹配成功: {len(matches)} 对目标, 拒识落单: {len(unmatched_0)} (View0) / {len(unmatched_1)} (View1)")`
  },
  train_cli: {
    title: "训练 CLI: 开启 FPN 金字塔、5D 直线编码与 AMP 极速训练",
    code: `# 推荐训练命令 (Structured3D 真实室内场景数据集)
python -u train.py \\
    --model xfeat_line_matcher \\
    --data-dir data/xfeat_line_matcher \\
    --epochs 15 \\
    --lr 5e-4 \\
    --dustbin-weight 0.1 \\
    --motion-weight 0.1 \\
    --min-area-ratio 0.40 \\
    --pyramid \\
    --amp \\
    --save-interval 5

# 提示: 训练将自动生成 curves.png, metrics.csv, report.md 及 config.json`
  },
  eval_cli: {
    title: "评测 CLI: 盲测基准评分与诊断大图生成",
    code: `# 1. 评估最新实验运行 (自动加载最佳权重并在盲测集打分)
python eval.py \\
    --checkpoint checkpoints/xfeat_line_matcher/latest \\
    --split test \\
    --min-area-ratio 0.50 \\
    --num-vis 10 \\
    --out-dir outputs/semantic_matching_vis

# 2. 导出成果物包含:
#   - eval_report.md (Markdown 详报)
#   - eval_results.json (定量指标 JSON)
#   - eval_sample_*.jpg (全景双排可视化诊断大图)`
  },
  dataset_cli: {
    title: "数据生成 CLI: 从 Structured3D 提取 CAD 直线真值与切片",
    code: `# 从原始 Structured3D 场景提取 100% CAD 直线真值与 Patch 5D 几何数据
python generate_dataset.py \\
    --task structured3d_lines \\
    --s3d-dir Structured3D_perspective_full_00/Structured3D \\
    --out-dir data/xfeat_line_matcher \\
    --max-scenes 50 \\
    --num-workers 8 \\
    --min-matches 3 \\
    --min-area-ratio 0.40`
  },
  review_cli: {
    title: "实验复盘 CLI: 命令行多实验对比与曲线渲染",
    code: `# 1. 列出历史所有实验记录
python review_runs.py --list

# 2. 详细复盘指定单次实验并输出 Markdown
python review_runs.py --run 20261026_145058

# 3. 横向对比两个不同权重实验 (如比较不同 dustbin-weight)
python review_runs.py --compare 20261023_233349 20261026_145058`
  }
};

function initModelDocs() {
  switchModelDoc("xfeat_line_matcher");
  updateCalculator();
  switchCodeSnippet("python_api");
}

function switchModelDoc(modelKey) {
  currentDocModel = modelKey;
  const data = MODEL_DOCS_DATA[modelKey];
  if (!data) return;

  // Update pills active status
  document.querySelectorAll(".model-pill").forEach(p => p.classList.remove("active"));
  const btnId = modelKey === "xfeat_line_matcher" ? "btn-doc-xfeat-line" :
                modelKey === "mask_line" ? "btn-doc-mask-line" : "btn-doc-line-transformer";
  const activeBtn = document.getElementById(btnId);
  if (activeBtn) activeBtn.classList.add("active");

  // Update headers
  const badgeEl = document.getElementById("doc-model-badge");
  if (badgeEl) {
    badgeEl.innerText = data.badge;
    badgeEl.className = `badge ${data.badgeClass}`;
  }

  const titleEl = document.getElementById("doc-model-title");
  if (titleEl) titleEl.innerText = data.title;

  const subEl = document.getElementById("doc-model-subtitle");
  if (subEl) subEl.innerText = data.subtitle;

  const metricEl = document.getElementById("doc-model-metric");
  if (metricEl) metricEl.innerText = data.metric;

  const metricLbl = document.getElementById("doc-model-metric-label");
  if (metricLbl) metricLbl.innerText = data.metricLabel;

  // Render stage nodes
  currentArchStages = data.stages;
  const stagesContainer = document.getElementById("arch-flow-stages");
  if (stagesContainer) {
    stagesContainer.innerHTML = data.stages.map((stg, idx) => `
      <div class="arch-step-node ${idx === 0 ? 'active' : ''}" onclick="selectArchStage(${idx})" id="arch-node-${idx}">
        <div class="step-num">${stg.num}</div>
        <div class="step-title">${stg.title}</div>
        <div class="step-desc">${stg.desc}</div>
      </div>
    `).join("");
  }

  selectArchStage(0);
}

function selectArchStage(idx) {
  if (!currentArchStages || !currentArchStages[idx]) return;
  const stage = currentArchStages[idx];

  // Update active card
  document.querySelectorAll(".arch-step-node").forEach(n => n.classList.remove("active"));
  const activeNode = document.getElementById(`arch-node-${idx}`);
  if (activeNode) activeNode.classList.add("active");

  // Render inspector box
  const panel = document.getElementById("arch-inspector-panel");
  if (panel) {
    panel.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; border-bottom: 1px solid #233047; padding-bottom:8px;">
        <div style="font-weight:700; color:#60A5FA; font-size:14px;">
          ✦ ${stage.num}: ${stage.title} 深度解析
        </div>
        <div style="font-size:12px; font-family:monospace; color:#A78BFA; background:#1A2338; padding:3px 8px; border-radius:4px;">
          源码定义: ${stage.code}
        </div>
      </div>
      <div style="margin-bottom: 10px;">
        <span style="font-size:11px; font-weight:700; color:#9CA3AF; text-transform:uppercase;">张量流动规格 (Tensor Dimensions):</span>
        <div style="font-family:monospace; font-size:12px; color:#FBBF24; background:#080C14; padding:6px 10px; border-radius:4px; margin-top:4px;">
          ${stage.shapes}
        </div>
      </div>
      <div style="font-size:13px; color:#D1D5DB; line-height:1.6;">
        ${stage.detail}
      </div>
    `;
  }
}

function updateCalculator() {
  const dmodelEl = document.getElementById("calc-dmodel");
  if (!dmodelEl) return;
  const d_model = parseInt(dmodelEl.value) || 128;
  const num_layers = parseInt(document.getElementById("calc-layers")?.value) || 2;
  const nhead = parseInt(document.getElementById("calc-nhead")?.value) || 4;
  const patches = parseInt(document.getElementById("calc-patches")?.value) || 24;
  const lines = parseInt(document.getElementById("calc-lines")?.value) || 8;

  // Update slider value labels
  const lblDmodel = document.getElementById("calc-lbl-dmodel");
  if (lblDmodel) lblDmodel.innerText = `${d_model}`;
  const lblLayers = document.getElementById("calc-lbl-layers");
  if (lblLayers) lblLayers.innerText = `${num_layers} 层`;
  const lblNhead = document.getElementById("calc-lbl-nhead");
  if (lblNhead) lblNhead.innerText = `${nhead} Heads`;
  const lblPatches = document.getElementById("calc-lbl-patches");
  if (lblPatches) lblPatches.innerText = `${patches} 个`;
  const lblLines = document.getElementById("calc-lbl-lines");
  if (lblLines) lblLines.innerText = `${lines} 根`;

  // Theoretical param computation
  // Backbone: ~1.08M (XFeat 5-stage Conv + FPN lateral layers)
  const backbone_params = 1085000;
  // Line encoder: MLP 5 -> 64 -> 64
  const line_encoder_params = (5 * 64 + 64) + (64 * 64 + 64) + 64; // ~4.8K
  // Fusion projection: Linear(192 + 64, d_model)
  const fusion_params = (256 * d_model + d_model);
  // GNN Transformer layers: each layer has 2 MHA (self + cross) + 2 FFN
  const mha_params = 4 * (d_model * d_model + d_model);
  const ffn_params = 2 * (d_model * (d_model * 2) + d_model * 2);
  const layer_params = 2 * mha_params + ffn_params;
  const total_gnn_params = num_layers * layer_params;
  // Matcher proj: Linear(d_model, d_model)
  const matcher_params = 2 * (d_model * d_model + d_model);

  const total_params = backbone_params + line_encoder_params + fusion_params + total_gnn_params + matcher_params;
  const total_params_m = (total_params / 1e6).toFixed(2);

  // FLOPs computation:
  // Backbone per 112x112 patch: ~0.25 GFLOPs * 2 * patches
  const backbone_flops = 0.25 * 2 * patches;
  // Attention FLOPs: O(M^2 * d_model)
  const attn_flops = (num_layers * (4 * patches * patches * d_model * 2)) / 1e9;
  const total_flops = (backbone_flops + attn_flops).toFixed(1);

  // VRAM estimation (AMP):
  const weights_mb = (total_params * 2) / (1024 * 1024); // FP16 weights
  const gradients_mb = (total_params * 4) / (1024 * 1024); // FP32 gradients
  const optimizer_mb = (total_params * 8) / (1024 * 1024); // Adam FP32 moments
  const act_mb = (patches * 112 * 112 * 32 * 2 * 4) / (1024 * 1024);
  const train_vram_gb = ((weights_mb + gradients_mb + optimizer_mb + act_mb + 450) / 1024).toFixed(2);

  // Throughput FPS
  const latency_ms = 10 + (patches * 0.35) + (num_layers * 1.5) + (d_model * 0.02);
  const est_fps = Math.round(1000.0 / latency_ms);

  const outParams = document.getElementById("calc-out-params");
  if (outParams) outParams.innerText = `${total_params_m} M`;
  const outFlops = document.getElementById("calc-out-flops");
  if (outFlops) outFlops.innerText = `${total_flops} G`;
  const outVram = document.getElementById("calc-out-train-vram");
  if (outVram) outVram.innerText = `${train_vram_gb} GB`;
  const outFps = document.getElementById("calc-out-fps");
  if (outFps) outFps.innerText = `${est_fps} FPS`;

  const recEl = document.getElementById("calc-out-recommendation");
  if (recEl) {
    if (parseFloat(train_vram_gb) < 3.0) {
      recEl.innerHTML = `💡 <strong>硬件适配诊断</strong>：当前参数配置非常轻量，完全可在 4GB~8GB 显卡（如 RTX 3050/4060 Laptop）开启 AMP 高速训练，单对推理延迟仅约 ~${Math.round(latency_ms)}ms。`;
      recEl.style.color = "#A7F3D0";
      recEl.style.borderColor = "rgba(16, 185, 129, 0.25)";
    } else {
      recEl.innerHTML = `⚠️ <strong>显存压力提示</strong>：候选切片或层数较多，训练显存预估达到 ${train_vram_gb} GB，建议在 DataLoader 中配置 <code>--grad-accum 2</code> 分摊显存开销。`;
      recEl.style.color = "#FDE68A";
      recEl.style.borderColor = "rgba(245, 158, 11, 0.25)";
    }
  }
}

function switchCodeSnippet(key) {
  currentCodeSnippetKey = key;
  const snip = CODE_SNIPPETS[key];
  if (!snip) return;

  // Update button highlights
  const btnMap = {
    python_api: "btn-snippet-python",
    train_cli: "btn-snippet-train",
    eval_cli: "btn-snippet-eval",
    dataset_cli: "btn-snippet-dataset",
    review_cli: "btn-snippet-review"
  };
  Object.values(btnMap).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = "btn btn-secondary btn-sm";
  });
  const activeBtn = document.getElementById(btnMap[key]);
  if (activeBtn) activeBtn.className = "btn btn-primary btn-sm";

  const titleEl = document.getElementById("code-snippet-title");
  if (titleEl) titleEl.innerText = snip.title;

  const contentEl = document.getElementById("code-snippet-content");
  if (contentEl) contentEl.innerText = snip.code;
}

function copyCurrentCodeSnippet() {
  const contentEl = document.getElementById("code-snippet-content");
  if (!contentEl) return;
  const text = contentEl.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("btn-copy-code");
    if (btn) {
      const orig = btn.innerText;
      btn.innerText = "✓ 已复制到剪贴板！";
      btn.style.background = "#10B981";
      btn.style.color = "#FFF";
      setTimeout(() => {
        btn.innerText = orig;
        btn.style.background = "";
        btn.style.color = "";
      }, 2000);
    }
  }).catch(err => {
    alert(`复制失败: ${err}`);
  });
}
