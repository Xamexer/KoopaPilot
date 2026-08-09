// Dashboard JS - Chart.js graphs and live polling

let rewardChart = null;
let lengthChart = null;
let progressChart = null;
let currentRun = null;
let comparisonRuns = [];
let pollInterval = null;
let smoothingWindow = 1;
const POLL_MS = window.SMW_DASHBOARD_POLL_MS || 5000;
const CHART_COLORS = [
  '#4ecca3',
  '#e94560',
  '#533483',
  '#f9a825',
  '#00bcd4',
  '#ff7043',
  '#8bc34a',
  '#ba68c8',
];

function ensureDashboardEnhancements() {
  if (!document.getElementById('dashboardResponsiveStyles')) {
    const styles = document.createElement('style');
    styles.id = 'dashboardResponsiveStyles';
    styles.textContent = `
            html, body { max-width: 100%; overflow-x: hidden; }
            .header { flex-wrap: wrap; gap: 12px; }
            .header .controls { flex-wrap: wrap; }
            .dashboard { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .card, .chart-container { min-width: 0; }
            .chart-container { width: 100%; }
            .chart-container canvas { max-width: 100%; }
            .stats-grid { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
            .smoothing-control {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #aaa;
                font-size: 12px;
            }
            .smoothing-control input { width: 150px; accent-color: #4ecca3; }
            .smoothing-value { min-width: 58px; color: #4ecca3; font-weight: bold; }
            @media (max-width: 900px) {
                .dashboard { grid-template-columns: minmax(0, 1fr); }
                .card.full { grid-column: auto; }
            }
            @media (max-width: 600px) {
                .header { padding: 12px 16px; }
                .header .controls { width: 100%; gap: 8px; }
                .header .controls select { max-width: 100%; }
                .dashboard { gap: 12px; padding: 12px; }
                .stats-grid { gap: 8px; padding: 12px 12px 0 !important; }
                .stat-box { padding: 8px; }
                .stat-box .value { font-size: 18px; }
                .chart-container { height: 240px; }
            }
        `;
    document.head.appendChild(styles);
  }

  if (!document.getElementById('smoothingSlider')) {
    const control = document.createElement('label');
    control.className = 'smoothing-control';
    control.htmlFor = 'smoothingSlider';
    control.innerHTML = `
            Smooth
            <input id="smoothingSlider" type="range" min="1" max="100"
                   value="1" step="1">
            <span class="smoothing-value" id="smoothingValue">Raw</span>
        `;
    const controls = document.querySelector('.header .controls');
    const status = document.getElementById('statusBadge');
    controls.insertBefore(control, status);
    document
      .getElementById('smoothingSlider')
      .addEventListener('input', (event) => {
        setSmoothing(event.target.value);
      });
  }
}

function smoothPoints(points) {
  if (smoothingWindow <= 1) return points;

  return points.map((point, index) => {
    const start = Math.max(0, index - smoothingWindow + 1);
    const values = points
      .slice(start, index + 1)
      .map((item) => Number(item.y))
      .filter(Number.isFinite);
    const average =
      values.reduce((sum, value) => sum + value, 0) / values.length;
    return { x: point.x, y: average };
  });
}

function setSmoothing(value) {
  smoothingWindow = Number(value);
  document.getElementById('smoothingValue').textContent =
    smoothingWindow === 1 ? 'Raw' : `${smoothingWindow} pts`;
  updateCharts();
}

// Initialize charts
function initCharts() {
  const rewardCtx = document.getElementById('rewardChart').getContext('2d');
  rewardChart = new Chart(rewardCtx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Timesteps', color: '#888' },
          ticks: { color: '#888' },
          grid: { color: '#1a1a2e' },
        },
        y: {
          title: { display: true, text: 'Reward', color: '#888' },
          ticks: { color: '#888' },
          grid: { color: '#0f3460' },
        },
      },
      plugins: {
        legend: { labels: { color: '#e0e0e0' } },
      },
    },
  });

  const lengthCtx = document.getElementById('lengthChart').getContext('2d');
  lengthChart = new Chart(lengthCtx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Timesteps', color: '#888' },
          ticks: { color: '#888' },
          grid: { color: '#1a1a2e' },
        },
        y: {
          title: { display: true, text: 'Steps', color: '#888' },
          ticks: { color: '#888' },
          grid: { color: '#0f3460' },
        },
      },
      plugins: {
        legend: { labels: { color: '#e0e0e0' } },
      },
    },
  });

  const progressCtx = document.getElementById('progressChart').getContext('2d');
  progressChart = new Chart(progressCtx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Timesteps', color: '#888' },
          ticks: { color: '#888' },
          grid: { color: '#1a1a2e' },
        },
        y: {
          title: { display: true, text: 'Level X', color: '#888' },
          ticks: { color: '#888' },
          grid: { color: '#0f3460' },
        },
      },
      plugins: {
        legend: { labels: { color: '#e0e0e0' } },
      },
    },
  });
}

// Fetch run list
async function fetchRuns() {
  try {
    const resp = await fetch('/api/runs');
    const runs = await resp.json();
    const select = document.getElementById('runSelect');
    select.innerHTML = '<option value="">Select run...</option>';
    runs.forEach((r) => {
      const opt = document.createElement('option');
      opt.value = r.name;
      opt.textContent = r.name;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error('Failed to fetch runs:', e);
  }
}

// Load a specific run
async function loadRun(runName) {
  if (!runName) return;
  try {
    const resp = await fetch(`/api/metrics/${runName}`);
    if (!resp.ok) throw new Error(`Request failed: ${resp.status}`);
    const data = await resp.json();
    currentRun = { name: runName, data: data };
    updateDashboard();
    startPolling(runName, false);
  } catch (e) {
    console.error('Failed to load run:', e);
  }
}

// Load latest run
async function loadLatest() {
  try {
    const resp = await fetch('/api/latest');
    if (!resp.ok) throw new Error(`Request failed: ${resp.status}`);
    const data = await resp.json();
    currentRun = {
      name: data.run_id ? `run_${data.run_id}` : 'latest',
      data: data,
    };
    updateDashboard();
    startPolling(null, true);
    document.getElementById('statusBadge').className = 'status live';
    document.getElementById('statusBadge').textContent = 'Live';
  } catch (e) {
    document.getElementById('statusBadge').className = 'status stopped';
    document.getElementById('statusBadge').textContent = 'No runs';
  }
}

// Add comparison run
function addComparison() {
  const select = document.getElementById('runSelect');
  const runName = select.value;
  if (!runName || comparisonRuns.find((r) => r.name === runName)) return;

  fetch(`/api/metrics/${runName}`)
    .then((r) => r.json())
    .then((data) => {
      comparisonRuns.push({ name: runName, data: data });
      updateCompareList();
      updateCharts();
    });
}

function removeComparison(name) {
  comparisonRuns = comparisonRuns.filter((r) => r.name !== name);
  updateCompareList();
  updateCharts();
}

function updateCompareList() {
  const list = document.getElementById('compareList');
  list.innerHTML = comparisonRuns
    .map(
      (r) =>
        `<span class="compare-tag" onclick="removeComparison('${r.name}')">${r.name} x</span>`,
    )
    .join('');
}

// Update stats and charts
function updateDashboard() {
  if (!currentRun) return;
  const iters = currentRun.data.iterations || [];
  const evaluations = currentRun.data.evaluations || [];
  const latestEvaluation = evaluations[evaluations.length - 1];

  // Show message if no data yet
  if (iters.length === 0) {
    document.getElementById('statTimesteps').textContent = '0';
    document.getElementById('statEpisodes').textContent = '0';
    document.getElementById('statMeanReward').textContent = 'No data yet';
    document.getElementById('statBestReward').textContent = 'Training...';
    document.getElementById('statGoalRate').textContent = '-';
    document.getElementById('statViewerReward').textContent = latestEvaluation
      ? Number(latestEvaluation.reward).toFixed(1)
      : '-';
    document.getElementById('statViewerGoal').textContent = latestEvaluation
      ? latestEvaluation.goal_reached
        ? 'Yes'
        : 'No'
      : '-';
    document.getElementById('statMeanMaxX').textContent = '-';
    document.getElementById('statBestMaxX').textContent = '-';

    // Config table still shows
    const cfg = currentRun.data.config_summary || {};
    const table = document.getElementById('configTable');
    table.innerHTML = Object.entries(cfg)
      .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
      .join('');

    // Clear charts
    updateCharts();
    lengthChart.data.datasets = [];
    lengthChart.update('none');
    progressChart.data.datasets = [];
    progressChart.update('none');
    return;
  }

  const last = iters[iters.length - 1];

  document.getElementById('statTimesteps').textContent = last
    ? last.timestep.toLocaleString()
    : '-';
  document.getElementById('statEpisodes').textContent = last
    ? last.episodes || '-'
    : '-';
  document.getElementById('statMeanReward').textContent = last
    ? last.mean_reward.toFixed(1)
    : '-';
  document.getElementById('statBestReward').textContent = last
    ? last.max_reward.toFixed(1)
    : '-';
  document.getElementById('statGoalRate').textContent =
    last && last.goal_rate !== undefined
      ? `${(last.goal_rate * 100).toFixed(0)}%`
      : '-';
  document.getElementById('statMeanMaxX').textContent =
    last && last.mean_max_x !== undefined ? last.mean_max_x.toFixed(0) : '-';
  document.getElementById('statBestMaxX').textContent =
    last && last.max_x !== undefined ? last.max_x.toFixed(0) : '-';

  document.getElementById('statViewerReward').textContent = latestEvaluation
    ? Number(latestEvaluation.reward).toFixed(1)
    : '-';
  document.getElementById('statViewerGoal').textContent = latestEvaluation
    ? latestEvaluation.goal_reached
      ? 'Yes'
      : 'No'
    : '-';

  // Config table
  const cfg = currentRun.data.config_summary || {};
  const table = document.getElementById('configTable');
  table.innerHTML = Object.entries(cfg)
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join('');

  updateCharts();
}

function updateCharts() {
  if (!currentRun) return;

  // Reward chart
  const datasets = [];
  const addDataset = (run, colorIdx) => {
    const iters = run.data.iterations || [];
    datasets.push({
      label: `Mean (${run.name})`,
      data: smoothPoints(
        iters.map((i) => ({ x: Number(i.timestep), y: Number(i.mean_reward) })),
      ),
      borderColor: CHART_COLORS[colorIdx % CHART_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
    });
    datasets.push({
      label: `Max (${run.name})`,
      data: smoothPoints(
        iters.map((i) => ({ x: Number(i.timestep), y: Number(i.max_reward) })),
      ),
      borderColor: CHART_COLORS[colorIdx % CHART_COLORS.length] + '80',
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [4, 4],
      pointRadius: 0,
      tension: 0.3,
    });
    const evaluations = run.data.evaluations || [];
    if (evaluations.length > 0) {
      datasets.push({
        label: `Deterministic viewer (${run.name})`,
        data: evaluations.map((item) => ({
          x: Number(item.timestep),
          y: Number(item.reward),
        })),
        borderColor: CHART_COLORS[colorIdx % CHART_COLORS.length],
        backgroundColor: CHART_COLORS[colorIdx % CHART_COLORS.length],
        showLine: false,
        pointRadius: 4,
        pointStyle: 'triangle',
      });
    }
  };

  addDataset(currentRun, 0);
  comparisonRuns.forEach((r, i) => addDataset(r, i + 1));

  rewardChart.data.datasets = datasets;
  rewardChart.update('none');

  // Length chart
  const iters = currentRun.data.iterations || [];
  lengthChart.data.datasets = [
    {
      label: 'Mean Episode Length',
      data: smoothPoints(
        iters.map((i) => ({ x: Number(i.timestep), y: Number(i.mean_length) })),
      ),
      borderColor: '#f9a825',
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
    },
  ];
  lengthChart.update('none');

  // Horizontal progress chart
  progressChart.data.datasets = [
    {
      label: 'Mean Episode Max X',
      data: smoothPoints(
        iters
          .filter((i) => i.mean_max_x !== undefined)
          .map((i) => ({ x: Number(i.timestep), y: Number(i.mean_max_x) })),
      ),
      borderColor: '#00bcd4',
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
    },
    {
      label: 'Best Episode Max X',
      data: smoothPoints(
        iters
          .filter((i) => i.max_x !== undefined)
          .map((i) => ({ x: Number(i.timestep), y: Number(i.max_x) })),
      ),
      borderColor: '#00bcd480',
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [4, 4],
      pointRadius: 0,
      tension: 0.3,
    },
  ];
  progressChart.update('none');
}

// Polling for live updates
function startPolling(runName, useLatest = false) {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const url = useLatest ? '/api/latest' : `/api/metrics/${runName}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Request failed: ${resp.status}`);
      const data = await resp.json();
      currentRun.data = data;
      updateDashboard();
      document.getElementById('statusBadge').className = 'status live';
      document.getElementById('statusBadge').textContent = 'Live';
    } catch (e) {
      document.getElementById('statusBadge').className = 'status stopped';
      document.getElementById('statusBadge').textContent = 'Offline';
    }
  }, POLL_MS);
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  ensureDashboardEnhancements();
  initCharts();
  fetchRuns();
  loadLatest();
});
