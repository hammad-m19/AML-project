/* Shared CSV workflow and chart rendering for ML/DL pages */

let currentSessionId = null;
let currentSummary = null;

function initCsvWorkflow(config) {
  const uploadForm = document.getElementById('upload-form');
  const algorithmSelect = document.getElementById('algorithm');
  const kmeansRow = document.getElementById('kmeans-row');

  if (algorithmSelect && kmeansRow) {
    const dbscanRow = document.getElementById('dbscan-row');
    algorithmSelect.addEventListener('change', () => {
      kmeansRow.style.display = algorithmSelect.value === 'kmeans' ? 'block' : 'none';
      if (dbscanRow) dbscanRow.style.display = algorithmSelect.value === 'dbscan' ? 'block' : 'none';
    });
  }

  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const status = document.getElementById('upload-status');
    const fileInput = document.getElementById('csv-file');
    if (!fileInput.files.length) {
      status.innerHTML = '<span class="error">Please select a CSV file.</span>';
      return;
    }

    status.innerHTML = '<span class="loading">Uploading and analyzing...</span>';
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    if (currentSessionId) formData.append('session_id', currentSessionId);

    try {
      const res = await fetch(config.uploadEndpoint, { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) {
        status.innerHTML = `<span class="error">${data.error}</span>`;
        return;
      }

      currentSessionId = data.session_id;
      currentSummary = data.summary;
      status.innerHTML = `<span class="success">Loaded ${data.filename} (${data.summary.rows} rows, ${data.summary.columns} columns)</span>`;

      document.getElementById('config-panel').style.display = 'block';
      document.getElementById('analysis-section').style.display = 'block';
      populateTargetSelect(data.summary.column_names);
      renderAnalysis(data.summary);
    } catch (err) {
      status.innerHTML = `<span class="error">${err.message}</span>`;
    }
  });

  const runBtn = document.getElementById('run-ml-btn') || document.getElementById('run-dl-btn');
  if (runBtn) {
    runBtn.addEventListener('click', () => runModel(config));
  }
}

function populateTargetSelect(columns) {
  const select = document.getElementById('target-column');
  if (!select) return;
  select.innerHTML = columns.map(c => `<option value="${c}">${c}</option>`).join('');
}

function renderAnalysis(summary) {
  const grid = document.getElementById('dataset-summary');
  grid.innerHTML = `
    <div class="summary-item"><div class="value">${summary.rows}</div><div class="label">Rows</div></div>
    <div class="summary-item"><div class="value">${summary.columns}</div><div class="label">Columns</div></div>
    <div class="summary-item"><div class="value">${summary.numeric_columns.length}</div><div class="label">Numeric</div></div>
    <div class="summary-item"><div class="value">${summary.categorical_columns.length}</div><div class="label">Categorical</div></div>
  `;

  const missingCols = Object.keys(summary.missing_values).filter(k => summary.missing_values[k] > 0);
  const missingChartEl = document.getElementById('missing-chart');
  if (missingCols.length) {
    missingChartEl.style.display = 'block';
    Plotly.newPlot('missing-chart', [{
      x: missingCols,
      y: missingCols.map(c => summary.missing_percent[c]),
      type: 'bar',
      marker: { color: '#f59e0b' }
    }], {
      title: 'Missing Values (%)',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#f1f5f9' },
      margin: { t: 40 }
    }, { responsive: true });
  } else {
    missingChartEl.style.display = 'none';
  }

  const corrKeys = Object.keys(summary.correlation);
  if (corrKeys.length >= 2) {
    const cols = summary.numeric_columns;
    const z = cols.map(r => cols.map(c => summary.correlation[r][c]));
    Plotly.newPlot('correlation-chart', [{
      z, x: cols, y: cols, type: 'heatmap',
      colorscale: 'Viridis'
    }], {
      title: 'Correlation Heatmap',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#f1f5f9' },
      margin: { t: 40 }
    }, { responsive: true });
  } else {
    document.getElementById('correlation-chart').innerHTML = '<p class="hint">Need 2+ numeric columns for correlation heatmap.</p>';
  }
}

async function runModel(config) {
  const resultsSection = document.getElementById('results-section');
  const metricsDisplay = document.getElementById('metrics-display');
  resultsSection.style.display = 'block';
  metricsDisplay.innerHTML = '<span class="loading">Training model...</span>';

  // Hide and clear all charts before running
  ['confusion-chart', 'feature-chart', 'prediction-chart', 'cluster-chart', 'training-chart'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.innerHTML = '';
      el.style.display = 'none';
    }
  });

  const payload = { session_id: currentSessionId };

  if (config.mode === 'ml') {
    payload.algorithm = document.getElementById('algorithm').value;
    payload.target_column = document.getElementById('target-column').value;
    if (payload.algorithm === 'kmeans') {
      payload.n_clusters = parseInt(document.getElementById('n-clusters').value);
    }
    if (payload.algorithm === 'dbscan') {
      payload.eps = parseFloat(document.getElementById('eps').value);
      payload.min_samples = parseInt(document.getElementById('min-samples').value);
    }
  } else {
    payload.target_column = document.getElementById('target-column').value;
    payload.epochs = parseInt(document.getElementById('epochs').value);
    payload.batch_size = parseInt(document.getElementById('batch-size').value);
    const archSelect = document.getElementById('architecture');
    if (archSelect) payload.architecture = archSelect.value;
  }

  try {
    const res = await fetch(config.runEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) {
      metricsDisplay.innerHTML = `<span class="error">${data.error}</span>`;
      return;
    }
    renderResults(data.results, config.mode);
  } catch (err) {
    metricsDisplay.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

function renderResults(results, mode) {
  const metricsDisplay = document.getElementById('metrics-display');
  const taskType = results.task_type;

  if (taskType === 'classification') {
    const m = results.metrics;
    metricsDisplay.innerHTML = `
      <div class="metric-card"><span class="metric-value">${(m.accuracy * 100).toFixed(1)}%</span><span class="metric-label">Accuracy</span></div>
      <div class="metric-card"><span class="metric-value">${results.model_name || results.algorithm}</span><span class="metric-label">Model</span></div>
    `;
    renderConfusionMatrix(m.confusion_matrix, m.labels);
  } else if (taskType === 'regression') {
    const m = results.metrics;
    metricsDisplay.innerHTML = `
      <div class="metric-card"><span class="metric-value">${m.r2_score}</span><span class="metric-label">R² Score</span></div>
      <div class="metric-card"><span class="metric-value">${m.rmse}</span><span class="metric-label">RMSE</span></div>
      <div class="metric-card"><span class="metric-value">${m.mae}</span><span class="metric-label">MAE</span></div>
    `;
    if (results.predictions) renderPredictionChart(results.predictions);
  } else if (taskType === 'clustering') {
    const m = results.metrics;
    metricsDisplay.innerHTML = `
      <div class="metric-card"><span class="metric-value">${m.n_clusters}</span><span class="metric-label">Clusters</span></div>
      <div class="metric-card"><span class="metric-value">${m.silhouette_score}</span><span class="metric-label">Silhouette</span></div>
      <div class="metric-card"><span class="metric-value">${m.inertia}</span><span class="metric-label">Inertia</span></div>
    `;
    renderClusterChart(results.cluster_labels);
  }

  if (results.feature_importance) {
    const indices = results.feature_importance.map((_, i) => `F${i + 1}`);
    document.getElementById('feature-chart').style.display = 'block';
    Plotly.newPlot('feature-chart', [{
      x: indices.slice(0, 20),
      y: results.feature_importance.slice(0, 20),
      type: 'bar',
      marker: { color: '#6366f1' }
    }], {
      title: 'Feature Importance (Top 20)',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#f1f5f9' },
      margin: { t: 40 }
    }, { responsive: true });
  }

  if (mode === 'dl' && results.training_history) {
    const h = results.training_history;
    const traces = [
      { x: h.loss.map((_, i) => i + 1), y: h.loss, name: 'Training Loss', type: 'scatter' },
      { x: h.val_loss.map((_, i) => i + 1), y: h.val_loss, name: 'Validation Loss', type: 'scatter' }
    ];
    if (h.accuracy) {
      traces.push({ x: h.accuracy.map((_, i) => i + 1), y: h.accuracy, name: 'Training Acc', type: 'scatter', yaxis: 'y2' });
      traces.push({ x: h.val_accuracy.map((_, i) => i + 1), y: h.val_accuracy, name: 'Val Acc', type: 'scatter', yaxis: 'y2' });
    }
    document.getElementById('training-chart').style.display = 'block';
    const trainingLayout = {
      title: 'Training History',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#f1f5f9' },
      xaxis: { title: 'Epoch' },
      yaxis: { title: 'Loss' },
      margin: { t: 40 }
    };
    if (h.accuracy) {
      trainingLayout.yaxis2 = { title: 'Accuracy', overlaying: 'y', side: 'right' };
    }
    Plotly.newPlot('training-chart', traces, trainingLayout, { responsive: true });
  }
}

function renderConfusionMatrix(cm, labels) {
  const el = document.getElementById('confusion-chart');
  if (!el || !cm.length) return;
  el.style.display = 'block';
  const n = cm.length;
  const lbls = labels || Array.from({ length: n }, (_, i) => `Class ${i}`);
  Plotly.newPlot('confusion-chart', [{
    z: cm, x: lbls, y: lbls, type: 'heatmap', colorscale: 'Blues'
  }], {
    title: 'Confusion Matrix',
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#f1f5f9' },
    margin: { t: 40 }
  }, { responsive: true });
}

function renderPredictionChart(predictions) {
  const el = document.getElementById('prediction-chart');
  if (el) el.style.display = 'block';
  Plotly.newPlot('prediction-chart', [
    { x: predictions.actual, y: predictions.predicted, mode: 'markers', type: 'scatter', name: 'Predictions', marker: { color: '#6366f1' } },
    { x: [Math.min(...predictions.actual), Math.max(...predictions.actual)], y: [Math.min(...predictions.actual), Math.max(...predictions.actual)], mode: 'lines', name: 'Ideal', line: { dash: 'dash', color: '#94a3b8' } }
  ], {
    title: 'Actual vs Predicted',
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#f1f5f9' },
    xaxis: { title: 'Actual' },
    yaxis: { title: 'Predicted' },
    margin: { t: 40 }
  }, { responsive: true });
}

function renderClusterChart(labels) {
  const counts = {};
  labels.forEach(l => { counts[l] = (counts[l] || 0) + 1; });
  const el = document.getElementById('cluster-chart');
  if (el) el.style.display = 'block';
  Plotly.newPlot('cluster-chart', [{
    x: Object.keys(counts).map(k => `Cluster ${k}`),
    y: Object.values(counts),
    type: 'bar',
    marker: { color: '#8b5cf6' }
  }], {
    title: 'Cluster Distribution',
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#f1f5f9' },
    margin: { t: 40 }
  }, { responsive: true });
}
