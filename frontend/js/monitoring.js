/**
 * monitoring.js — Centralised Performance Monitoring, Execution History & Resource Telemetry
 */
const Monitoring = (() => {
  let isPolling = false;
  let pollTimer = null;
  let currentPage = 1;
  const perPage = 15;
  const selectedRecordIds = new Set();

  const dom = {};

  function cacheDom() {
    // Header & Controls
    dom.refreshBtn = document.getElementById('mon-refresh-btn');
    dom.autoPollToggle = document.getElementById('mon-auto-refresh-toggle');

    // Telemetry Elements
    dom.cpuVal = document.getElementById('tele-cpu-val');
    dom.cpuBar = document.getElementById('tele-cpu-bar');
    dom.cpuUser = document.getElementById('tele-cpu-user');
    dom.cpuSys = document.getElementById('tele-cpu-sys');

    dom.ramVal = document.getElementById('tele-ram-val');
    dom.ramBar = document.getElementById('tele-ram-bar');
    dom.ramVms = document.getElementById('tele-ram-vms');
    dom.pid = document.getElementById('tele-pid');

    dom.gpuVal = document.getElementById('tele-gpu-val');
    dom.gpuUnit = document.getElementById('tele-gpu-unit');
    dom.gpuBar = document.getElementById('tele-gpu-bar');
    dom.gpuBadge = document.getElementById('tele-gpu-badge');
    dom.gpuSub1 = document.getElementById('tele-gpu-sub1');
    dom.gpuSub2 = document.getElementById('tele-gpu-sub2');

    dom.diskRead = document.getElementById('tele-disk-read');
    dom.diskWrite = document.getElementById('tele-disk-write');
    dom.threads = document.getElementById('tele-threads');

    // Storage Elements
    dom.storageDbSize = document.getElementById('storage-db-size');
    dom.storageArticles = document.getElementById('storage-articles-count');
    dom.storageEmb = document.getElementById('storage-emb-count');
    dom.storageRecs = document.getElementById('storage-recs-count');
    dom.storageTablesBody = document.getElementById('storage-tables-body');

    // Live Latency Elements
    dom.liveP50 = document.getElementById('live-p50');
    dom.liveP90 = document.getElementById('live-p90');
    dom.liveP95 = document.getElementById('live-p95');
    dom.liveP99 = document.getElementById('live-p99');
    dom.liveAvg = document.getElementById('live-avg');
    dom.liveMeta = document.getElementById('live-recs-meta');
    dom.liveStageTags = document.getElementById('live-stage-breakdown-tags');

    // History & Filter Elements
    dom.filterJobType = document.getElementById('filter-job-type');
    dom.filterStatus = document.getElementById('filter-status');
    dom.filterVersion = document.getElementById('filter-version');
    dom.filterDate = document.getElementById('filter-date');
    dom.filterResetBtn = document.getElementById('filter-reset-btn');

    dom.historyBody = document.getElementById('execution-history-body');
    dom.selectAllExecs = document.getElementById('select-all-execs');
    dom.paginationInfo = document.getElementById('exec-pagination-info');
    dom.prevBtn = document.getElementById('exec-prev-btn');
    dom.nextBtn = document.getElementById('exec-next-btn');
    dom.compareBtn = document.getElementById('btn-compare-selected');
    dom.cleanupBtn = document.getElementById('btn-cleanup-modal-trigger');

    // Comparison View
    dom.comparisonCard = document.getElementById('comparison-card');
    dom.comparisonGrid = document.getElementById('comparison-grid');
    dom.closeComparisonBtn = document.getElementById('btn-close-comparison');

    // Modal
    dom.modal = document.getElementById('exec-details-modal');
    dom.modalCloseBtn = document.getElementById('modal-close-btn');
    dom.modalContent = document.getElementById('modal-content');
  }

  function bindEvents() {
    if (dom.refreshBtn) {
      dom.refreshBtn.addEventListener('click', () => {
        refreshAll();
      });
    }

    if (dom.autoPollToggle) {
      dom.autoPollToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
          startPolling();
        } else {
          stopPolling();
        }
      });
    }

    // Filter change listeners
    [dom.filterJobType, dom.filterStatus, dom.filterVersion, dom.filterDate].forEach(el => {
      if (el) {
        el.addEventListener('change', () => {
          currentPage = 1;
          loadHistory();
        });
      }
    });

    if (dom.filterResetBtn) {
      dom.filterResetBtn.addEventListener('click', () => {
        if (dom.filterJobType) dom.filterJobType.value = '';
        if (dom.filterStatus) dom.filterStatus.value = '';
        if (dom.filterVersion) dom.filterVersion.value = '';
        if (dom.filterDate) dom.filterDate.value = '';
        currentPage = 1;
        loadHistory();
      });
    }

    // Pagination
    if (dom.prevBtn) {
      dom.prevBtn.addEventListener('click', () => {
        if (currentPage > 1) {
          currentPage--;
          loadHistory();
        }
      });
    }

    if (dom.nextBtn) {
      dom.nextBtn.addEventListener('click', () => {
        currentPage++;
        loadHistory();
      });
    }

    // Compare button
    if (dom.compareBtn) {
      dom.compareBtn.addEventListener('click', () => {
        if (selectedRecordIds.size >= 2) {
          runComparison(Array.from(selectedRecordIds));
        }
      });
    }

    if (dom.closeComparisonBtn) {
      dom.closeComparisonBtn.addEventListener('click', () => {
        if (dom.comparisonCard) dom.comparisonCard.classList.add('hidden');
      });
    }

    // Modal close
    if (dom.modalCloseBtn && dom.modal) {
      dom.modalCloseBtn.addEventListener('click', () => {
        dom.modal.classList.add('hidden');
      });
      dom.modal.addEventListener('click', (e) => {
        if (e.target === dom.modal) dom.modal.classList.add('hidden');
      });
    }

    // Cleanup button
    if (dom.cleanupBtn) {
      dom.cleanupBtn.addEventListener('click', handleRetentionCleanup);
    }
  }

  // ─── Telemetry Polling ──────────────────────────────────────────────────────

  async function fetchResources() {
    try {
      const snap = await Api.monitoringResources();
      // CPU
      const cpu = snap.cpu_percent !== null && snap.cpu_percent !== undefined ? snap.cpu_percent : 0.0;
      if (dom.cpuVal) dom.cpuVal.textContent = cpu.toFixed(1);
      if (dom.cpuBar) dom.cpuBar.style.width = `${Math.min(cpu, 100)}%`;
      if (dom.cpuUser) dom.cpuUser.textContent = snap.cpu_time_user !== null ? `${snap.cpu_time_user}s` : '—';
      if (dom.cpuSys) dom.cpuSys.textContent = snap.cpu_time_sys !== null ? `${snap.cpu_time_sys}s` : '—';

      // RAM
      if (dom.ramVal) dom.ramVal.textContent = snap.ram_mb !== null ? Math.round(snap.ram_mb) : '—';
      if (dom.ramVms) dom.ramVms.textContent = snap.ram_vms_mb !== null ? `${Math.round(snap.ram_vms_mb)} MB` : '—';
      if (dom.pid) dom.pid.textContent = snap.pid || '—';
      if (dom.ramBar && snap.ram_mb) {
        // Assume 16GB max for bar scale visualisation
        const pct = Math.min((snap.ram_mb / 16384) * 100, 100);
        dom.ramBar.style.width = `${Math.max(pct, 5)}%`;
      }

      // GPU (Distinguish unavailable metrics from genuine zero)
      if (snap.gpu_available) {
        if (dom.gpuBadge) {
          dom.gpuBadge.textContent = 'ONLINE (CUDA)';
          dom.gpuBadge.style.background = 'rgba(34, 197, 94, 0.15)';
          dom.gpuBadge.style.color = '#22c55e';
        }
        if (dom.gpuVal) dom.gpuVal.textContent = snap.gpu_util_pct !== null ? snap.gpu_util_pct.toFixed(0) : '0';
        if (dom.gpuUnit) dom.gpuUnit.textContent = '% Util';
        if (dom.gpuBar) dom.gpuBar.style.width = `${Math.min(snap.gpu_util_pct || 0, 100)}%`;
        if (dom.gpuSub1) {
          const used = snap.gpu_mem_used_mb !== null ? `${Math.round(snap.gpu_mem_used_mb)} MB` : '0 MB';
          const total = snap.gpu_mem_total_mb !== null ? `${Math.round(snap.gpu_mem_total_mb)} MB` : '—';
          dom.gpuSub1.textContent = `Memory: ${used} / ${total}`;
        }
      } else {
        if (dom.gpuBadge) {
          dom.gpuBadge.textContent = 'N/A (CPU-ONLY)';
          dom.gpuBadge.style.background = 'rgba(255, 255, 255, 0.05)';
          dom.gpuBadge.style.color = 'var(--text-muted)';
        }
        if (dom.gpuVal) dom.gpuVal.textContent = 'N/A';
        if (dom.gpuUnit) dom.gpuUnit.textContent = '';
        if (dom.gpuBar) dom.gpuBar.style.width = '0%';
        if (dom.gpuSub1) dom.gpuSub1.textContent = 'No CUDA GPU detected';
      }

      // Disk & Threads
      if (dom.diskRead) dom.diskRead.textContent = snap.disk_read_mb !== null ? snap.disk_read_mb.toFixed(1) : '—';
      if (dom.diskWrite) dom.diskWrite.textContent = snap.disk_write_mb !== null ? `${snap.disk_write_mb.toFixed(1)} MB` : '—';
      if (dom.threads) dom.threads.textContent = snap.active_threads || 1;

    } catch (e) {
      console.warn('Telemetry fetch error:', e);
    }
  }

  // ─── Database Storage Analytics ─────────────────────────────────────────────

  async function fetchStorage() {
    try {
      const storage = await Api.monitoringStorage();
      if (storage.error) return;

      if (dom.storageDbSize) dom.storageDbSize.textContent = storage.database_size_pretty || '—';
      if (dom.storageArticles) dom.storageArticles.textContent = (storage.total_articles || 0).toLocaleString();
      if (dom.storageEmb) {
        dom.storageEmb.textContent = `${(storage.embedded_articles || 0).toLocaleString()} (${storage.embedding_coverage_pct || 0}%)`;
      }
      if (dom.storageRecs) {
        dom.storageRecs.textContent = `${(storage.total_precomputed_recs || 0).toLocaleString()} (${storage.recommendation_coverage_pct || 0}%)`;
      }

      if (dom.storageTablesBody && Array.isArray(storage.tables)) {
        dom.storageTablesBody.innerHTML = '';
        storage.tables.forEach(t => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><strong style="color:var(--text-primary);">${t.table_name}</strong></td>
            <td>${t.approx_rows.toLocaleString()}</td>
            <td><span style="color:var(--brand-primary);font-weight:700;">${t.total_size}</span></td>
            <td>${t.table_size}</td>
            <td>${t.index_size}</td>
          `;
          dom.storageTablesBody.appendChild(tr);
        });
      }
    } catch (e) {
      console.warn('Storage fetch warning:', e);
    }
  }

  // ─── Live Recommendation Latency ───────────────────────────────────────────

  async function fetchLiveStats() {
    try {
      const stats = await Api.monitoringLiveStats();
      const p = stats.percentiles || {};
      if (dom.liveP50) dom.liveP50.textContent = p.p50_ms !== undefined ? `${p.p50_ms} ms` : '—';
      if (dom.liveP90) dom.liveP90.textContent = p.p90_ms !== undefined ? `${p.p90_ms} ms` : '—';
      if (dom.liveP95) dom.liveP95.textContent = p.p95_ms !== undefined ? `${p.p95_ms} ms` : '—';
      if (dom.liveP99) dom.liveP99.textContent = p.p99_ms !== undefined ? `${p.p99_ms} ms` : '—';
      if (dom.liveAvg) dom.liveAvg.textContent = p.avg_ms !== undefined ? `${p.avg_ms} ms` : '—';
      if (dom.liveMeta) {
        dom.liveMeta.textContent = `${stats.window_size || 0} queries in active rolling window (${stats.total_requests || 0} total)`;
      }

      if (dom.liveStageTags && stats.stage_breakdown_avg_ms) {
        dom.liveStageTags.innerHTML = '';
        const labels = {
          prep_ms: 'Query Prep',
          bm25_ms: 'BM25 Retrieval',
          dense_ms: 'Dense Vector Scan',
          rerank_ms: 'Cross-Encoder Rerank',
          blend_ms: 'Hybrid Blend',
          boosts_ms: 'News Boosts',
          mmr_ms: 'MMR Diversity',
          calibration_ms: 'Platt Calibration',
        };
        for (const [k, v] of Object.entries(stats.stage_breakdown_avg_ms)) {
          if (v > 0) {
            const chip = document.createElement('span');
            chip.className = 'status-badge';
            chip.style.background = 'rgba(255,255,255,0.04)';
            chip.style.border = '1px solid var(--border-subtle)';
            chip.style.color = 'var(--text-secondary)';
            chip.innerHTML = `${labels[k] || k}: <strong style="color:var(--text-primary);margin-left:4px;">${v} ms</strong>`;
            dom.liveStageTags.appendChild(chip);
          }
        }
      }
    } catch (e) {
      console.warn('Live stats fetch warning:', e);
    }
  }

  // ─── Execution History & Filters ───────────────────────────────────────────

  async function loadHistory() {
    if (!dom.historyBody) return;
    try {
      const filters = {
        page: currentPage,
        perPage: perPage,
        jobType: dom.filterJobType ? dom.filterJobType.value : '',
        status: dom.filterStatus ? dom.filterStatus.value : '',
        pipelineVersion: dom.filterVersion ? dom.filterVersion.value.trim() : '',
        startDate: dom.filterDate ? dom.filterDate.value : '',
      };

      const res = await Api.monitoringRecords(filters);
      const records = res.records || [];
      const total = res.total || 0;
      const pages = res.pages || 1;

      // Update pagination controls
      if (dom.paginationInfo) {
        const start = total === 0 ? 0 : (currentPage - 1) * perPage + 1;
        const end = Math.min(currentPage * perPage, total);
        dom.paginationInfo.textContent = `Showing ${start}–${end} of ${total} runs (Page ${currentPage} of ${pages})`;
      }
      if (dom.prevBtn) dom.prevBtn.disabled = currentPage <= 1;
      if (dom.nextBtn) dom.nextBtn.disabled = currentPage >= pages;

      if (records.length === 0) {
        dom.historyBody.innerHTML = '<tr><td colspan="11" style="text-align:center;padding:20px;color:var(--text-muted)">No execution records found matching filters.</td></tr>';
        return;
      }

      dom.historyBody.innerHTML = '';
      records.forEach(r => {
        const tr = document.createElement('tr');
        const isChecked = selectedRecordIds.has(r.id);
        const statusClass = r.status === 'completed' ? 'status-completed' : (r.status === 'failed' ? 'status-failed' : 'status-running');

        const durationStr = r.duration_s !== null ? `${r.duration_s}s` : (r.duration_ms ? `${Math.round(r.duration_ms)}ms` : '—');
        const throughputStr = r.throughput_rps ? `${r.throughput_rps} rps` : '—';
        const cpuStr = r.cpu_peak_pct !== null && r.cpu_peak_pct !== undefined ? `${r.cpu_peak_pct}%` : '—';
        const ramStr = r.ram_peak_mb !== null && r.ram_peak_mb !== undefined ? `${Math.round(r.ram_peak_mb)} MB` : '—';
        const verStr = r.pipeline_version ? r.pipeline_version.substring(0, 10) : '—';
        const jobRef = r.job_id ? `<div style="font-size:0.7rem;color:var(--text-muted);">${r.job_id}</div>` : '';

        tr.innerHTML = `
          <td><input type="checkbox" class="record-select-checkbox" data-id="${r.id}" ${isChecked ? 'checked' : ''} /></td>
          <td>
            <strong style="color:var(--text-primary);font-family:monospace;">${r.id}</strong>
            ${jobRef}
          </td>
          <td><span class="badge" style="font-size:0.7rem;">${r.job_type}</span></td>
          <td>${durationStr}</td>
          <td><strong style="color:var(--color-success);">${throughputStr}</strong></td>
          <td>${cpuStr}</td>
          <td>${ramStr}</td>
          <td>${r.articles_success || 0}/${r.articles_total || 0}</td>
          <td><code style="font-size:0.75rem;">${verStr}</code></td>
          <td><span class="status-badge ${statusClass}">${r.status}</span></td>
          <td>
            <button type="button" class="btn btn-secondary btn-inspect-record" data-id="${r.id}" style="padding:4px 8px;font-size:0.72rem;">
              Inspect
            </button>
          </td>
        `;
        dom.historyBody.appendChild(tr);
      });

      // Bind row events
      dom.historyBody.querySelectorAll('.record-select-checkbox').forEach(cb => {
        cb.addEventListener('change', (e) => {
          const id = e.target.dataset.id;
          if (e.target.checked) selectedRecordIds.add(id);
          else selectedRecordIds.delete(id);
          updateCompareButtonState();
        });
      });

      dom.historyBody.querySelectorAll('.btn-inspect-record').forEach(btn => {
        btn.addEventListener('click', () => {
          openDetailsModal(btn.dataset.id);
        });
      });

    } catch (e) {
      console.warn('Execution history load warning:', e);
    }
  }

  function updateCompareButtonState() {
    const count = selectedRecordIds.size;
    if (dom.compareBtn) {
      dom.compareBtn.disabled = count < 2;
      dom.compareBtn.textContent = `⚖️ Compare Selected (${count})`;
    }
  }

  // ─── Execution Details Modal ────────────────────────────────────────────────

  async function openDetailsModal(recordId) {
    if (!dom.modal || !dom.modalContent) return;
    dom.modalContent.innerHTML = '<div style="text-align:center;padding:30px;color:var(--text-secondary);">Loading execution telemetry...</div>';
    dom.modal.classList.remove('hidden');

    try {
      const data = await Api.monitoringRecord(recordId);
      const r = data.record;
      const stages = data.stages || [];

      // Build waterfall strip if stage_timings exists
      let waterfallHtml = '';
      if (r.stage_timings && Object.keys(r.stage_timings).length > 0) {
        const total = r.duration_ms || Object.values(r.stage_timings).reduce((a, b) => a + b, 0);
        const palette = ['#38bdf8', '#a855f7', '#ec4899', '#f59e0b', '#22c55e', '#6366f1', '#14b8a6'];
        let idx = 0;
        const segs = Object.entries(r.stage_timings).map(([stg, ms]) => {
          const pct = Math.max((ms / total) * 100, 2);
          const color = palette[idx++ % palette.length];
          return `<div class="waterfall-seg" style="width:${pct}%;background:${color};" title="${stg}: ${ms}ms (${Math.round(pct)}%)">${stg} (${Math.round(ms)}ms)</div>`;
        }).join('');
        waterfallHtml = `
          <div style="font-size:0.75rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;margin-top:14px;">
            Stage Timing Waterfall
          </div>
          <div class="waterfall-bar-wrap">${segs}</div>
        `;
      }

      // Stage rows
      let stagesRows = '';
      if (stages.length > 0) {
        stagesRows = stages.map(s => `
          <tr>
            <td><strong>${s.stage_name}</strong></td>
            <td>${s.duration_ms !== null ? `${s.duration_ms} ms` : '—'}</td>
            <td>${s.articles_processed || 0}</td>
            <td>${s.cpu_pct !== null ? `${s.cpu_pct}%` : '—'}</td>
            <td>${s.ram_mb !== null ? `${Math.round(s.ram_mb)} MB` : '—'}</td>
          </tr>
        `).join('');
      } else if (r.stage_timings) {
        stagesRows = Object.entries(r.stage_timings).map(([k, v]) => `
          <tr>
            <td><strong>${k}</strong></td>
            <td>${v} ms</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
          </tr>
        `).join('');
      } else {
        stagesRows = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">No stage breakdown recorded for this job.</td></tr>';
      }

      const errorBanner = r.error_message
        ? `<div style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.35);color:#ef4444;padding:12px;border-radius:6px;margin-bottom:16px;">
            <strong>Error:</strong> ${r.error_message}
           </div>`
        : '';

      dom.modalContent.innerHTML = `
        <h2 style="font-size:1.2rem;font-weight:800;color:var(--text-primary);margin-bottom:4px;">
          Execution Telemetry: <code>${r.id}</code>
        </h2>
        <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:16px;">
          Job Type: <strong>${r.job_type}</strong> | Started: <strong>${r.started_at || '—'}</strong> | Completed: <strong>${r.completed_at || '—'}</strong>
        </div>

        ${errorBanner}

        <div class="storage-stats-row">
          <div class="storage-stat-pill">
            <div class="storage-pill-label">Total Duration</div>
            <div class="storage-pill-val">${r.duration_s ? `${r.duration_s}s` : `${Math.round(r.duration_ms || 0)}ms`}</div>
          </div>
          <div class="storage-stat-pill">
            <div class="storage-pill-label">Throughput</div>
            <div class="storage-pill-val" style="color:#22c55e;">${r.throughput_rps ? `${r.throughput_rps} rps` : '—'}</div>
          </div>
          <div class="storage-stat-pill">
            <div class="storage-pill-label">Peak RAM</div>
            <div class="storage-pill-val">${r.ram_peak_mb ? `${Math.round(r.ram_peak_mb)} MB` : '—'}</div>
          </div>
          <div class="storage-stat-pill">
            <div class="storage-pill-label">Peak CPU</div>
            <div class="storage-pill-val">${r.cpu_peak_pct !== null && r.cpu_peak_pct !== undefined ? `${r.cpu_peak_pct}%` : '—'}</div>
          </div>
        </div>

        ${waterfallHtml}

        <div style="font-size:0.75rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;margin:18px 0 8px;">
          Granular Stage Timings & Resource Allocation
        </div>
        <table class="monitoring-table">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Duration</th>
              <th>Articles Processed</th>
              <th>CPU %</th>
              <th>RAM</th>
            </tr>
          </thead>
          <tbody>${stagesRows}</tbody>
        </table>
      `;
    } catch (err) {
      dom.modalContent.innerHTML = `<div style="color:#ef4444;padding:20px;">Failed to load details: ${err.message}</div>`;
    }
  }

  // ─── Side-by-Side Execution Comparison ──────────────────────────────────────

  async function runComparison(recordIds) {
    if (!dom.comparisonCard || !dom.comparisonGrid) return;
    dom.comparisonCard.classList.remove('hidden');
    dom.comparisonGrid.innerHTML = '<div style="padding:20px;color:var(--text-secondary);">Loading side-by-side comparison...</div>';

    try {
      const data = await Api.monitoringCompare(recordIds);
      const records = data.records || [];

      dom.comparisonGrid.innerHTML = '';
      records.forEach(item => {
        const r = item.record;
        const col = document.createElement('div');
        col.className = 'comparison-column';

        let stageList = '';
        if (r.stage_timings) {
          stageList = Object.entries(r.stage_timings).map(([k, v]) => `
            <div class="compare-metric-row">
              <span style="color:var(--text-muted);">${k}:</span>
              <strong>${v} ms</strong>
            </div>
          `).join('');
        }

        col.innerHTML = `
          <div class="comparison-column-hdr">
            <div>${r.id}</div>
            <small style="color:var(--text-muted);font-weight:normal;">${r.job_type} (${r.status})</small>
          </div>
          <div class="compare-metric-row">
            <span>Pipeline Version:</span>
            <code>${r.pipeline_version ? r.pipeline_version.substring(0, 10) : '—'}</code>
          </div>
          <div class="compare-metric-row">
            <span>Duration:</span>
            <strong>${r.duration_s ? `${r.duration_s}s` : `${Math.round(r.duration_ms || 0)}ms`}</strong>
          </div>
          <div class="compare-metric-row">
            <span>Throughput:</span>
            <strong style="color:var(--color-success);">${r.throughput_rps ? `${r.throughput_rps} rps` : '—'}</strong>
          </div>
          <div class="compare-metric-row">
            <span>Peak RAM:</span>
            <strong>${r.ram_peak_mb ? `${Math.round(r.ram_peak_mb)} MB` : '—'}</strong>
          </div>
          <div class="compare-metric-row">
            <span>Peak CPU:</span>
            <strong>${r.cpu_peak_pct !== null && r.cpu_peak_pct !== undefined ? `${r.cpu_peak_pct}%` : '—'}</strong>
          </div>
          <div class="compare-metric-row">
            <span>Articles Success:</span>
            <strong>${r.articles_success || 0}/${r.articles_total || 0}</strong>
          </div>

          <div style="font-size:0.72rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;margin:12px 0 6px;">
            Stage Breakdown
          </div>
          ${stageList || '<div style="color:var(--text-muted);font-size:0.75rem;">No stage timings</div>'}
        `;
        dom.comparisonGrid.appendChild(col);
      });
      dom.comparisonCard.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
      dom.comparisonGrid.innerHTML = `<div style="color:#ef4444;padding:20px;">Comparison failed: ${e.message}</div>`;
    }
  }

  // ─── Retention Cleanup ──────────────────────────────────────────────────────

  async function handleRetentionCleanup() {
    if (!confirm('Run retention cleanup to prune benchmark execution records older than 90 days? (All ingestion upload history will be permanently preserved).')) {
      return;
    }
    try {
      const res = await Api.monitoringCleanup(90, ['ingestion']);
      if (window.Ui && typeof window.Ui.toast === 'function') {
        window.Ui.toast(`Cleaned up ${res.deleted_records} old execution records.`);
      } else {
        alert(`Cleaned up ${res.deleted_records} old execution records.`);
      }
      loadHistory();
    } catch (e) {
      alert(`Cleanup failed: ${e.message}`);
    }
  }

  // ─── Lifecycle & Auto-Refresh ───────────────────────────────────────────────

  function refreshAll() {
    fetchResources();
    fetchStorage();
    fetchLiveStats();
    loadHistory();
  }

  function startPolling() {
    if (isPolling) return;
    isPolling = true;
    pollTimer = setInterval(() => {
      fetchResources();
      fetchLiveStats();
    }, 3000);
  }

  function stopPolling() {
    isPolling = false;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function activate() {
    refreshAll();
    if (dom.autoPollToggle && dom.autoPollToggle.checked) {
      startPolling();
    }
  }

  function deactivate() {
    stopPolling();
  }

  async function init() {
    cacheDom();
    bindEvents();
  }

  return {
    init,
    activate,
    deactivate,
    refreshAll,
  };
})();

window.Monitoring = Monitoring;
