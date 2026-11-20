/**
 * editor.js — Complete Two-Pipeline Studio Controller.
 * Sub-Tab 1: Precomputation Pipeline Studio (with Cross-Encoder reranking & top-10 precomputation)
 * Sub-Tab 2: Live Computation Pipeline Studio (Table-separated, BGE-dense embedding only, zero Cross-Encoder)
 */
const Editor = (() => {
  let selectedPrecompFile = null;
  let selectedLiveFile = null;
  let activePrecompJobId = null;
  let activeLiveJobId = null;
  let precompPollInterval = null;
  let livePollInterval = null;
  const precompLogged = new Set();
  let currentStudio = 'precomputed';

  const escHtml = (s) => String(s == null ? '' : s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // DOM elements cache
  const dom = {};

  function cacheDom() {
    // Top Tabs
    dom.btnTabPrecomp = document.getElementById('tab-btn-precomputed');
    dom.btnTabLive = document.getElementById('tab-btn-live');
    dom.panelPrecomp = document.getElementById('editor-panel-precomputed');
    dom.panelLive = document.getElementById('editor-panel-live');

    // ── Precomputed Studio DOM ──
    dom.precompDsSelect = document.getElementById('editor-active-dataset-select');
    dom.precompSchemaBadge = document.getElementById('active-dataset-schema-badge');
    dom.precompStatsInline = document.getElementById('active-dataset-stats-inline');
    dom.btnNewPrecompRun = document.getElementById('btn-show-create-dataset');
    dom.btnCloneCorpus = document.getElementById('btn-show-clone-dataset');
    dom.btnDeletePrecompRun = document.getElementById('btn-delete-precomp-dataset');
    dom.btnRefreshPrecomp = document.getElementById('btn-refresh-datasets');

    dom.createPrecompCard = document.getElementById('create-dataset-form-card');
    dom.createPrecompForm = document.getElementById('create-dataset-form');
    dom.btnCancelCreatePrecomp = document.getElementById('btn-cancel-create-ds');

    dom.cloneCorpusCard = document.getElementById('clone-dataset-form-card');
    dom.cloneSourceSelect = document.getElementById('clone-source-dataset-select');
    dom.cloneWithEmbeddings = document.getElementById('clone-with-embeddings');
    dom.btnExecuteClone = document.getElementById('btn-execute-clone');
    dom.btnCancelClone = document.getElementById('btn-cancel-clone');

    dom.precompFileInput = document.getElementById('csv-file-input');
    dom.precompDropzone = document.getElementById('upload-dropzone');
    dom.precompUploadBtn = document.getElementById('start-ingest-btn');
    dom.precompDemoBtn = document.getElementById('ingest-demo-btn');
    dom.precompPreview = document.getElementById('file-preview');
    dom.precompPreviewName = document.getElementById('file-preview-name');

    dom.toggleManualBtn = document.getElementById('toggle-manual-form-btn');
    dom.manualForm = document.getElementById('manual-article-form');
    dom.manualAlert = document.getElementById('manual-alert-box');
    dom.manualSubmitBtn = document.getElementById('manual-submit-btn');

    dom.visualizer = document.getElementById('pipeline-visualizer-card');
    dom.completion = document.getElementById('pipeline-completion-banner');
    dom.progressPct = document.getElementById('visualizer-progress-pct');
    dom.progressFill = document.getElementById('visualizer-progress-fill');
    dom.statusMsg = document.getElementById('visualizer-status-msg');
    dom.logWindow = document.getElementById('pipeline-log-window');
    dom.switchBtn = document.getElementById('btn-switch-to-reader');

    dom.recentJobsBody = document.getElementById('recent-jobs-body');
    dom.inspectorSelect = document.getElementById('inspector-article-select');
    dom.inspectorCandidatesList = document.getElementById('inspector-candidates-list');
    dom.inspectorSourceBox = document.getElementById('inspector-source-box');
    dom.inspectorActions = document.getElementById('inspector-actions');
    dom.inspectorEmpty = document.getElementById('inspector-empty');
    dom.btnToggleAll = document.getElementById('btn-toggle-all-candidates');

    // ── Live Studio DOM ──
    dom.liveDsSelect = document.getElementById('live-active-dataset-select');
    dom.liveSchemaBadge = document.getElementById('live-dataset-schema-badge');
    dom.liveStatsInline = document.getElementById('live-dataset-stats-inline');
    dom.btnNewLiveDs = document.getElementById('btn-show-create-live-dataset');
    dom.btnDeleteLiveDs = document.getElementById('btn-delete-live-dataset');
    dom.btnRefreshLive = document.getElementById('btn-refresh-live-datasets');

    dom.createLiveCard = document.getElementById('create-live-dataset-form-card');
    dom.createLiveForm = document.getElementById('create-live-dataset-form');
    dom.btnCancelCreateLive = document.getElementById('btn-cancel-create-live-ds');

    dom.liveFileInput = document.getElementById('live-csv-file-input');
    dom.liveDropzone = document.getElementById('live-upload-dropzone');
    dom.liveUploadBtn = document.getElementById('live-start-csv-btn');
    dom.livePreview = document.getElementById('live-file-preview');
    dom.livePreviewName = document.getElementById('live-file-preview-name');
    dom.liveBatchSelect = document.getElementById('live-csv-batch-size');

    dom.toggleManualLiveBtn = document.getElementById('toggle-manual-live-form-btn');
    dom.manualLiveForm = document.getElementById('manual-live-form');
    dom.manualLiveAlert = document.getElementById('manual-live-alert-box');
    dom.manualLiveSubmitBtn = document.getElementById('manual-live-submit-btn');

    dom.startLivePrepBtn = document.getElementById('start-live-prep-btn');
    dom.livePrepProgress = document.getElementById('live-prep-progress-card');
    dom.livePrepStatusText = document.getElementById('live-prep-status-text');
    dom.livePrepProgressPct = document.getElementById('live-prep-progress-pct');
    dom.livePrepProgressFill = document.getElementById('live-prep-progress-fill');
    dom.livePrepRecompute = document.getElementById('live-prep-recompute-checkbox');
    dom.livePrepBatch = document.getElementById('live-prep-batch-select');

    dom.liveJobsBody = document.getElementById('live-jobs-body');
  }

  async function init() {
    cacheDom();
    bindEvents();
    await loadPrecomputedDatasets();
    await loadLiveDatasets();
    await refreshPrecompDbHealth();
    loadInspectorArticles();
    loadRecentPrecompJobs();
    loadRecentLiveJobs();
  }

  function bindEvents() {
    // 1. Studio Sub-Tab Switcher
    if (dom.btnTabPrecomp) {
      dom.btnTabPrecomp.addEventListener('click', () => switchStudio('precomputed'));
    }
    if (dom.btnTabLive) {
      dom.btnTabLive.addEventListener('click', () => switchStudio('live'));
    }

    // 2. Precomputed Dataset Selector Switch
    if (dom.precompDsSelect) {
      dom.precompDsSelect.addEventListener('change', async (e) => {
        const dsId = e.target.value;
        State.set({ activeDataset: dsId });
        showToast(`Switched active precompute run to: ${dsId}`, 'info');
        await refreshPrecompDbHealth();
        loadInspectorArticles();
        loadRecentPrecompJobs();
      });
    }

    // 3. Live Dataset Selector Switch
    if (dom.liveDsSelect) {
      dom.liveDsSelect.addEventListener('change', async (e) => {
        const dsId = e.target.value;
        showToast(`Switched active live dataset to: ${dsId}`, 'info');
        await refreshLiveDbHealth();
        loadRecentLiveJobs();
      });
    }

    // 4. Create Precompute Run Toggle & Submit
    if (dom.btnNewPrecompRun && dom.createPrecompCard) {
      dom.btnNewPrecompRun.addEventListener('click', () => {
        const isHidden = dom.createPrecompCard.style.display === 'none';
        dom.createPrecompCard.style.display = isHidden ? 'block' : 'none';
        if (dom.cloneCorpusCard) dom.cloneCorpusCard.style.display = 'none';
      });
    }
    if (dom.btnCancelCreatePrecomp && dom.createPrecompCard) {
      dom.btnCancelCreatePrecomp.addEventListener('click', () => {
        dom.createPrecompCard.style.display = 'none';
      });
    }
    if (dom.createPrecompForm) {
      dom.createPrecompForm.addEventListener('submit', handleCreatePrecompDataset);
    }

    // 5. Clone Corpus Toggle & Execute
    if (dom.btnCloneCorpus && dom.cloneCorpusCard) {
      dom.btnCloneCorpus.addEventListener('click', () => {
        const isHidden = dom.cloneCorpusCard.style.display === 'none';
        dom.cloneCorpusCard.style.display = isHidden ? 'block' : 'none';
        if (dom.createPrecompCard) dom.createPrecompCard.style.display = 'none';
      });
    }
    if (dom.btnCancelClone && dom.cloneCorpusCard) {
      dom.btnCancelClone.addEventListener('click', () => {
        dom.cloneCorpusCard.style.display = 'none';
      });
    }
    if (dom.btnExecuteClone) {
      dom.btnExecuteClone.addEventListener('click', handleCloneCorpus);
    }

    // 6. Delete Precompute Run
    if (dom.btnDeletePrecompRun) {
      dom.btnDeletePrecompRun.addEventListener('click', handleDeletePrecompDataset);
    }
    if (dom.btnRefreshPrecomp) {
      dom.btnRefreshPrecomp.addEventListener('click', async () => {
        await loadPrecomputedDatasets();
        await refreshPrecompDbHealth();
        loadInspectorArticles();
        loadRecentPrecompJobs();
        showToast('Precomputed runs refreshed.', 'info');
      });
    }

    // 7. Create Live Dataset Toggle & Submit
    if (dom.btnNewLiveDs && dom.createLiveCard) {
      dom.btnNewLiveDs.addEventListener('click', () => {
        const isHidden = dom.createLiveCard.style.display === 'none';
        dom.createLiveCard.style.display = isHidden ? 'block' : 'none';
      });
    }
    if (dom.btnCancelCreateLive && dom.createLiveCard) {
      dom.btnCancelCreateLive.addEventListener('click', () => {
        dom.createLiveCard.style.display = 'none';
      });
    }
    if (dom.createLiveForm) {
      dom.createLiveForm.addEventListener('submit', handleCreateLiveDataset);
    }
    if (dom.btnDeleteLiveDs) {
      dom.btnDeleteLiveDs.addEventListener('click', handleDeleteLiveDataset);
    }
    if (dom.btnRefreshLive) {
      dom.btnRefreshLive.addEventListener('click', async () => {
        await loadLiveDatasets();
        await refreshLiveDbHealth();
        loadRecentLiveJobs();
        showToast('Live datasets refreshed.', 'info');
      });
    }

    // 8. Manual Precomputed Article Form Toggle & Submit
    if (dom.toggleManualBtn && dom.manualForm) {
      dom.toggleManualBtn.addEventListener('click', () => {
        const isHidden = dom.manualForm.style.display === 'none';
        dom.manualForm.style.display = isHidden ? 'block' : 'none';
        dom.toggleManualBtn.innerHTML = isHidden
          ? '<span>✕</span> Close Manual Editor'
          : '<span id="manual-form-toggle-icon">✏️</span> Write / Publish Single Article';
      });
      dom.manualForm.addEventListener('submit', handleManualPrecompSubmit);
    }

    // 9. Manual Live Article Form Toggle & Submit
    if (dom.toggleManualLiveBtn && dom.manualLiveForm) {
      dom.toggleManualLiveBtn.addEventListener('click', () => {
        const isHidden = dom.manualLiveForm.style.display === 'none';
        dom.manualLiveForm.style.display = isHidden ? 'block' : 'none';
        dom.toggleManualLiveBtn.innerHTML = isHidden
          ? '<span>✕</span> Close Form'
          : '<span id="manual-live-toggle-icon">✏️</span> Write Single Article';
      });
      dom.manualLiveForm.addEventListener('submit', handleManualLiveSubmit);
    }

    // 10. Precomputed CSV File Dropzone & Upload
    if (dom.precompDropzone && dom.precompFileInput) {
      dom.precompDropzone.addEventListener('click', (e) => {
        if (e.target !== dom.precompFileInput) dom.precompFileInput.click();
      });
      dom.precompFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) onPrecompFileChosen(e.target.files[0]);
      });
      ['dragenter', 'dragover'].forEach(ev => {
        dom.precompDropzone.addEventListener(ev, (e) => {
          e.preventDefault();
          dom.precompDropzone.classList.add('dragover');
        });
      });
      ['dragleave', 'drop'].forEach(ev => {
        dom.precompDropzone.addEventListener(ev, (e) => {
          e.preventDefault();
          dom.precompDropzone.classList.remove('dragover');
        });
      });
      dom.precompDropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) onPrecompFileChosen(e.dataTransfer.files[0]);
      });
    }
    if (dom.precompUploadBtn) dom.precompUploadBtn.addEventListener('click', handlePrecompUpload);
    if (dom.precompDemoBtn) dom.precompDemoBtn.addEventListener('click', handleDemoIngest);

    // 11. Live CSV File Dropzone & Upload
    if (dom.liveDropzone && dom.liveFileInput) {
      dom.liveDropzone.addEventListener('click', (e) => {
        if (e.target !== dom.liveFileInput) dom.liveFileInput.click();
      });
      dom.liveFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) onLiveFileChosen(e.target.files[0]);
      });
      ['dragenter', 'dragover'].forEach(ev => {
        dom.liveDropzone.addEventListener(ev, (e) => {
          e.preventDefault();
          dom.liveDropzone.classList.add('dragover');
        });
      });
      ['dragleave', 'drop'].forEach(ev => {
        dom.liveDropzone.addEventListener(ev, (e) => {
          e.preventDefault();
          dom.liveDropzone.classList.remove('dragover');
        });
      });
      dom.liveDropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) onLiveFileChosen(e.dataTransfer.files[0]);
      });
    }
    if (dom.liveUploadBtn) dom.liveUploadBtn.addEventListener('click', handleLiveCsvUpload);

    // 12. Live Preprocessing Trigger
    if (dom.startLivePrepBtn) {
      dom.startLivePrepBtn.addEventListener('click', handleLivePreprocessing);
    }

    // 13. Inspector & Navigation
    if (dom.inspectorSelect) {
      dom.inspectorSelect.addEventListener('change', (e) => {
        if (e.target.value) inspectArticle(e.target.value);
      });
    }
    if (dom.switchBtn) {
      dom.switchBtn.addEventListener('click', () => {
        if (window.setAppRole) window.setAppRole('reader');
      });
    }
  }

  function switchStudio(type) {
    currentStudio = type;
    const isPrecomp = type === 'precomputed';
    if (dom.btnTabPrecomp) dom.btnTabPrecomp.classList.toggle('active', isPrecomp);
    if (dom.btnTabLive) dom.btnTabLive.classList.toggle('active', !isPrecomp);
    if (dom.panelPrecomp) dom.panelPrecomp.classList.toggle('hidden', !isPrecomp);
    if (dom.panelLive) dom.panelLive.classList.toggle('hidden', isPrecomp);

    if (isPrecomp) {
      refreshPrecompDbHealth();
      loadRecentPrecompJobs();
    } else {
      refreshLiveDbHealth();
      loadRecentLiveJobs();
    }
  }

  // ─── Dataset Loading & Switching ──────────────────────────────────────────
  async function loadPrecomputedDatasets() {
    try {
      const res = await Api.listDatasets('precomputed');
      const datasets = res.datasets || [];
      const currentVal = dom.precompDsSelect?.value || State.get().activeDataset || 'default';

      if (dom.precompDsSelect) {
        dom.precompDsSelect.innerHTML = datasets.map(d => `
          <option value="${escHtml(d.id)}" ${d.id === currentVal ? 'selected' : ''}>
            ${escHtml(d.name)} (${escHtml(d.schema_name)})${d.is_protected ? ' [Protected]' : ''}
          </option>
        `).join('');
      }

      if (dom.cloneSourceSelect) {
        dom.cloneSourceSelect.innerHTML = datasets.map(d => `
          <option value="${escHtml(d.id)}">${escHtml(d.name)} (${escHtml(d.schema_name)})</option>
        `).join('');
      }

      // Also sync reader precomputed selectors and comparison dropdowns
      const precompSelectors = [
        document.getElementById('reader-precomp-dataset-select'),
        document.getElementById('similar-precomp-dataset-select'),
      ];
      const precompOpts = datasets.map(d => `
        <option value="${escHtml(d.id)}" ${d.id === currentVal ? 'selected' : ''}>
          ${escHtml(d.name)}${d.is_protected ? ' [Protected]' : ''}
        </option>
      `).join('');
      precompSelectors.forEach(sel => {
        if (sel) sel.innerHTML = precompOpts;
      });
      if (window.updatePrecompStatsBadge) window.updatePrecompStatsBadge(currentVal);

      if (window.Compare && typeof window.Compare.populateDatasets === 'function') {
        window.Compare.populateDatasets();
      }
    } catch (err) {
      console.warn('Load precomputed datasets warning:', err);
    }
  }

  async function loadLiveDatasets() {
    try {
      const res = await Api.listDatasets('live');
      const datasets = res.datasets || [];
      const currentVal = dom.liveDsSelect?.value || (datasets[0] ? datasets[0].id : '');

      if (dom.liveDsSelect) {
        if (!datasets.length) {
          dom.liveDsSelect.innerHTML = '<option value="">No live datasets created yet</option>';
        } else {
          dom.liveDsSelect.innerHTML = datasets.map(d => `
            <option value="${escHtml(d.id)}" ${d.id === currentVal ? 'selected' : ''}>
              ${escHtml(d.name)} (${escHtml(d.schema_name)})
            </option>
          `).join('');
        }
      }

      // Also sync reader live selectors (allow picking default production OR any live dataset)
      const liveSelectors = [
        document.getElementById('reader-live-dataset-select'),
        document.getElementById('hybrid-lab-dataset-select'),
      ];
      const defaultLiveOpt = `<option value="default">Default (Production Database - 4,193 articles)</option>`;
      const customLiveOpts = datasets.map(d => `
        <option value="${escHtml(d.id)}">
          ${escHtml(d.name)} (${escHtml(d.schema_name)})
        </option>
      `).join('');
      const liveOpts = defaultLiveOpt + customLiveOpts;
      liveSelectors.forEach(sel => {
        if (sel) {
          const prev = sel.value;
          sel.innerHTML = liveOpts;
          if (prev && sel.querySelector(`option[value="${prev}"]`)) sel.value = prev;
        }
      });
      if (window.updateLiveStatsBadge) window.updateLiveStatsBadge(liveSelectors[0]?.value || 'default');

      if (window.Compare && typeof window.Compare.populateDatasets === 'function') {
        window.Compare.populateDatasets();
      }
    } catch (err) {
      console.warn('Load live datasets warning:', err);
    }
  }

  async function refreshPrecompDbHealth() {
    try {
      const targetDs = dom.precompDsSelect?.value || State.get().activeDataset || 'default';
      const data = await Api.dbHealth(targetDs);

      const pgBadge = document.getElementById('pg-status-badge');
      const vecBadge = document.getElementById('vector-status-badge');
      const totalArts = document.getElementById('db-total-articles');
      const totalVecs = document.getElementById('db-total-vectors');
      const totalRecs = document.getElementById('db-total-recs');

      if (pgBadge) {
        pgBadge.className = data.connected ? 'stat-badge ready' : 'stat-badge error';
        pgBadge.innerHTML = data.connected ? `🟢 Connected (${data.schema || 'public'})` : '🔴 Disconnected';
      }
      if (vecBadge) {
        vecBadge.className = data.pgvector_ready ? 'stat-badge ready' : 'stat-badge warning';
        vecBadge.innerHTML = data.pgvector_ready ? '⚡ pgvector Ready (384d)' : '⚠️ Extension Missing';
      }
      if (totalArts) totalArts.textContent = (data.total_articles || 0).toLocaleString();
      if (totalVecs) totalVecs.textContent = (data.embedded_articles || 0).toLocaleString();
      if (totalRecs) totalRecs.textContent = (data.total_recommendations || 0).toLocaleString();

      if (dom.precompSchemaBadge) {
        dom.precompSchemaBadge.textContent = `${data.schema || 'public'}${targetDs === 'default' ? ' (Protected)' : ''}`;
      }
      if (dom.precompStatsInline) {
        dom.precompStatsInline.textContent = `${(data.total_articles || 0).toLocaleString()} arts | ${(data.total_recommendations || 0).toLocaleString()} recs`;
      }
    } catch (e) {
      console.warn('Precomputed DB health warning:', e);
    }
  }

  async function refreshLiveDbHealth() {
    try {
      const targetDs = dom.liveDsSelect?.value;
      if (!targetDs) return;
      const data = await Api.dbHealth(targetDs);

      const pgBadge = document.getElementById('live-pg-status-badge');
      const totalArts = document.getElementById('live-db-total-articles');
      const totalVecs = document.getElementById('live-db-total-vectors');

      if (pgBadge) {
        pgBadge.className = data.connected ? 'stat-badge ready' : 'stat-badge error';
        pgBadge.innerHTML = data.connected ? `🟢 Connected (${data.schema})` : '🔴 Disconnected';
      }
      if (totalArts) totalArts.textContent = (data.total_articles || 0).toLocaleString();
      if (totalVecs) totalVecs.textContent = (data.embedded_articles || 0).toLocaleString();

      if (dom.liveSchemaBadge) {
        dom.liveSchemaBadge.textContent = data.schema || targetDs;
      }
      if (dom.liveStatsInline) {
        dom.liveStatsInline.textContent = `${(data.total_articles || 0).toLocaleString()} arts | ${(data.embedded_articles || 0).toLocaleString()} embedded`;
      }
    } catch (e) {
      console.warn('Live DB health warning:', e);
    }
  }

  // ─── Dataset Actions (Create / Clone / Delete) ─────────────────────────────
  async function handleCreatePrecompDataset(e) {
    e.preventDefault();
    const name = document.getElementById('new-ds-name')?.value.trim();
    const customId = document.getElementById('new-ds-id')?.value.trim();
    const desc = document.getElementById('new-ds-desc')?.value.trim();
    if (!name) return;

    try {
      showToast(`Creating isolated precompute run "${name}"...`, 'info');
      const res = await Api.createDataset({ name, description: desc, datasetId: customId || undefined, pipelineType: 'precomputed' });
      showToast(`Precomputed run "${res.dataset.name}" created!`, 'success');
      if (dom.createPrecompCard) dom.createPrecompCard.style.display = 'none';
      if (dom.createPrecompForm) dom.createPrecompForm.reset();
      await loadPrecomputedDatasets();
      if (dom.precompDsSelect) dom.precompDsSelect.value = res.dataset.id;
      State.set({ activeDataset: res.dataset.id });
      await refreshPrecompDbHealth();
      loadInspectorArticles();
      loadRecentPrecompJobs();
    } catch (err) {
      showToast(`Failed to create precompute run: ${err.message}`, 'error');
    }
  }

  async function handleCloneCorpus() {
    const targetId = dom.precompDsSelect?.value;
    const sourceId = dom.cloneSourceSelect?.value || 'default';
    const withEmbeddings = dom.cloneWithEmbeddings?.checked ?? true;
    if (!targetId || targetId === 'default') {
      showToast('Please select an isolated experimental run as target (cannot clone into default).', 'warn');
      return;
    }

    try {
      showToast(`Cloning articles from "${sourceId}" to "${targetId}"...`, 'info');
      const res = await Api.cloneDataset(targetId, sourceId, withEmbeddings);
      showToast(`Cloned ${res.articles_copied} articles into "${targetId}"!`, 'success');
      if (dom.cloneCorpusCard) dom.cloneCorpusCard.style.display = 'none';
      await refreshPrecompDbHealth();
      loadInspectorArticles();
    } catch (err) {
      showToast(`Clone failed: ${err.message}`, 'error');
    }
  }

  async function handleDeletePrecompDataset() {
    const dsId = dom.precompDsSelect?.value;
    if (!dsId || dsId === 'default') {
      showToast('The default precomputed dataset is protected and cannot be deleted.', 'warn');
      return;
    }
    if (!confirm(`Are you sure you want to permanently delete precomputed run "${dsId}" and drop all its tables?`)) {
      return;
    }

    try {
      showToast(`Deleting run "${dsId}"...`, 'info');
      await Api.deleteDataset(dsId);
      showToast(`Run "${dsId}" deleted.`, 'success');
      State.set({ activeDataset: 'default' });
      await loadPrecomputedDatasets();
      if (dom.precompDsSelect) dom.precompDsSelect.value = 'default';
      await refreshPrecompDbHealth();
      loadInspectorArticles();
      loadRecentPrecompJobs();
    } catch (err) {
      showToast(`Delete failed: ${err.message}`, 'error');
    }
  }

  async function handleCreateLiveDataset(e) {
    e.preventDefault();
    const name = document.getElementById('new-live-ds-name')?.value.trim();
    const customId = document.getElementById('new-live-ds-id')?.value.trim();
    const desc = document.getElementById('new-live-ds-desc')?.value.trim();
    if (!name) return;

    try {
      showToast(`Creating isolated live dataset "${name}"...`, 'info');
      const res = await Api.createDataset({ name, description: desc, datasetId: customId || undefined, pipelineType: 'live' });
      showToast(`Live dataset "${res.dataset.name}" created (Recs table omitted)!`, 'success');
      if (dom.createLiveCard) dom.createLiveCard.style.display = 'none';
      if (dom.createLiveForm) dom.createLiveForm.reset();
      await loadLiveDatasets();
      if (dom.liveDsSelect) dom.liveDsSelect.value = res.dataset.id;
      await refreshLiveDbHealth();
      loadRecentLiveJobs();
    } catch (err) {
      showToast(`Failed to create live dataset: ${err.message}`, 'error');
    }
  }

  async function handleDeleteLiveDataset() {
    const dsId = dom.liveDsSelect?.value;
    if (!dsId) return;
    if (!confirm(`Are you sure you want to permanently delete live dataset "${dsId}"?`)) return;

    try {
      showToast(`Deleting live dataset "${dsId}"...`, 'info');
      await Api.deleteDataset(dsId);
      showToast(`Live dataset "${dsId}" deleted.`, 'success');
      await loadLiveDatasets();
      await refreshLiveDbHealth();
      loadRecentLiveJobs();
    } catch (err) {
      showToast(`Delete failed: ${err.message}`, 'error');
    }
  }

  // ─── Precomputed Pipeline Ingestion Handlers ──────────────────────────────
  function onPrecompFileChosen(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showToast('Please select a valid CSV file (.csv)', 'warn');
      return;
    }
    selectedPrecompFile = file;
    if (dom.precompPreview && dom.precompPreviewName) {
      dom.precompPreviewName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
      dom.precompPreview.classList.remove('hidden');
    }
    if (dom.precompUploadBtn) {
      dom.precompUploadBtn.disabled = false;
      dom.precompUploadBtn.textContent = `Run Ingestion on ${file.name.slice(0, 18)}...`;
    }
  }

  async function handlePrecompUpload() {
    if (!selectedPrecompFile) return;
    if (dom.precompUploadBtn) dom.precompUploadBtn.disabled = true;
    if (dom.precompDemoBtn) dom.precompDemoBtn.disabled = true;
    const targetDs = dom.precompDsSelect?.value || State.get().activeDataset || 'default';
    showToast(`Uploading CSV to precompute run "${targetDs}"...`, 'info');

    try {
      const res = await Api.uploadCsv(selectedPrecompFile, targetDs);
      activePrecompJobId = res.job_id;
      showToast(`Pipeline started: ${res.job_id} in ${targetDs}`, 'success');
      startPrecompPipelineMonitor(res.job_id);
    } catch (err) {
      showToast(`Upload failed: ${err.message}`, 'error');
      if (dom.precompUploadBtn) dom.precompUploadBtn.disabled = false;
      if (dom.precompDemoBtn) dom.precompDemoBtn.disabled = false;
    }
  }

  async function handleDemoIngest() {
    if (dom.precompDemoBtn) {
      dom.precompDemoBtn.disabled = true;
      dom.precompDemoBtn.textContent = 'Starting Demo Pipeline...';
    }
    if (dom.precompUploadBtn) dom.precompUploadBtn.disabled = true;
    const targetDs = dom.precompDsSelect?.value || State.get().activeDataset || 'default';
    showToast(`Starting sample dataset ingestion (100 demo) into "${targetDs}"...`, 'info');

    try {
      const res = await Api.ingestDefault(100, targetDs);
      activePrecompJobId = res.job_id;
      showToast(`Demo pipeline active: ${res.job_id}`, 'success');
      startPrecompPipelineMonitor(res.job_id);
    } catch (err) {
      showToast(`Error: ${err.message}`, 'error');
      if (dom.precompDemoBtn) {
        dom.precompDemoBtn.disabled = false;
        dom.precompDemoBtn.textContent = '⚡ Ingest Sample (100 Demo)';
      }
      if (dom.precompUploadBtn && selectedPrecompFile) dom.precompUploadBtn.disabled = false;
    }
  }

  async function handleManualPrecompSubmit(e) {
    e.preventDefault();
    if (!dom.manualAlert) return;
    dom.manualAlert.style.display = 'block';
    dom.manualAlert.className = 'alert alert-info';
    dom.manualAlert.innerHTML = '⏳ <strong>Publishing &amp; Precomputing...</strong> Generating 384d vectors and reranking top-10 candidates...';
    if (dom.manualSubmitBtn) dom.manualSubmitBtn.disabled = true;

    try {
      const targetDs = dom.precompDsSelect?.value || State.get().activeDataset || 'default';
      const payload = {
        title: document.getElementById('manual-title').value.trim(),
        section: document.getElementById('manual-section').value,
        subsection: document.getElementById('manual-subsection').value.trim(),
        date: document.getElementById('manual-date').value ? new Date(document.getElementById('manual-date').value).toISOString() : new Date().toISOString(),
        page_views: parseInt(document.getElementById('manual-views').value, 10) || 0,
        description: document.getElementById('manual-description').value.trim(),
        body: document.getElementById('manual-body').value.trim(),
        url: document.getElementById('manual-url').value.trim(),
        image: document.getElementById('manual-image').value.trim(),
        precompute_recommendations: document.getElementById('manual-precompute').checked,
        reverse_push: document.getElementById('manual-reverse-push').checked,
        dataset: targetDs,
      };

      const res = await Api.addManualArticle(payload);
      dom.manualAlert.className = 'alert alert-success';
      dom.manualAlert.innerHTML = `
        <div style="font-weight:700;margin-bottom:4px;">🎉 Article Ingested &amp; Precomputed Successfully!</div>
        <div style="font-size:0.83rem;">
          <strong>ID:</strong> ${res.article_id} | <strong>Run:</strong> ${targetDs} | <strong>Section:</strong> ${res.article.section}<br/>
          <strong>Precomputed Recs:</strong> ${res.recommendations_count} saved | 
          <strong>Matching Archive Stories Refreshed:</strong> ${res.refreshed_old_articles}
        </div>
        <div style="margin-top:8px;">
          <button type="button" class="btn btn-secondary" style="font-size:0.75rem;padding:4px 10px;" onclick="Editor.inspectArticle('${res.article_id}')">
            🔍 Inspect Stored Evidence &amp; Recommendations
          </button>
        </div>
      `;

      await refreshPrecompDbHealth();
      loadInspectorArticles();
      loadRecentPrecompJobs();
    } catch (err) {
      dom.manualAlert.className = 'alert alert-danger';
      dom.manualAlert.innerHTML = `<strong>Error Publishing Article:</strong> ${err.message}`;
    } finally {
      if (dom.manualSubmitBtn) dom.manualSubmitBtn.disabled = false;
    }
  }

  // ─── Live Pipeline Ingestion Handlers ─────────────────────────────────────
  function onLiveFileChosen(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showToast('Please select a valid CSV file (.csv)', 'warn');
      return;
    }
    selectedLiveFile = file;
    if (dom.livePreview && dom.livePreviewName) {
      dom.livePreviewName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
      dom.livePreview.classList.remove('hidden');
    }
    if (dom.liveUploadBtn) {
      dom.liveUploadBtn.disabled = false;
      dom.liveUploadBtn.textContent = `🚀 Ingest ${file.name.slice(0, 18)} (Fast Embed)`;
    }
  }

  async function handleLiveCsvUpload() {
    if (!selectedLiveFile) return;
    const targetDs = dom.liveDsSelect?.value;
    if (!targetDs) {
      showToast('Please select or create a live dataset first.', 'warn');
      return;
    }
    const batchSize = parseInt(dom.liveBatchSelect?.value || '64', 10);
    if (dom.liveUploadBtn) dom.liveUploadBtn.disabled = true;

    showToast(`Uploading CSV to live dataset "${targetDs}" (Zero Rerank)...`, 'info');
    try {
      const res = await Api.uploadLiveCsv(selectedLiveFile, targetDs, batchSize);
      activeLiveJobId = res.job_id;
      showToast(`Fast live ingestion started: ${res.job_id}`, 'success');
      startLiveJobMonitor(res.job_id);
    } catch (err) {
      showToast(`Live upload failed: ${err.message}`, 'error');
      if (dom.liveUploadBtn) dom.liveUploadBtn.disabled = false;
    }
  }

  async function handleManualLiveSubmit(e) {
    e.preventDefault();
    if (!dom.manualLiveAlert) return;
    const targetDs = dom.liveDsSelect?.value;
    if (!targetDs) {
      showToast('Please select or create a live dataset first.', 'warn');
      return;
    }

    dom.manualLiveAlert.style.display = 'block';
    dom.manualLiveAlert.className = 'alert alert-info';
    dom.manualLiveAlert.innerHTML = '⏳ <strong>Ingesting Article...</strong> Generating 384d embedding vector...';
    if (dom.manualLiveSubmitBtn) dom.manualLiveSubmitBtn.disabled = true;

    try {
      const payload = {
        title: document.getElementById('manual-live-title').value.trim(),
        section: document.getElementById('manual-live-section').value,
        subsection: document.getElementById('manual-live-subsection').value.trim(),
        date: document.getElementById('manual-live-date').value ? new Date(document.getElementById('manual-live-date').value).toISOString() : new Date().toISOString(),
        page_views: parseInt(document.getElementById('manual-live-views').value, 10) || 0,
        description: document.getElementById('manual-live-description').value.trim(),
        body: document.getElementById('manual-live-body').value.trim(),
        url: document.getElementById('manual-live-url').value.trim(),
        image: document.getElementById('manual-live-image').value.trim(),
        dataset: targetDs,
      };

      const res = await Api.addManualLiveArticle(payload);
      dom.manualLiveAlert.className = 'alert alert-success';
      dom.manualLiveAlert.innerHTML = `
        <div style="font-weight:700;margin-bottom:4px;">⚡ Article Ingested &amp; Embedded Successfully!</div>
        <div style="font-size:0.83rem;">
          <strong>ID:</strong> ${res.article_id} | <strong>Live Dataset:</strong> ${targetDs} | <strong>Execution:</strong> ${res.duration_ms} ms<br/>
          <em>Article is immediately searchable in live semantic and hybrid search without precomputation table overhead.</em>
        </div>
      `;

      await refreshLiveDbHealth();
      loadRecentLiveJobs();
    } catch (err) {
      dom.manualLiveAlert.className = 'alert alert-danger';
      dom.manualLiveAlert.innerHTML = `<strong>Error Publishing Article:</strong> ${err.message}`;
    } finally {
      if (dom.manualLiveSubmitBtn) dom.manualLiveSubmitBtn.disabled = false;
    }
  }

  async function handleLivePreprocessing() {
    const targetDs = dom.liveDsSelect?.value;
    if (!targetDs) {
      showToast('Please select or create a live dataset first.', 'warn');
      return;
    }

    const batchSize = parseInt(dom.livePrepBatch?.value || '64', 10);
    const recomputeAll = !!dom.livePrepRecompute?.checked;

    if (dom.startLivePrepBtn) {
      dom.startLivePrepBtn.disabled = true;
      dom.startLivePrepBtn.textContent = '⚡ Running Live Preprocessing...';
    }
    if (dom.livePrepProgress) dom.livePrepProgress.style.display = 'block';
    if (dom.livePrepStatusText) dom.livePrepStatusText.textContent = `Generating embeddings for dataset "${targetDs}"...`;
    if (dom.livePrepProgressFill) dom.livePrepProgressFill.style.width = '20%';
    if (dom.livePrepProgressPct) dom.livePrepProgressPct.textContent = '20%';

    try {
      const res = await Api.preprocessLive({ dataset: targetDs, batchSize, recomputeAll });
      showToast(`Started live preprocessing job: ${res.job_id}`, 'info');
      startLiveJobMonitor(res.job_id);
    } catch (err) {
      showToast(`Preprocessing error: ${err.message}`, 'error');
      if (dom.livePrepStatusText) dom.livePrepStatusText.textContent = `Error: ${err.message}`;
      if (dom.startLivePrepBtn) {
        dom.startLivePrepBtn.disabled = false;
        dom.startLivePrepBtn.textContent = '⚡ Run Live Preprocessing';
      }
    }
  }

  // ─── Pipeline Monitors ────────────────────────────────────────────────────
  function startPrecompPipelineMonitor(jobId) {
    if (dom.visualizer) dom.visualizer.classList.remove('hidden');
    if (dom.completion) dom.completion.classList.add('hidden');
    if (dom.logWindow) dom.logWindow.innerHTML = '';
    precompLogged.clear();

    resetPipelineNodes();
    appendPrecompLog(`[Pipeline Started] Task ${jobId} initiated in PostgreSQL.`);

    if (precompPollInterval) clearInterval(precompPollInterval);

    precompPollInterval = setInterval(async () => {
      try {
        const job = await Api.jobStatus(jobId);
        if (dom.progressPct) dom.progressPct.textContent = `${job.progress_percent}%`;
        if (dom.progressFill) dom.progressFill.style.width = `${job.progress_percent}%`;

        const rawMsg = job.message || '';
        const stage = rawMsg.startsWith('STAGE:') ? rawMsg.slice(6).split('|')[0] : 'PARSE';
        const cleanMsg = rawMsg.includes('|') ? rawMsg.split('|')[1] : rawMsg;

        if (dom.statusMsg) dom.statusMsg.textContent = cleanMsg;
        if (!precompLogged.has(cleanMsg)) {
          precompLogged.add(cleanMsg);
          appendPrecompLog(cleanMsg);
        }

        updatePipelineNodes(stage);

        if (job.status === 'completed') {
          clearInterval(precompPollInterval);
          precompPollInterval = null;
          setAllNodesDone();
          const refreshed = job.refreshed_articles || 0;
          appendPrecompLog('🎉 [Success] All articles and top 10 recommendations stored in PostgreSQL.');
          if (refreshed > 0) {
            appendPrecompLog(`🔄 [Reverse Update] Refreshed recommendations for ${refreshed} existing article(s).`);
          }

          if (dom.completion) {
            dom.completion.classList.remove('hidden');
            const summary = document.getElementById('completion-summary-text');
            if (summary) {
              const durStr = formatDuration(job.duration_seconds);
              const whenStr = formatWhen(job.created_at);
              const fileName = job.filename || selectedPrecompFile?.name || 'CSV Batch Upload';
              summary.innerHTML = `
                <div style="margin-bottom:6px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                  <span><strong>📄 File:</strong> ${escHtml(fileName)}</span>
                  <span>&middot;</span>
                  <span><strong>📅 Uploaded:</strong> ${whenStr}</span>
                  <span>&middot;</span>
                  <span class="badge" style="background:rgba(99,102,241,0.15);color:#818cf8;border:1px solid rgba(99,102,241,0.3);padding:3px 8px;font-weight:700;">
                    ⏱️ Time Taken: ${durStr}
                  </span>
                </div>
                <div style="font-size:0.83rem;color:var(--text-secondary);">
                  Indexed ${job.total_articles} articles in PostgreSQL with 384d vectors and top-10 precomputed recommendations. Refreshed ${refreshed} archive stories.
                </div>
              `;
            }
          }

          showToast('Precomputed Run Finished! Recommendations are ready.', 'success');
          await refreshPrecompDbHealth();
          loadInspectorArticles();
          loadRecentPrecompJobs();
          if (dom.precompUploadBtn && selectedPrecompFile) dom.precompUploadBtn.disabled = false;
        } else if (job.status === 'failed') {
          clearInterval(precompPollInterval);
          precompPollInterval = null;
          appendPrecompLog(`❌ [Failed] ${job.error || 'Pipeline encountered an error.'}`);
          showToast(`Pipeline failed: ${job.error}`, 'error');
          if (dom.precompUploadBtn && selectedPrecompFile) dom.precompUploadBtn.disabled = false;
        }
      } catch (e) {
        console.warn('Precomp poll error:', e);
      }
    }, 1200);
  }

  function startLiveJobMonitor(jobId) {
    if (livePollInterval) clearInterval(livePollInterval);

    livePollInterval = setInterval(async () => {
      try {
        const job = await Api.jobStatus(jobId);
        const pct = job.progress_percent || 0;
        if (dom.livePrepProgressFill) dom.livePrepProgressFill.style.width = `${pct}%`;
        if (dom.livePrepProgressPct) dom.livePrepProgressPct.textContent = `${pct}%`;
        if (dom.livePrepStatusText) dom.livePrepStatusText.textContent = job.message || 'Processing live embeddings...';

        if (job.status === 'completed') {
          clearInterval(livePollInterval);
          livePollInterval = null;
          if (dom.livePrepProgressFill) dom.livePrepProgressFill.style.width = '100%';
          if (dom.livePrepProgressPct) dom.livePrepProgressPct.textContent = '100%';
          const durStr = formatDuration(job.duration_seconds);
          const whenStr = formatWhen(job.created_at);
          const fileName = job.filename || selectedLiveFile?.name || 'Live Fast Batch Upload';
          if (dom.livePrepStatusText) {
            dom.livePrepStatusText.innerHTML = `
              🎉 <strong>Completed!</strong> 📄 ${escHtml(fileName)} &middot; 📅 ${whenStr} &middot;
              <span class="badge" style="background:rgba(13,148,136,0.15);color:#2dd4bf;border:1px solid rgba(13,148,136,0.3);padding:2px 7px;font-weight:700;">
                ⏱️ Time Taken: ${durStr}
              </span>
            `;
          }
          showToast(`Live upload completed in ${durStr}!`, 'success');
          if (dom.startLivePrepBtn) {
            dom.startLivePrepBtn.disabled = false;
            dom.startLivePrepBtn.textContent = '⚡ Run Live Preprocessing';
          }
          if (dom.liveUploadBtn && selectedLiveFile) dom.liveUploadBtn.disabled = false;
          await refreshLiveDbHealth();
          loadRecentLiveJobs();
        } else if (job.status === 'failed') {
          clearInterval(livePollInterval);
          livePollInterval = null;
          showToast(`Live job failed: ${job.error}`, 'error');
          if (dom.startLivePrepBtn) {
            dom.startLivePrepBtn.disabled = false;
            dom.startLivePrepBtn.textContent = '⚡ Run Live Preprocessing';
          }
          if (dom.liveUploadBtn && selectedLiveFile) dom.liveUploadBtn.disabled = false;
          loadRecentLiveJobs();
        }
      } catch (e) {
        console.warn('Live poll error:', e);
      }
    }, 1000);
  }

  function resetPipelineNodes() {
    ['node-parse', 'node-clean', 'node-storage', 'node-embed', 'node-rerank', 'node-reverse'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.className = 'pipeline-node';
        const st = el.querySelector('.node-status');
        if (st) st.textContent = 'Waiting';
      }
    });
  }

  function updatePipelineNodes(stage) {
    const stageMap = {
      'PARSE':     { curr: 'node-parse',   done: [] },
      'CLEAN':     { curr: 'node-clean',   done: ['node-parse'] },
      'STORAGE':   { curr: 'node-storage', done: ['node-parse', 'node-clean'] },
      'EMBEDDING': { curr: 'node-embed',   done: ['node-parse', 'node-clean', 'node-storage'] },
      'INDEX':     { curr: 'node-rerank',  done: ['node-parse', 'node-clean', 'node-storage', 'node-embed'] },
      'PGVECTOR':  { curr: 'node-rerank',  done: ['node-parse', 'node-clean', 'node-storage', 'node-embed'] },
      'RERANK':    { curr: 'node-rerank',  done: ['node-parse', 'node-clean', 'node-storage', 'node-embed'] },
      'REVERSE':   { curr: 'node-reverse', done: ['node-parse', 'node-clean', 'node-storage', 'node-embed', 'node-rerank'] },
      'COMPLETE':  { curr: null,           done: ['node-parse', 'node-clean', 'node-storage', 'node-embed', 'node-rerank', 'node-reverse'] },
    };

    const conf = stageMap[stage] || stageMap['PARSE'];
    conf.done.forEach(id => {
      const el = document.getElementById(id);
      if (el && !el.classList.contains('done')) {
        el.className = 'pipeline-node done';
        const st = el.querySelector('.node-status');
        if (st) st.textContent = '✓ Done';
      }
    });

    if (conf.curr) {
      const el = document.getElementById(conf.curr);
      if (el) {
        el.className = 'pipeline-node active';
        const st = el.querySelector('.node-status');
        if (st) st.textContent = '⚡ Running...';
      }
    }
  }

  function setAllNodesDone() {
    ['node-parse', 'node-clean', 'node-storage', 'node-embed', 'node-rerank', 'node-reverse'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.className = 'pipeline-node done';
        const st = el.querySelector('.node-status');
        if (st) st.textContent = '✓ Stored';
      }
    });
  }

  function appendPrecompLog(msg) {
    if (!dom.logWindow) return;
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = 'log-entry';
    line.innerHTML = `<span class="log-time">[${time}]</span> <span>${msg}</span>`;
    dom.logWindow.appendChild(line);
    dom.logWindow.scrollTop = dom.logWindow.scrollHeight;
  }

  function formatWhen(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '—';
    return d.toLocaleString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    });
  }

  function formatDuration(sec) {
    if (sec == null || isNaN(sec)) return '—';
    sec = Math.round(Number(sec) * 10) / 10;
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60);
    const remS = Math.round(sec % 60);
    if (m < 60) return `${m}m ${remS.toString().padStart(2, '0')}s`;
    const h = Math.floor(m / 60);
    const remM = m % 60;
    return `${h}h ${remM}m ${remS.toString().padStart(2, '0')}s`;
  }

  // ─── Recent Jobs Loaders ──────────────────────────────────────────────────
  async function loadRecentPrecompJobs() {
    if (!dom.recentJobsBody) return;
    try {
      const targetDs = dom.precompDsSelect?.value || State.get().activeDataset || 'default';
      const data = await Api.recentJobs(targetDs, 'precomputed');
      const jobs = data.jobs || [];
      const summary = data.summary || {};

      const summaryEls = {
        uploads:   document.getElementById('summary-uploads'),
        articles:  document.getElementById('summary-articles'),
        refreshed: document.getElementById('summary-refreshed'),
        failed:    document.getElementById('summary-failed'),
      };
      if (summaryEls.uploads)   summaryEls.uploads.textContent   = summary.total_jobs ?? jobs.length;
      if (summaryEls.articles)  summaryEls.articles.textContent  = (summary.articles_ingested || 0).toLocaleString();
      if (summaryEls.refreshed) summaryEls.refreshed.textContent = (summary.refreshed_articles || 0).toLocaleString();
      if (summaryEls.failed)    summaryEls.failed.textContent    = summary.failed_jobs || 0;

      if (!jobs.length) {
        dom.recentJobsBody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:14px;">No precomputed runs recorded yet.</td></tr>';
        return;
      }

      dom.recentJobsBody.innerHTML = jobs.map(job => {
        const badgeClass = job.status === 'completed' ? 'ready' : (job.status === 'failed' ? 'error' : 'warning');
        const refreshed = job.refreshed_articles || 0;
        const articlesCell = job.status === 'completed' ? `${job.total_articles}` : `${job.processed_articles}/${job.total_articles || '?'}`;
        const durationStr = formatDuration(job.duration_seconds);
        const uploadDate = formatWhen(job.created_at);
        const fileName = job.filename || 'Direct Batch Upload';

        return `
          <tr>
            <td>
              <div style="font-weight:600;font-size:0.79rem;color:var(--text-primary);white-space:nowrap;">${uploadDate}</div>
            </td>
            <td>
              <div class="job-file" style="font-weight:700;color:var(--text-primary);display:flex;align-items:center;gap:6px;">
                <span>📄</span> ${escHtml(fileName)}
              </div>
              <div class="job-id" style="font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono);">${escHtml(job.id)}</div>
            </td>
            <td><strong>${articlesCell}</strong></td>
            <td>
              <span class="badge" style="background:rgba(99,102,241,0.12);color:#818cf8;border:1px solid rgba(99,102,241,0.25);font-weight:700;font-size:0.78rem;padding:3px 8px;display:inline-flex;align-items:center;gap:4px;white-space:nowrap;">
                <span>⏱️</span> ${durationStr}
              </span>
            </td>
            <td><span class="job-refresh-count ${refreshed > 0 ? '' : 'zero'}">${refreshed > 0 ? refreshed : '—'}</span></td>
            <td><span class="stat-badge ${badgeClass}">${job.status}</span></td>
            <td>
              <button type="button" class="btn btn-secondary btn-inspect-job-mon" data-id="${job.id}" style="font-size:0.72rem;padding:3px 8px;">
                📊 Telemetry
              </button>
            </td>
          </tr>
        `;
      }).join('');

      dom.recentJobsBody.querySelectorAll('.btn-inspect-job-mon').forEach(btn => {
        btn.addEventListener('click', () => {
          if (window.setAppRole) {
            window.setAppRole('monitor');
            setTimeout(() => {
              if (window.Monitoring && typeof window.Monitoring.refreshAll === 'function') {
                window.Monitoring.refreshAll();
              }
            }, 100);
          }
        });
      });
    } catch (e) {
      console.warn('Recent precomp jobs warning:', e);
    }
  }

  async function loadRecentLiveJobs() {
    if (!dom.liveJobsBody) return;
    try {
      const targetDs = dom.liveDsSelect?.value;
      if (!targetDs) {
        dom.liveJobsBody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:14px;">Select or create a live dataset above.</td></tr>';
        return;
      }
      const data = await Api.recentJobs(targetDs, 'live');
      const jobs = data.jobs || [];

      if (!jobs.length) {
        dom.liveJobsBody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:14px;">No live ingestion jobs in this dataset yet.</td></tr>';
        return;
      }

      dom.liveJobsBody.innerHTML = jobs.map(job => {
        const badgeClass = job.status === 'completed' ? 'ready' : (job.status === 'failed' ? 'error' : 'warning');
        const durationStr = formatDuration(job.duration_seconds);
        const uploadDate = formatWhen(job.created_at);
        const fileName = job.filename || 'Live Fast Batch Upload';
        const throughput = (job.total_articles && job.duration_seconds)
          ? `${(job.total_articles / Math.max(job.duration_seconds, 0.01)).toFixed(1)} art/s`
          : '—';

        return `
          <tr>
            <td>
              <div style="font-weight:600;font-size:0.79rem;color:var(--text-primary);white-space:nowrap;">${uploadDate}</div>
            </td>
            <td>
              <div class="job-file" style="font-weight:700;color:#2dd4bf;display:flex;align-items:center;gap:6px;">
                <span>🚀</span> ${escHtml(fileName)}
              </div>
              <div class="job-id" style="font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono);">${escHtml(job.id)}</div>
            </td>
            <td><strong>${job.processed_articles || job.total_articles || 0}</strong></td>
            <td>
              <span class="badge" style="background:rgba(13,148,136,0.12);color:#2dd4bf;border:1px solid rgba(13,148,136,0.25);font-weight:700;font-size:0.78rem;padding:3px 8px;display:inline-flex;align-items:center;gap:4px;white-space:nowrap;">
                <span>⏱️</span> ${durationStr}
              </span>
            </td>
            <td><span style="color:#34d399;font-weight:700;font-size:0.78rem;">${throughput}</span></td>
            <td><span class="stat-badge ${badgeClass}">${job.status}</span></td>
            <td>
              <button type="button" class="btn btn-secondary btn-inspect-live-mon" data-id="${job.id}" style="font-size:0.72rem;padding:3px 8px;">
                📊 Telemetry
              </button>
            </td>
          </tr>
        `;
      }).join('');

      dom.liveJobsBody.querySelectorAll('.btn-inspect-live-mon').forEach(btn => {
        btn.addEventListener('click', () => {
          if (window.setAppRole) {
            window.setAppRole('monitor');
            setTimeout(() => {
              if (window.Monitoring && typeof window.Monitoring.refreshAll === 'function') {
                window.Monitoring.refreshAll();
              }
            }, 100);
          }
        });
      });
    } catch (e) {
      console.warn('Recent live jobs warning:', e);
    }
  }

  // ─── Inspector Logic ──────────────────────────────────────────────────────
  async function loadInspectorArticles() {
    if (!dom.inspectorSelect) return;
    try {
      const targetDs = dom.precompDsSelect?.value || State.get().activeDataset || 'default';
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = `-- Choose an article to inspect precomputed recs --`;
      dom.inspectorSelect.innerHTML = '';
      dom.inspectorSelect.appendChild(placeholder);

      const perPage = 100;
      let page = 1;
      let count = 0;
      while (page <= 5) {
        const data = await Api.articles({ page, perPage, sort: 'date', dataset: targetDs });
        const arts = data.articles || [];
        arts.forEach(art => {
          const opt = document.createElement('option');
          opt.value = art.id;
          opt.textContent = `[${art.section || 'General'}] ${art.title.slice(0, 80)}`;
          dom.inspectorSelect.appendChild(opt);
        });
        count += arts.length;
        if (arts.length < perPage) break;
        page += 1;
      }
      placeholder.textContent = `-- Choose an article to inspect precomputed recs (${count}) --`;
    } catch (e) {
      console.warn('Inspector list warning:', e);
    }
  }

  async function inspectArticle(articleId) {
    if (!dom.inspectorCandidatesList) return;
    dom.inspectorCandidatesList.innerHTML = '<div style="text-align:center;padding:24px;color:var(--text-secondary);">Fetching precomputed recommendations from PostgreSQL…</div>';
    if (dom.inspectorSourceBox) dom.inspectorSourceBox.style.display = 'none';
    if (dom.inspectorActions) dom.inspectorActions.style.display = 'none';

    try {
      const targetDs = dom.precompDsSelect?.value || State.get().activeDataset || 'default';
      const res = await Api.precomputedRecs(articleId, 10, targetDs);
      const recs = res.recommendations || [];
      const sourceArt = res.source_article || null;

      if (!recs.length) {
        dom.inspectorCandidatesList.innerHTML = '';
        if (dom.inspectorEmpty) dom.inspectorEmpty.classList.remove('hidden');
        return;
      }

      if (dom.inspectorEmpty) dom.inspectorEmpty.classList.add('hidden');
      if (dom.inspectorActions) dom.inspectorActions.style.display = 'block';

      if (window.Evaluator && window.Evaluator.renderReviewGroup) {
        dom.inspectorCandidatesList.innerHTML = '';
        const group = window.Evaluator.renderReviewGroup(sourceArt, recs, {
          pipeline: res.model_version || '384d',
          metadata: res.scoring_metadata || null,
          openFirst: 2,
        });
        dom.inspectorCandidatesList.appendChild(group);
      }
    } catch (err) {
      dom.inspectorCandidatesList.innerHTML = `<div style="color:var(--color-danger);text-align:center;padding:24px;">Failed to fetch recommendations: ${err.message}</div>`;
    }
  }

  return {
    init,
    refreshDbHealth: refreshPrecompDbHealth,
    refreshPrecompDbHealth,
    refreshLiveDbHealth,
    inspectArticle,
    loadPrecomputedDatasets,
    loadLiveDatasets,
    switchStudio,
  };
})();

window.Editor = Editor;
