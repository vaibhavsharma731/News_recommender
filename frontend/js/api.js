/**
 * api.js — All HTTP calls to the FastAPI backend.
 * Centralised so the URL is changed in exactly one place.
 */
const API_BASE = (window.location.protocol.startsWith('http'))
  ? `${window.location.origin}/api`
  : 'http://localhost:5000/api';

const Api = (() => {

  async function _fetch(path, opts = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      let msg = err.error || err.message || `HTTP ${res.status}`;
      if (typeof err.detail === 'string') {
        msg = err.detail;
      } else if (Array.isArray(err.detail)) {
        msg = err.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
      }
      throw new Error(msg);
    }
    return res.json();
  }

  /** Engine status + config */
  async function status(pipeline = '', dataset = 'default') {
    const pl = pipeline || (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    return _fetch(`/status?pipeline=${encodeURIComponent(pl)}&dataset=${encodeURIComponent(ds)}`);
  }

  /** Trending / hero articles */
  async function trending(n = 6, pipeline = '', dataset = 'default') {
    const pl = pipeline || (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    return _fetch(`/articles/trending?n=${n}&pipeline=${encodeURIComponent(pl)}&dataset=${encodeURIComponent(ds)}`);
  }

  /** Paginated article list */
  async function articles({ page = 1, perPage = 20, section = '', sort = 'date', pipeline = '', search = '', dataset = 'default' } = {}) {
    const pl = pipeline || (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    const params = new URLSearchParams({ page, per_page: perPage, section, sort, pipeline: pl, dataset: ds });
    if (search) params.append('search', search);
    return _fetch(`/articles?${params}`);
  }

  /** All sections */
  async function sections(pipeline = '', dataset = 'default') {
    const pl = pipeline || (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    return _fetch(`/sections?pipeline=${encodeURIComponent(pl)}&dataset=${encodeURIComponent(ds)}`);
  }

  /**
   * Recommend endpoint: supports query or articleId across datasets.
   */
  async function recommend({
    query      = null,
    articleId  = null,
    dataset    = 'default',
    topK       = 10,
    method     = 'hybrid+rerank',
    useBoosts  = true,
    diversify    = true,
    calibrate    = true,
    section      = null,
    modelVersion = '384d',
    candidateStrategy = 'quota',
  } = {}) {
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    return _fetch('/recommend', {
      method: 'POST',
      body: JSON.stringify({
        query,
        article_id:         articleId,
        dataset:            ds,
        top_k:              topK,
        method,
        use_boosts:         useBoosts,
        diversify,
        calibrate,
        section,
        model_version:      modelVersion,
        candidate_strategy: candidateStrategy,
      }),
    });
  }

  /** PostgreSQL & Editor Endpoints */
  async function dbHealth(dataset = 'default') {
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    return _fetch(`/db/health?dataset=${encodeURIComponent(ds)}`);
  }

  async function dbInit() {
    return _fetch('/db/init', { method: 'POST' });
  }

  async function dbClear(dataset = 'default') {
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    return _fetch(`/db/clear?dataset=${encodeURIComponent(ds)}`, { method: 'POST' });
  }

  async function uploadCsv(file, dataset = 'default') {
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/db/upload-csv?dataset=${encodeURIComponent(ds)}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'CSV upload failed');
    }
    return res.json();
  }

  async function jobStatus(jobId) {
    return _fetch(`/db/jobs/${encodeURIComponent(jobId)}`);
  }

  async function recentJobs(dataset = 'default', pipelineType = null) {
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    const params = new URLSearchParams({ dataset: ds });
    if (pipelineType) params.append('pipeline_type', pipelineType);
    return _fetch(`/db/jobs?${params}`);
  }

  async function ingestDefault(limit = null, dataset = 'default') {
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    const query = `?dataset=${encodeURIComponent(ds)}${limit ? `&limit=${limit}` : ''}`;
    const res = await fetch(`${API_BASE}/db/ingest-default${query}`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Default dataset ingest failed');
    }
    return res.json();
  }

  /** Precomputed recommendations lookup — sub-5ms from PostgreSQL */
  async function precomputedRecs(articleId, limit = 10, dataset = 'default') {
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    return _fetch(`/db/recommendations/${encodeURIComponent(articleId)}?limit=${limit}&dataset=${encodeURIComponent(ds)}`);
  }

  /** Manual article addition directly into PostgreSQL with instant precomputation */
  async function addManualArticle(data) {
    return _fetch('/db/articles/manual', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /** Single article live addition directly into PostgreSQL (embed only, no reranker) */
  async function addManualLiveArticle(data) {
    return _fetch('/db/articles/manual-live', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /** Live CSV batch ingestion (embed only, no Cross-Encoder reranker) */
  async function uploadLiveCsv(file, dataset = 'default', batchSize = 64) {
    const ds = dataset || (typeof State !== 'undefined' ? State.get().activeDataset : 'default') || 'default';
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/db/upload-live-csv?dataset=${encodeURIComponent(ds)}&batch_size=${batchSize}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Live CSV upload failed');
    }
    return res.json();
  }

  /** Live Preprocessing (Pipeline B) — embedding generation without CE reranker */
  async function preprocessLive({ dataset = 'default', batchSize = 64, recomputeAll = false } = {}) {
    return _fetch('/db/preprocess-live', {
      method: 'POST',
      body: JSON.stringify({ dataset, batch_size: batchSize, recompute_all: recomputeAll }),
    });
  }

  /** Dataset Management Endpoints */
  async function listDatasets(pipelineType = null) {
    const params = pipelineType ? `?pipeline_type=${encodeURIComponent(pipelineType)}` : '';
    return _fetch(`/datasets${params}`);
  }

  async function getDataset(id) {
    return _fetch(`/datasets/${encodeURIComponent(id)}`);
  }

  async function createDataset({ name, description = '', datasetId = null, pipelineType = 'precomputed' }) {
    return _fetch('/datasets', {
      method: 'POST',
      body: JSON.stringify({ name, description, dataset_id: datasetId, pipeline_type: pipelineType }),
    });
  }

  async function deleteDataset(id) {
    return _fetch(`/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  async function cloneDataset(targetId, sourceId = 'default', withEmbeddings = true) {
    return _fetch(`/datasets/${encodeURIComponent(targetId)}/clone`, {
      method: 'POST',
      body: JSON.stringify({ source_dataset_id: sourceId, with_embeddings: withEmbeddings }),
    });
  }

  /** Performance Monitoring & Execution History Endpoints */
  async function monitoringResources() {
    return _fetch('/monitoring/resources/current');
  }

  async function monitoringSummary() {
    return _fetch('/monitoring/summary');
  }

  async function monitoringLiveStats() {
    return _fetch('/monitoring/live/stats');
  }

  async function monitoringRecords({ jobType = '', status = '', pipelineVersion = '', startDate = '', endDate = '', page = 1, perPage = 20 } = {}) {
    const params = new URLSearchParams({ page, per_page: perPage });
    if (jobType) params.append('job_type', jobType);
    if (status) params.append('status', status);
    if (pipelineVersion) params.append('pipeline_version', pipelineVersion);
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    return _fetch(`/monitoring/records?${params}`);
  }

  async function monitoringRecord(recordId) {
    return _fetch(`/monitoring/records/${encodeURIComponent(recordId)}`);
  }

  async function monitoringCompare(recordIds) {
    const ids = Array.isArray(recordIds) ? recordIds.join(',') : recordIds;
    return _fetch(`/monitoring/comparison?record_ids=${encodeURIComponent(ids)}`);
  }

  async function monitoringStorage() {
    return _fetch('/monitoring/database/storage');
  }

  async function monitoringCleanup(days = 90, keepTypes = ['ingestion']) {
    return _fetch('/monitoring/cleanup', {
      method: 'POST',
      body: JSON.stringify({ days, keep_job_types: keepTypes }),
    });
  }

  return {
    status,
    trending,
    articles,
    sections,
    recommend,
    dbHealth,
    dbInit,
    dbClear,
    uploadCsv,
    ingestDefault,
    jobStatus,
    recentJobs,
    precomputedRecs,
    addManualArticle,
    addManualLiveArticle,
    uploadLiveCsv,
    preprocessLive,
    listDatasets,
    getDataset,
    createDataset,
    deleteDataset,
    cloneDataset,
    monitoringResources,
    monitoringSummary,
    monitoringLiveStats,
    monitoringRecords,
    monitoringRecord,
    monitoringCompare,
    monitoringStorage,
    monitoringCleanup,
  };
})();


