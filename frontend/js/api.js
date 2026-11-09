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
  async function status(pipeline = '') {
    const pl = pipeline || (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
    return _fetch(`/status?pipeline=${encodeURIComponent(pl)}`);
  }

  /** Trending / hero articles */
  async function trending(n = 6, pipeline = '') {
    const pl = pipeline || (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
    return _fetch(`/articles/trending?n=${n}&pipeline=${encodeURIComponent(pl)}`);
  }

  /** Paginated article list */
  async function articles({ page = 1, perPage = 20, section = '', sort = 'date', pipeline = '' } = {}) {
    const pl = pipeline || (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
    const params = new URLSearchParams({ page, per_page: perPage, section, sort, pipeline: pl });
    return _fetch(`/articles?${params}`);
  }

  /** All sections */
  async function sections(pipeline = '') {
    const pl = pipeline || (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
    return _fetch(`/sections?pipeline=${encodeURIComponent(pl)}`);
  }

  /**
   * Recommend endpoint: supports query or articleId.
   */
  async function recommend({
    query      = null,
    articleId  = null,
    topK       = 10,
    method     = 'hybrid+rerank',
    useBoosts  = true,
    diversify    = true,
    section      = null,
    modelVersion = '384d',
  } = {}) {
    return _fetch('/recommend', {
      method: 'POST',
      body: JSON.stringify({
        query,
        article_id:    articleId,
        top_k:         topK,
        method,
        use_boosts:    useBoosts,
        diversify,
        section,
        model_version: modelVersion,
      }),
    });
  }

  /** PostgreSQL & Editor Endpoints */
  async function dbHealth() {
    return _fetch('/db/health');
  }

  async function dbInit() {
    return _fetch('/db/init', { method: 'POST' });
  }

  async function dbClear() {
    return _fetch('/db/clear', { method: 'POST' });
  }

  async function uploadCsv(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/db/upload-csv`, {
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

  async function recentJobs() {
    return _fetch('/db/jobs');
  }

  async function ingestDefault(limit = null) {
    const query = limit ? `?limit=${limit}` : '';
    const res = await fetch(`${API_BASE}/db/ingest-default${query}`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Default dataset ingest failed');
    }
    return res.json();
  }

  /** Precomputed recommendations lookup — sub-5ms from PostgreSQL */
  async function precomputedRecs(articleId, limit = 10) {
    return _fetch(`/db/recommendations/${encodeURIComponent(articleId)}?limit=${limit}`);
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
    precomputedRecs
  };
})();


