/**
 * state.js — Single source of truth for all app state.
 * Components read from State and dispatch mutations here.
 */
const State = (() => {
  let _state = {
    userHistory:   [],   // array of article IDs
    topK:          10,
    method:        'hybrid+rerank',
    useBoosts:     true,
    diversify:     true,
    activeSection: '',   // '' = All
    activeTab:     'foryou',
    engineInfo:    null,
    allSections:   [],
    theme:         localStorage.getItem('theme') || 'dark',
  };

  const _listeners = [];

  function get() { return { ..._state }; }

  function set(updates) {
    _state = { ..._state, ...updates };
    _listeners.forEach(fn => fn(_state));
  }

  function subscribe(fn) { _listeners.push(fn); }

  // History helpers
  function addToHistory(id) {
    if (_state.userHistory.includes(id)) return false;
    const history = [..._state.userHistory, id];
    set({ userHistory: history });
    return true;
  }
  function clearHistory() { set({ userHistory: [] }); }

  // Persist history to sessionStorage
  function loadHistory() {
    try {
      const h = sessionStorage.getItem('newsrec_history');
      if (h) set({ userHistory: JSON.parse(h) });
    } catch (_) {}
  }
  function saveHistory() {
    try { sessionStorage.setItem('newsrec_history', JSON.stringify(_state.userHistory)); }
    catch (_) {}
  }

  return { get, set, subscribe, addToHistory, clearHistory, loadHistory, saveHistory };
})();
