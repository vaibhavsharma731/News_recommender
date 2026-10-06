/**
 * state.js — Single source of truth for all app state.
 * Components read from State and dispatch mutations here.
 */
const State = (() => {
  let _state = {
    topK:          10,
    method:        'hybrid+rerank',
    useBoosts:     true,
    diversify:     true,
    activeSection: '',   // '' = All
    activeTab:     'similar',
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

  return { get, set, subscribe };
})();
