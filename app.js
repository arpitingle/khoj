// Set window.INDGOV_API_URL before app.js on hosted frontend deployments.
// Defaults to same-origin /api for local development.
const API = String(window.INDGOV_API_URL || '/api').replace(/\/+$/, '');
const form = document.querySelector('#search-form');
const queryInput = document.querySelector('#query');
const resultList = document.querySelector('#result-list');
const resultCount = document.querySelector('#result-count');
const noResults = document.querySelector('#no-results');
const searchError = document.querySelector('#search-error');
const statusLabel = document.querySelector('#data-status');
const categoryFilter = document.querySelector('#category-filter');
const stateFilter = document.querySelector('#state-filter');
const reviewFilter = document.querySelector('#include-review');
const loadMore = document.querySelector('#load-more');
let requestNumber = 0;
let activeLimit = 12;

const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

function excerpt(text, query, maxLength = 440) {
  const clean = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (clean.length <= maxLength) return clean;
  const words = query.toLowerCase().split(/\s+/).filter(word => word.length > 3);
  const lower = clean.toLowerCase();
  const positions = words.map(word => lower.indexOf(word)).filter(index => index >= 0);
  const matchAt = positions.length ? Math.min(...positions) : 0;
  const start = Math.max(0, Math.min(matchAt - 90, clean.length - maxLength));
  return `${start ? '…' : ''}${clean.slice(start, start + maxLength)}…`;
}

function checkedUrl(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : '';
  } catch { return ''; }
}

function renderResults(query, results) {
  resultList.innerHTML = results.map(doc => {
    const url = checkedUrl(doc.url);
    const source = doc.host || (url ? new URL(url).hostname : 'Government document');
    const date = doc.fetched_at ? String(doc.fetched_at).slice(0, 10) : '';
    const labels = [doc.category, doc.state, doc.kind].filter(Boolean).slice(0, 3);
    const statusLabel = doc.status === 'review' ? 'Review' : 'Good';
    return `
      <article class="result-card">
        <div class="card-top"><span class="quality-pill ${doc.status === 'review' ? 'quality-review' : 'quality-good'}">${statusLabel}</span>${labels.map(label => `<span class="topic-pill">${escapeHTML(label)}</span>`).join('')}<span class="card-date">${escapeHTML(date)}</span></div>
        <h3>${escapeHTML(doc.title || 'Untitled government document')}</h3>
        <p class="excerpt">${escapeHTML(excerpt(doc.text, query))}</p>
        <div class="card-bottom"><div class="source"><span class="source-seal">${escapeHTML(source.charAt(0).toUpperCase())}</span><span class="source-text">${escapeHTML(source)}${doc.trust ? ` <span class="source-domain">· ${escapeHTML(doc.trust)}</span>` : ''}</span></div>${url ? `<a class="view-source" href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">Visit source <span>↗</span></a>` : '<span class="view-source">Source URL unavailable</span>'}</div>
      </article>`;
  }).join('');
  resultList.hidden = results.length === 0;
  noResults.hidden = results.length > 0;
  searchError.hidden = true;
  resultCount.innerHTML = `Showing <b>${results.length}</b> semantic matches`;
  loadMore.hidden = results.length < activeLimit || activeLimit >= 50;
}

async function search() {
  const query = queryInput.value.trim();
  if (query.length < 2) {
    resultList.innerHTML = '';
    resultList.hidden = true;
    noResults.hidden = true;
    searchError.hidden = true;
    loadMore.hidden = true;
    resultCount.textContent = 'Enter at least two characters to search';
    return;
  }
  const current = ++requestNumber;
  resultCount.textContent = 'Finding semantically similar documents…';
  resultList.hidden = true;
  noResults.hidden = true;
  searchError.hidden = true;
  loadMore.hidden = true;
  try {
    const response = await fetch(`${API}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        limit: activeLimit,
        category: categoryFilter.value || null,
        state: stateFilter.value || null,
        include_review: reviewFilter.checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Search returned ${response.status}`);
    if (current !== requestNumber) return;
    renderResults(query, payload.results || []);
  } catch (error) {
    if (current !== requestNumber) return;
    document.querySelector('#search-error p').textContent = error.message || 'The search service is not available. Start the local API and try again.';
    searchError.hidden = false;
    resultCount.textContent = 'Search service unavailable';
  }
}

async function loadStatusAndFacets() {
  try {
    const response = await fetch(`${API}/health`);
    if (!response.ok) throw new Error('API unavailable');
    const health = await response.json();
    statusLabel.textContent = !health.index_ready ? 'Build index to start' : !health.model_ready ? 'Model access needed' : `${health.good_rows.toLocaleString()} screened · ${health.review_rows.toLocaleString()} review`;
    if (!health.index_ready || !health.model_ready) statusLabel.classList.add('status-warning');
  } catch {
    statusLabel.textContent = 'Start local API';
    statusLabel.classList.add('status-warning');
  }

  try {
    const response = await fetch(`${API}/facets`);
    if (!response.ok) return;
    const facets = await response.json();
    for (const [select, values, placeholder] of [[categoryFilter, facets.categories, 'All topics'], [stateFilter, facets.states, 'All jurisdictions']]) {
      select.replaceChildren(new Option(placeholder, ''), ...(values || []).map(value => new Option(value, value)));
    }
  } catch { /* The API status above explains why filters are unavailable. */ }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  activeLimit = 12;
  search();
  document.querySelector('.results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

document.querySelectorAll('.quick-searches button').forEach(button => button.addEventListener('click', () => {
  queryInput.value = button.dataset.query;
  activeLimit = 12;
  search();
}));

categoryFilter.addEventListener('change', () => { activeLimit = 12; search(); });
stateFilter.addEventListener('change', () => { activeLimit = 12; search(); });
reviewFilter.addEventListener('change', () => { activeLimit = 12; search(); });
document.querySelector('#clear-filters').addEventListener('click', event => {
  event.stopPropagation();
  categoryFilter.value = '';
  stateFilter.value = '';
  reviewFilter.checked = true;
  queryInput.value = '';
  activeLimit = 12;
  ++requestNumber;
  resultList.innerHTML = '';
  resultList.hidden = true;
  noResults.hidden = true;
  searchError.hidden = true;
  loadMore.hidden = true;
  resultCount.textContent = 'Enter a question above to search the embedded documents';
});
loadMore.addEventListener('click', () => {
  activeLimit = Math.min(activeLimit + 12, 50);
  search();
});
document.querySelector('#retry-search').addEventListener('click', search);
document.querySelector('.filter-head').addEventListener('click', event => {
  if (window.matchMedia('(max-width: 650px)').matches && !event.target.closest('button')) document.querySelector('.filters').classList.toggle('open');
});

resultCount.textContent = 'Enter a question above to search the embedded documents';
loadStatusAndFacets();
