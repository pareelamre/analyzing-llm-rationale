/**
 * Foresea Drop-In Prediction Market & Forecasting Widget
 * (C) Foresea 2026 - https://foresea.ink
 *
 * Usage:
 *   <script src="https://foresea.ink/widget.js" async></script>
 *
 * Example 1 (By Question):
 *   <foresea-card data-question="Will SpaceX land Starship on Mars by 2028?" data-theme="dark"></foresea-card>
 *
 * Example 2 (By Share ID):
 *   <foresea-card data-share-id="abc123xyz"></foresea-card>
 */

(function () {
  'use strict';

  const FORESEA_API_BASE = (function () {
    const scripts = document.getElementsByTagName('script');
    for (let i = scripts.length - 1; i >= 0; i--) {
      const src = scripts[i].src || '';
      if (src.includes('widget.js')) {
        try {
          const url = new URL(src);
          return url.origin;
        } catch (e) {
          break;
        }
      }
    }
    return 'https://foresea.ink';
  })();

  const STYLES = `
    .foresea-widget-card {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      border-radius: 12px;
      padding: 16px 20px;
      margin: 12px 0;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: #121824;
      color: #e2e8f0;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
      box-sizing: border-box;
      max-width: 540px;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .foresea-widget-card:hover {
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    }
    .foresea-widget-light {
      background: #ffffff;
      color: #1e293b;
      border-color: #e2e8f0;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }
    .foresea-widget-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .foresea-widget-brand {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #00d2ff;
      text-decoration: none;
    }
    .foresea-widget-badge {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 20px;
      background: rgba(0, 210, 255, 0.12);
      color: #00d2ff;
    }
    .foresea-widget-question {
      font-size: 15px;
      font-weight: 600;
      line-height: 1.4;
      margin: 0 0 14px 0;
      color: inherit;
    }
    .foresea-widget-bar-container {
      margin-bottom: 12px;
    }
    .foresea-widget-bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 6px;
    }
    .foresea-widget-bar-bg {
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 4px;
      overflow: hidden;
      display: flex;
    }
    .foresea-widget-light .foresea-widget-bar-bg {
      background: #f1f5f9;
    }
    .foresea-widget-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #00d2ff, #00f5a0);
      border-radius: 4px;
      transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .foresea-widget-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      color: #94a3b8;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      padding-top: 10px;
    }
    .foresea-widget-light .foresea-widget-footer {
      border-top-color: #f1f5f9;
      color: #64748b;
    }
    .foresea-widget-footer a {
      color: inherit;
      text-decoration: none;
      font-weight: 500;
    }
    .foresea-widget-footer a:hover {
      color: #00d2ff;
      text-decoration: underline;
    }
    .foresea-widget-loading {
      font-size: 13px;
      color: #94a3b8;
      padding: 12px 0;
      text-align: center;
    }
  `;

  function injectStyles() {
    if (document.getElementById('foresea-widget-styles')) return;
    const style = document.createElement('style');
    style.id = 'foresea-widget-styles';
    style.textContent = STYLES;
    document.head.appendChild(style);
  }

  function renderWidget(element) {
    if (element.getAttribute('data-foresea-rendered')) return;
    element.setAttribute('data-foresea-rendered', 'true');

    const question = element.getAttribute('data-question') || element.getAttribute('data-q') || '';
    const shareId = element.getAttribute('data-share-id') || '';
    const theme = element.getAttribute('data-theme') || 'dark';

    const card = document.createElement('div');
    card.className = 'foresea-widget-card' + (theme === 'light' ? ' foresea-widget-light' : '');
    card.innerHTML = '<div class="foresea-widget-loading">Loading Foresea forecast...</div>';
    element.appendChild(card);

    if (shareId) {
      fetch(`${FORESEA_API_BASE}/forecast/${shareId}`)
        .then(r => r.json())
        .then(data => populateCard(card, data.question || question, data.probability, data.predicted_answer, shareId))
        .catch(() => {
          card.innerHTML = '<div class="foresea-widget-loading">Unable to load forecast.</div>';
        });
    } else if (question) {
      fetch(`${FORESEA_API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question }),
      })
        .then(r => r.json())
        .then(data => populateCard(card, question, data.predicted_probability, data.predicted_answer, ''))
        .catch(() => {
          card.innerHTML = '<div class="foresea-widget-loading">Unable to generate forecast.</div>';
        });
    } else {
      card.innerHTML = '<div class="foresea-widget-loading">No question provided.</div>';
    }
  }

  function populateCard(card, question, prob, ans, shareId) {
    const p = typeof prob === 'number' ? prob : 0.5;
    const pct = Math.round(p * 100);
    const ansText = ans ? String(ans).toUpperCase() : (p >= 0.5 ? 'YES' : 'NO');
    const link = shareId ? `${FORESEA_API_BASE}/forecast/${shareId}` : `${FORESEA_API_BASE}/?q=${encodeURIComponent(question)}`;

    card.innerHTML = `
      <div class="foresea-widget-header">
        <a class="foresea-widget-brand" href="${FORESEA_API_BASE}" target="_blank" rel="noopener">
          <span>🌊 FORESEA</span>
        </a>
        <span class="foresea-widget-badge">Calibrated AI</span>
      </div>
      <div class="foresea-widget-question">${escapeHtml(question)}</div>
      <div class="foresea-widget-bar-container">
        <div class="foresea-widget-bar-label">
          <span>Probability: ${pct}%</span>
          <span style="color: ${pct >= 50 ? '#00f5a0' : '#ff5577'}">${ansText}</span>
        </div>
        <div class="foresea-widget-bar-bg">
          <div class="foresea-widget-bar-fill" style="width: ${pct}%"></div>
        </div>
      </div>
      <div class="foresea-widget-footer">
        <span>Verified Calibration Track Record</span>
        <a href="${link}" target="_blank" rel="noopener">View Full Thesis →</a>
      </div>
    `;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
  }

  function init() {
    injectStyles();
    const elements = document.querySelectorAll('foresea-card, .foresea-widget, div[data-foresea-widget]');
    elements.forEach(renderWidget);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Export global initializer
  window.ForeseaWidget = { init: init };
})();
