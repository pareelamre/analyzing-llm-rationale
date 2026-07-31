type ForeseaPageContext = {
  page?: string;
  path?: string;
  canonical?: string;
  api?: Record<string, string>;
  auth?: Record<string, boolean>;
};

declare global {
  interface Window {
    ForeseaPageContext?: ForeseaPageContext;
  }
}

const contextEl = document.getElementById("foresea-page-context");

if (contextEl?.textContent) {
  try {
    window.ForeseaPageContext = JSON.parse(contextEl.textContent) as ForeseaPageContext;
  } catch {
    window.ForeseaPageContext = {};
  }
} else {
  window.ForeseaPageContext = {};
}
