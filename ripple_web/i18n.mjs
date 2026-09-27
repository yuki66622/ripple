// Language is a URL option so links remain shareable and data stays identical.
export const isEnglish = new URLSearchParams(globalThis.location?.search || '').get('lang') === 'en';
export const locale = isEnglish ? 'en-US' : 'zh-CN';
export const tr = (zh, en) => isEnglish ? en : zh;

export function syncLanguageLink() {
  const link = document.getElementById('language-switch');
  if (!link) return;
  const url = new URL(location.href);
  if (isEnglish) url.searchParams.delete('lang');
  else url.searchParams.set('lang', 'en');
  link.href = `${url.pathname}${url.search}${url.hash}`;
  link.textContent = isEnglish ? '中文' : 'English';
  link.lang = isEnglish ? 'zh-CN' : 'en';
  link.hreflang = link.lang;
  link.setAttribute('aria-label', tr('Switch to English', '切换到中文'));
}

export function initializeLanguage() {
  document.documentElement.lang = isEnglish ? 'en' : 'zh-CN';
  if (isEnglish) {
    document.querySelectorAll('[data-en]').forEach(node => { node.textContent = node.dataset.en; });
    for (const attribute of ['aria-label', 'title', 'content']) {
      document.querySelectorAll(`[data-en-${attribute}]`).forEach(node => {
        node.setAttribute(attribute, node.getAttribute(`data-en-${attribute}`));
      });
    }
  }
  syncLanguageLink();
  document.getElementById('language-switch')?.addEventListener('click', syncLanguageLink);
}

if (typeof document !== 'undefined') initializeLanguage();
