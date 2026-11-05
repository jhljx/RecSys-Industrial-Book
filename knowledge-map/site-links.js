/* Resolve Jekyll Markdown routes without breaking direct file:// browsing. */
(() => {
  'use strict';
  const root = new URL('../', document.currentScript.src);
  const local = location.protocol === 'file:';
  document.querySelectorAll('a[href]').forEach(anchor => {
    const href = anchor.getAttribute('href');
    if (!href || href.startsWith('#')) return;
    const target = new URL(href, location.href);
    if (target.origin !== root.origin || !target.pathname.startsWith(root.pathname)) return;
    const path = decodeURIComponent(target.pathname.slice(root.pathname.length));
    if (local) {
      if (target.pathname.endsWith('/')) {
        target.pathname += path === '' ? 'knowledge-map/index.html' : path === 'knowledge-map/' ? 'index.html' : 'README.md';
      } else if (/^(?:README\.en|(?:knowledge|learning-paths|docs|templates)\/.*)\.html$/.test(path)) {
        target.pathname = target.pathname.replace(/\.html$/, '.md');
      }
    } else if (/\/README\.md$/.test(path)) {
      target.pathname = target.pathname.replace(/README\.md$/, '');
      if (target.pathname === root.pathname) target.pathname += 'knowledge-map/';
    } else if (/\.md$/.test(path)) {
      target.pathname = target.pathname.replace(/\.md$/, '.html');
    } else return;
    anchor.href = target.href;
  });
})();
