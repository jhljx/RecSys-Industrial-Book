/* Keep Markdown navigation within the themed Pages routes; preserve local file links. */
(() => {
  if (location.protocol === 'file:') return;
  const base = new URL('../', document.currentScript.src);
  document.querySelectorAll('a[href]').forEach(anchor => {
    const url = new URL(anchor.getAttribute('href'), location.href);
    if (url.origin !== base.origin || !url.pathname.startsWith(base.pathname)) return;
    if (/\/README\.md$/.test(url.pathname)) {
      url.pathname = url.pathname.replace(/README\.md$/, '');
      if (url.pathname === base.pathname) url.pathname += 'knowledge-map/';
    } else if (/\.md$/.test(url.pathname)) {
      url.pathname = url.pathname.replace(/\.md$/, '.html');
    } else return;
    anchor.href = url.href;
  });
})();
