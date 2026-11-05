/* Dependency-free PDF reader. Published chapter metadata comes from data.js. */
(() => {
  'use strict';
  const data = window.KNOWLEDGE_MAP;
  const $ = id => document.getElementById(id);
  const rows = Array.isArray(data?.chapters) ? [...data.chapters].sort((a, b) => a.number - b.number) : [];
  if (!rows.length) {
    $('chapter-title').textContent = '章节数据不可用';
    $('reader-status').textContent = '请检查 data.js，或运行 python3 tools/build_knowledge_map.py。';
    $('chapter-count').textContent = '暂无已发布章节';
    $('page-form').hidden = true;
    return;
  }
  const byNumber = new Map(rows.map(row => [row.number, row]));
  const chapterHash = (number, page = 1) => `#ch${number}&page=${page}`;
  const siteHref = href => {
    if (href === '../') return './index.html';
    if (location.protocol === 'file:') return href;
    if (/\/README\.md(?:#.*)?$/.test(href)) return href.replace(/\/README\.md(?=#|$)/, '/');
    return href.replace(/\.md(?=#|$)/, '.html');
  };
  document.querySelectorAll('a[href="../"]').forEach(anchor => { anchor.href = siteHref('../'); });
  // A PDF fragment is interpreted by the browser's built-in PDF viewer, not fetched via GitHub raw.
  const pdfHref = (chapter, page) => `${chapter.pdf}#page=${page}&view=FitH`;
  let activeNumber;
  let activePage;

  for (const chapter of rows) {
    const item = document.createElement('li');
    const anchor = document.createElement('a');
    anchor.href = chapterHash(chapter.number);
    anchor.textContent = `第 ${chapter.number} 章 · ${chapter.title}`;
    anchor.dataset.chapter = String(chapter.number);
    item.append(anchor);
    $('chapter-list').append(item);
  }
  $('chapter-count').textContent = `共 ${rows.length} 章 · 按已发布章节排列`;

  function readLocation() {
    // Unknown or unpublished chapter hashes fall back to the first released chapter.
    const match = /^#ch(\d+)(?:&page=(\d+))?$/.exec(location.hash);
    const number = match ? Number(match[1]) : activeNumber ?? rows[0].number;
    const page = match?.[2] ? Number(match[2]) : 1;
    return { chapter: byNumber.get(number) || rows[0], page: Number.isSafeInteger(page) && page > 0 ? page : 1 };
  }

  function setAdjacent(id, chapter) {
    const anchor = $(id);
    if (chapter) {
      anchor.href = chapterHash(chapter.number);
      anchor.removeAttribute('aria-disabled');
    } else {
      anchor.removeAttribute('href');
      anchor.setAttribute('aria-disabled', 'true');
    }
  }

  function render() {
    const { chapter, page } = readLocation();
    const index = rows.indexOf(chapter);
    document.title = `第 ${chapter.number} 章 · ${chapter.title} | PDF 阅读器`;
    $('chapter-title').textContent = `第 ${chapter.number} 章 · ${chapter.title}`;
    $('reader-status').textContent = `当前：第 ${page} 页（共 ${rows.length} 个已发布章节）`;
    $('page-number').value = String(page);
    $('pdf-frame').title = `第 ${chapter.number} 章 ${chapter.title}，PDF 第 ${page} 页`;
    $('direct-pdf').href = pdfHref(chapter, page);
    $('chapter-map').href = siteHref(chapter.map);
    setAdjacent('previous', rows[index - 1]);
    setAdjacent('next', rows[index + 1]);
    $('chapter-list').querySelectorAll('a').forEach(anchor => {
      const current = Number(anchor.dataset.chapter) === chapter.number;
      if (current) anchor.setAttribute('aria-current', 'page');
      else anchor.removeAttribute('aria-current');
    });
    // Avoid reloading a chapter when a user merely revisits its current hash.
    if (activeNumber !== chapter.number || activePage !== page) {
      $('pdf-frame').src = pdfHref(chapter, page);
      activeNumber = chapter.number;
      activePage = page;
    }
  }

  document.querySelector('.skip').addEventListener('click', event => {
    event.preventDefault();
    $('reader').scrollIntoView();
    $('reader').focus({ preventScroll: true });
  });
  $('page-form').addEventListener('submit', event => {
    event.preventDefault();
    const input = $('page-number');
    const page = Number(input.value);
    if (!Number.isSafeInteger(page) || page < 1) {
      input.setCustomValidity('请输入大于 0 的整数页码');
      input.reportValidity();
      return;
    }
    input.setCustomValidity('');
    const target = chapterHash(readLocation().chapter.number, page);
    if (location.hash === target) render();
    else location.hash = target;
  });
  $('page-number').addEventListener('input', () => $('page-number').setCustomValidity(''));
  window.addEventListener('hashchange', render);
  render();
})();
