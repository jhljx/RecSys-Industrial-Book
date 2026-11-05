/* Shared chapter-code catalog and source-page behavior. No untrusted HTML insertion. */
(() => {
  'use strict';
  const code = document.getElementById('source-code');
  if (code) {
    if (location.protocol !== 'file:') {
      document.querySelectorAll('a[href$="/README.md"]').forEach(a => {
        a.href = a.getAttribute('href').replace(/README\.md$/, '');
      });
    }
    const button = document.getElementById('copy-code');
    button?.addEventListener('click', async () => {
      const source = [...code.querySelectorAll('.source-line > span')].map(line => line.textContent).join('\n');
      try { await navigator.clipboard.writeText(source); button.textContent = '已复制'; }
      catch { button.textContent = '复制不可用，请选中源码'; }
    });
    return;
  }
  const records = window.CHAPTER_CODE;
  const map = window.KNOWLEDGE_MAP;
  const results = document.getElementById('code-results');
  const input = document.getElementById('code-search');
  const select = document.getElementById('code-chapter');
  if (!results || !input || !select) return;
  if (!Array.isArray(records) || !map) { results.textContent = '代码目录数据缺失，请重新生成公开站点数据。'; return; }
  const chapters = new Map(map.chapters.map(c => [c.number, c]));
  const params = new URLSearchParams(location.search);
  const initialChapter = /^#ch(\d+)$/.exec(location.hash)?.[1] || params.get('chapter') || '';
  const initialQuery = params.get('q') || '';
  for (const n of [...new Set(records.map(row => row.chapter))].sort((a, b) => a - b)) {
    const option = document.createElement('option'); option.value = String(n);
    option.textContent = `第 ${n} 章 · ${chapters.get(n)?.title || ''}`; select.append(option);
  }
  if ([...select.options].some(o => o.value === initialChapter)) select.value = initialChapter;
  input.value = initialQuery;
  const el = (tag, className, text) => {
    const node = document.createElement(tag); if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text); return node;
  };
  const validationLabel = {ci_smoke: 'CI 主路径冒烟', syntax_only: 'Python 语法检查', source_only: '仅源码展示'};
  const show = () => {
    const query = input.value.trim().toLocaleLowerCase(); const chapter = select.value;
    const matches = records.filter(row => (!chapter || String(row.chapter) === chapter)
      && (!query || `${row.path} ${row.description}`.toLocaleLowerCase().includes(query)));
    results.replaceChildren(); document.getElementById('code-count').textContent = `${matches.length} 个代码示例 · ${chapter ? `第 ${chapter} 章` : '全部章节'}`;
    if (!matches.length) { results.append(el('p', 'empty', '没有匹配的代码示例。请换用方法名、文件名或章节目录继续查找。')); return; }
    const groups = new Map();
    for (const row of matches) { if (!groups.has(row.chapter)) groups.set(row.chapter, []); groups.get(row.chapter).push(row); }
    for (const [number, rows] of groups) {
      const section = el('section', 'code-group'); section.id = `ch${number}`;
      const heading = el('div', 'section-heading'); const title = el('div');
      title.append(el('span', 'overline', `CH ${String(number).padStart(2, '0')} / ${rows.length} EXAMPLES`), el('h2', '', chapters.get(number)?.title || `第 ${number} 章`));
      heading.append(title); section.append(heading);
      const list = el('div', 'code-items');
      for (const row of rows) {
        const a = el('a', 'code-item'); a.href = row.url;
        a.append(el('small', '', `${row.path.split('/').slice(-2, -1)[0] || `ch${number}`} · ${validationLabel[row.validation] || '未验证'}`),
          el('strong', '', row.path.split('/').at(-1)),
          el('span', '', row.description || '暂无逐文件说明；打开源码查看结构和完整内容。'));
        list.append(a);
      }
      section.append(list); results.append(section);
    }
  };
  input.addEventListener('input', show); select.addEventListener('change', show); show();
})();
