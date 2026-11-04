/* Separate editorial taxonomy from the prerequisite dependency graph. Markmap handles layout and pan/zoom. */
(() => {
  'use strict';
  const data = window.KNOWLEDGE_MAP;
  const select = document.getElementById('paper-chapter');
  const svg = document.getElementById('paper-map-view');
  const detail = document.getElementById('paper-detail');
  if (!data || !select || !svg || !detail) return;
  const chapters = new Map(data.chapters.map(row => [row.number, row]));
  const concepts = new Map(data.concepts.map(row => [row.id, row]));
  const papers = data.papers || [];
  const chapterNumbers = [...new Set(papers.map(row => row.chapter))].sort((a, b) => a - b);
  const escapeHTML = text => String(text).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const siteHref = href => location.protocol === 'file:' ? href : href.replace(/\.md(?=#|$)/, '.html');
  const addLink = (label, href) => { const a = document.createElement('a'); a.textContent = label; a.href = siteHref(href); detail.append(a); };
  chapterNumbers.forEach(number => {
    const option = document.createElement('option');
    option.value = String(number); option.textContent = `第 ${number} 章 · ${chapters.get(number).title}`;
    select.append(option);
  });
  if (!chapterNumbers.length) {
    document.getElementById('paper-count').textContent = '暂无经人工策划的代表论文方法。';
    return;
  }
  if (!window.markmap || !window.markmap.Markmap) {
    document.getElementById('paper-count').textContent = '图谱组件未加载；仍可通过章节目录阅读 PDF。';
    return;
  }
  let mm;
  let selected = new Map();
  const node = (content, children = [], payload = {}) => ({content: escapeHTML(content), children, payload});
  function showPaper(paper) {
    detail.replaceChildren();
    const label = document.createElement('span'); label.className = 'overline'; label.textContent = '代表论文方法 / 本书依据'; detail.append(label);
    const title = document.createElement('h3'); title.textContent = paper.method; detail.append(title);
    const full = document.createElement('p'); full.textContent = paper.title; detail.append(full);
    const citation = document.createElement('p'); citation.className = 'citation'; citation.textContent = `《${chapters.get(paper.chapter).title}》，PDF 第 ${paper.pages[0]}–${paper.pages[1]} 页。`; detail.append(citation);
    addLink('站内阅读对应 PDF 页 ↗', `./reader.html#ch${paper.chapter}&page=${paper.pages[0]}`);
    const related = document.createElement('p'); related.textContent = '关联概念：' + paper.concepts.map(id => concepts.get(id).title).join('、'); detail.append(related);
    paper.concepts.forEach(id => addLink(`阅读概念卡 · ${concepts.get(id).title} ↗`, concepts.get(id).url));
    if (paper.url) addLink('论文原文（扩展说明）↗', paper.url);
  }
  async function render() {
    const number = Number(select.value);
    const rows = papers.filter(paper => paper.chapter === number);
    selected = new Map(rows.map(paper => [paper.id, paper]));
    document.getElementById('paper-count').textContent = `${rows.length} 个代表论文方法 · ${new Set(rows.flatMap(row => row.concepts)).size} 个关联概念`;
    const byConcept = new Map();
    rows.forEach(paper => paper.concepts.forEach(id => {
      if (!byConcept.has(id)) byConcept.set(id, []);
      byConcept.get(id).push(paper);
    }));
    const branches = [...byConcept].map(([id, items]) => node(concepts.get(id).title, items.map(paper => node(paper.method, [], {paperId: paper.id}))));
    const tree = node(`第 ${number} 章 · ${chapters.get(number).title}`, branches);
    try {
      if (mm) { await mm.setData(tree); await mm.fit(); }
      else {
      mm = window.markmap.Markmap.create(svg, {
        color: node => ['#9b3924', '#546d73', '#6b675e'][Math.min((node.state?.depth || 1) - 1, 2)], colorFreezeLevel: 2,
        initialExpandLevel: 3, maxWidth: 220, spacingHorizontal: 90, spacingVertical: 10,
        duration: 240, maxInitialScale: 1.4
      });
      await mm.setData(tree); await mm.fit();
      }
    } catch (error) { document.getElementById('paper-count').textContent = `图谱渲染失败：${error.message}`; console.error(error); }
    detail.replaceChildren();
    const hint = document.createElement('p'); hint.textContent = '展开概念分支，点击论文方法查看书中页码和阅读入口。连线只表示主题归属，不表示模型前置或演进关系。'; detail.append(hint);
  }
  svg.addEventListener('click', event => {
    const group = event.target.closest('g.markmap-node');
    const paper = group && selected.get(window.d3.select(group).datum()?.payload?.paperId);
    if (paper) showPaper(paper);
  });
  select.addEventListener('change', render);
  select.value = String(chapterNumbers.includes(12) ? 12 : chapterNumbers[0]);
  render();
})();
