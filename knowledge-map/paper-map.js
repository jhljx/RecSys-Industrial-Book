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
  const hierarchy = data.paperHierarchy || {};
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
  let selectedGroups = new Map();
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
    const practice = document.createElement('p'); practice.textContent = '选型与复现：先固定本章所比较的目标、样本和线上预算，再查看关联概念卡的工程检查与章节代码；论文方法链接仅供延伸阅读。'; detail.append(practice);
    if (paper.url) addLink('论文原文（扩展说明）↗', paper.url);
  }
  function showConcept(concept, group) {
    detail.replaceChildren();
    const label = document.createElement('span'); label.className = 'overline';
    label.textContent = '汇总概念 / 方法归属'; detail.append(label);
    const title = document.createElement('h3'); title.textContent = concept.title; detail.append(title);
    const definition = document.createElement('p'); definition.textContent = concept.definition; detail.append(definition);
    const summary = document.createElement('p'); summary.textContent = concept.summary; detail.append(summary);
    addLink('阅读概念卡 ↗', concept.url);
    const count = document.createElement('p'); count.className = 'citation';
    count.textContent = `本分支包含 ${group.papers.length} 个代表论文方法；点击圆点展开或收起。`;
    detail.append(count);
  }
  async function render() {
    const number = Number(select.value);
    const rows = papers.filter(paper => paper.chapter === number);
    selected = new Map(rows.map(paper => [paper.id, paper]));
    const groups = hierarchy[number] || [];
    selectedGroups = new Map(groups.map(group => [group.concept, group]));
    document.getElementById('paper-count').textContent = `${groups.length} 类汇总概念 · ${rows.length} 个代表论文方法`;
    const branches = groups.map(group => {
      const items = group.papers.map(id => node(selected.get(id).method, [], {paperId: id}));
      for (const id of group.extraConcepts || [])
        items.push(node(concepts.get(id).title, [], {conceptId: id}));
      return node(concepts.get(group.concept).title, items, {conceptId: group.concept});
    });
    const tree = node(`第 ${number} 章 · ${chapters.get(number).title}`, branches);
    try {
      if (mm) { await mm.setData(tree); await mm.fit(); }
      else {
      mm = window.markmap.Markmap.create(svg, {
        color: node => ['#9b3924', '#546d73', '#6b675e'][Math.min((node.state?.depth || 1) - 1, 2)], colorFreezeLevel: 2,
        initialExpandLevel: 2, maxWidth: 240, spacingHorizontal: 100, spacingVertical: 14,
        duration: 240, maxInitialScale: 1.4
      });
      await mm.setData(tree); await mm.fit();
      }
    } catch (error) { document.getElementById('paper-count').textContent = `图谱渲染失败：${error.message}`; console.error(error); }
    detail.replaceChildren();
    const label = document.createElement('span'); label.className = 'overline'; label.textContent = '本章方法路线'; detail.append(label);
    const title = document.createElement('h3'); title.textContent = `${groups.length} 类汇总概念`; detail.append(title);
    const list = document.createElement('ul'); list.className = 'paper-branch-list';
    for (const group of groups) {
      const item = document.createElement('li');
      item.textContent = `${concepts.get(group.concept).title} · ${group.papers.length} 篇`;
      list.append(item);
    }
    detail.append(list);
    const hint = document.createElement('p'); hint.textContent = '点击圆点展开方法，点击名称看 PDF 页码与专属概念卡；连线仅表示归属。'; detail.append(hint);
  }
  svg.addEventListener('click', event => {
    const group = event.target.closest('g.markmap-node');
    const payload = group && window.d3.select(group).datum()?.payload;
    if (payload?.paperId) showPaper(selected.get(payload.paperId));
    else if (payload?.conceptId) showConcept(concepts.get(payload.conceptId), selectedGroups.get(payload.conceptId) || {papers: []});
  });
  select.addEventListener('change', render);
  select.value = String(chapterNumbers.includes(20) ? 20 : chapterNumbers[0]);
  render();
})();
