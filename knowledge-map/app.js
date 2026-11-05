/* Static, dependency-free reader UI. Data is generated from released repository metadata. */
(() => {
  'use strict';
  const data = window.KNOWLEDGE_MAP;
  const $ = id => document.getElementById(id);
  if (!data) { $('items').textContent = '知识地图数据缺失，请重新运行 python3 tools/build_knowledge_map.py。'; return; }
  const chapters = new Map(data.chapters.map(row => [row.number, row]));
  const concepts = new Map(data.concepts.map(row => [row.id, row]));
  const codeByChapter = new Map();
  for (const row of window.CHAPTER_CODE || []) {
    if (!codeByChapter.has(row.chapter)) codeByChapter.set(row.chapter, []);
    codeByChapter.get(row.chapter).push(row);
  }
  const codeForConcept = concept => (codeByChapter.get(concept.chapter) || [])
    .filter(row => row.concept_ids?.includes(concept.id)).slice(0, 8);
  const groups = new Map(data.groups.map(row => [row.id, row]));
  const state = { kind: 'chapters', group: 'all', query: '', selected: null, course: data.paths[0].id, graphChapter: chapters.has(4) ? 4 : data.chapters[0].number };
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const siteHref = href => {
    if (href === '../') return './index.html';
    if (location.protocol === 'file:') return href;
    if (/\/README\.md(?:#.*)?$/.test(href)) return href.replace(/\/README\.md(?=#|$)/, '/');
    return href.replace(/\.md(?=#|$)/, '.html');
  };
  document.querySelectorAll('a[href]').forEach(a => { a.href = siteHref(a.getAttribute('href')); });
  const link = (label, href) => { const a = el('a', '', label); a.href = siteHref(href); return a; };
  const addAgentCommand = (parent, topic) => {
    const quote = value => `'${value.replaceAll("'", "'\"'\"'")}'`;
    const command = `python3 tools/recsys_agent.py ask ${quote(`书中如何解释${topic}？`)} --session reading`;
    const block = el('div', 'agent-command');
    add(block, el('h4', '', '继续向本地 Agent 提问'), el('p', '', '在仓库根目录运行以下命令；问答使用本机 PDF 索引。'));
    const code = el('code', '', command); add(block, code);
    const copy = el('button', 'inline-link', '复制命令'); copy.type = 'button';
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(command); copy.textContent = '已复制'; }
      catch { copy.textContent = '请手动选中命令复制'; }
    });
    add(block, copy); add(parent, block);
  };
  const add = (parent, ...children) => { for (const child of children) parent.append(child); return parent; };
  const groupOf = row => chapters.get(row.chapter)?.group || 'frontiers';
  const chapterLabel = n => chapters.get(n) ? `第 ${n} 章 · ${chapters.get(n).title}` : `第 ${n} 章 · 尚未发布`;
  const appendHeading = (parent, label, title) => add(parent, el('span', 'overline', label), el('h3', '', title));
  const stats = [['章节地图', data.chapters.length], ['核心概念', data.concepts.length], ['前置关系', data.edges.length], ['课程路线', data.paths.length]];
  for (const [label, value] of stats) add($('stats'), add(el('div', 'stat'), el('strong', '', value), el('span', '', label)));
  data.groups.forEach((g, index) => {
    const button = el('button', 'group-card'); button.type = 'button';
    const tags = el('div', 'chapter-tags'); g.chapters.forEach(n => add(tags, el('span', '', `CH ${String(n).padStart(2, '0')}`)));
    add(button, el('span', 'index', String(index + 1).padStart(2, '0') + ' / ' + String(data.groups.length).padStart(2, '0')), el('strong', '', g.title), el('small', '', g.description), tags);
    button.addEventListener('click', () => { state.group = g.id; state.kind = 'chapters'; state.query = ''; $('search').value = ''; render(); $('catalog').scrollIntoView({ behavior: 'smooth' }); });
    add($('blueprint-groups'), button);
  });
  for (const [id, title] of [['all', '全部板块'], ...data.groups.map(g => [g.id, g.title])]) {
    const button = el('button', 'filter', title); button.type = 'button'; button.dataset.group = id;
    button.addEventListener('click', () => { state.group = id; state.selected = null; render(); }); add($('filters'), button);
  }
  document.querySelectorAll('.tab[data-kind]').forEach(button => button.addEventListener('click', () => { state.kind = button.dataset.kind; state.selected = null; render(); }));
  $('search').addEventListener('input', e => { state.query = e.target.value.trim().toLocaleLowerCase(); state.selected = null; render(); });
  // Keep the graph chapter-scoped: only one-hop edges touching a concept in this chapter.
  const graphChapter = $('graph-chapter');
  data.chapters.forEach(chapter => {
    const option = el('option', '', chapterLabel(chapter.number));
    option.value = chapter.number;
    add(graphChapter, option);
  });
  graphChapter.value = String(state.graphChapter);
  graphChapter.addEventListener('change', () => { state.graphChapter = Number(graphChapter.value); renderGraph(); });
  const graphView = $('graph-view');
  let graphEdges = [];
  let graphNodes = new Map();
  let graphSvg;
  const graphSort = (a, b) => a.chapter - b.chapter || a.title.localeCompare(b.title, 'zh-CN');
  function graphButton(concept, external, direction) {
    const button = el('button', `graph-node${external ? ' external' : ' local'}`);
    button.type = 'button';
    button.dataset.concept = concept.id;
    button.setAttribute('aria-label', `${concept.title}，${chapterLabel(concept.chapter)}${external ? `，${direction}` : '，本章概念'}`);
    add(button, el('span', 'graph-node-chapter', chapterLabel(concept.chapter)), el('strong', '', concept.title));
    button.addEventListener('click', () => {
      state.kind = 'concepts'; state.group = 'all'; state.query = ''; $('search').value = '';
      state.selected = `k:${concept.id}`; render();
      $('catalog').scrollIntoView({ behavior: 'smooth' });
      $('detail').focus({ preventScroll: true });
    });
    return button;
  }
  function drawGraphEdges() {
    if (!graphSvg) return;
    graphSvg.replaceChildren();
    const bounds = graphView.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    graphSvg.setAttribute('viewBox', `0 0 ${bounds.width} ${bounds.height}`);
    const ns = 'http://www.w3.org/2000/svg';
    const marker = document.createElementNS(ns, 'marker');
    marker.setAttribute('id', 'graph-arrow'); marker.setAttribute('markerWidth', '10'); marker.setAttribute('markerHeight', '10');
    marker.setAttribute('markerUnits', 'userSpaceOnUse'); marker.setAttribute('viewBox', '0 0 10 10');
    marker.setAttribute('refX', '9'); marker.setAttribute('refY', '5'); marker.setAttribute('orient', 'auto');
    const tip = document.createElementNS(ns, 'path'); tip.setAttribute('d', 'M 0 0 L 10 5 L 0 10 Z'); tip.setAttribute('fill', '#a34934');
    marker.append(tip); const defs = document.createElementNS(ns, 'defs'); defs.append(marker); graphSvg.append(defs);
    const compact = window.matchMedia('(max-width: 700px)').matches;
    graphEdges.forEach(([from, to]) => {
      const source = (graphNodes.get(`${from}:本章`) || graphNodes.get(`${from}:前置`))?.getBoundingClientRect();
      const target = (graphNodes.get(`${to}:本章`) || graphNodes.get(`${to}:后续`))?.getBoundingClientRect();
      if (!source || !target) return;
      const sameLane = !compact && source.left === target.left;
      const x1 = compact ? source.left + source.width / 2 - bounds.left : source.right - bounds.left;
      const y1 = compact ? source.bottom - bounds.top : source.top + source.height / 2 - bounds.top;
      const x2 = compact ? target.left + target.width / 2 - bounds.left : (sameLane ? target.right : target.left) - bounds.left;
      const y2 = compact ? target.top - bounds.top : target.top + target.height / 2 - bounds.top;
      const path = document.createElementNS(ns, 'path');
      const bend = compact ? Math.max(12, Math.abs(y2 - y1) / 3) : Math.max(18, Math.abs(x2 - x1) / 3);
      path.setAttribute('d', compact ? `M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}` : sameLane ? `M ${x1} ${y1} C ${x1 + 28} ${y1}, ${x2 + 28} ${y2}, ${x2} ${y2}` : `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`);
      path.setAttribute('marker-end', 'url(#graph-arrow)');
      graphSvg.append(path);
    });
  }
  function renderGraph() {
    const number = state.graphChapter;
    graphChapter.value = String(number);
    const local = data.concepts.filter(c => c.chapter === number).sort(graphSort);
    const ids = new Set(local.map(c => c.id));
    graphEdges = data.edges.filter(([from, to]) => ids.has(from) || ids.has(to)).filter(([from, to]) => concepts.has(from) && concepts.has(to));
    const before = [...new Set(graphEdges.filter(([, to]) => ids.has(to)).map(([from]) => from))].filter(id => !ids.has(id)).map(id => concepts.get(id)).sort(graphSort);
    const after = [...new Set(graphEdges.filter(([from]) => ids.has(from)).map(([, to]) => to))].filter(id => !ids.has(id)).map(id => concepts.get(id)).sort(graphSort);
    graphView.replaceChildren(); graphNodes = new Map(); graphSvg = null;
    const relations = $('graph-relations'); relations.replaceChildren();
    $('graph-count').textContent = `${local.length} 个本章概念 · ${before.length + after.length} 个跨章位置 · ${graphEdges.length} 条前置关系`;
    if (!local.length) {
      add(graphView, el('p', 'graph-empty', '本章暂无独立概念卡。请通过章节目录阅读 PDF 与章节地图。'));
      return;
    }
    const columns = [
      ['跨章前置', before, '前置'], ['本章概念', local, '本章'], ['跨章后续', after, '后续']
    ];
    columns.forEach(([title, rows, direction], index) => {
      const column = el('div', `graph-column graph-column-${index}`);
      add(column, el('h3', '', `${title} · ${rows.length}`));
      if (!rows.length) add(column, el('p', 'graph-empty', '暂无直接关系'));
      rows.forEach(concept => {
        const node = graphButton(concept, index !== 1, direction);
        // One concept can be on both sides; the edge uses its position for the relevant direction.
        graphNodes.set(`${concept.id}:${direction}`, node);
        add(column, node);
      });
      add(graphView, column);
    });
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'graph-lines'); svg.setAttribute('aria-hidden', 'true');
    graphView.prepend(svg); graphSvg = svg;
    // Each relation remains available as text when lines cannot be seen (including narrow screens).
    if (graphEdges.length) {
      add(relations, el('h3', '', '有向关系 · 前置 → 后续'));
      const list = el('ul');
      graphEdges.forEach(([from, to]) => add(list, el('li', '', `${concepts.get(from).title}（${chapterLabel(concepts.get(from).chapter)}） → ${concepts.get(to).title}（${chapterLabel(concepts.get(to).chapter)}）`)));
      add(relations, list);
    }
    syncGraphSelection();
    requestAnimationFrame(drawGraphEdges);
  }
  function syncGraphSelection() {
    graphView.querySelectorAll('.graph-node').forEach(button => {
      const active = state.selected === `k:${button.dataset.concept}`;
      button.classList.toggle('selected', active);
      button.setAttribute('aria-pressed', String(active));
    });
  }
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(drawGraphEdges).observe(graphView);
  else window.addEventListener('resize', drawGraphEdges);
  renderGraph();
  function render() {
    document.querySelectorAll('.tab[data-kind]').forEach(b => { const active = b.dataset.kind === state.kind; b.classList.toggle('active', active); b.setAttribute('aria-pressed', active); });
    document.querySelectorAll('.filter').forEach(b => { const active = b.dataset.group === state.group; b.classList.toggle('active', active); b.setAttribute('aria-pressed', active); });
    const rows = (state.kind === 'chapters' ? data.chapters : data.concepts).filter(row => {
      const group = state.kind === 'chapters' ? row.group : groupOf(row);
      const haystack = state.kind === 'chapters' ? `${row.number} ${row.title} ${row.sections.map(s => s.section + s.question).join(' ')}` : `${row.id} ${row.title} ${row.summary} ${row.source}`;
      return (state.group === 'all' || group === state.group) && haystack.toLocaleLowerCase().includes(state.query);
    });
    $('result-count').textContent = `${rows.length} 个${state.kind === 'chapters' ? '章节' : '概念'} · ${state.group === 'all' ? '全书' : groups.get(state.group).title}`;
    $('items').replaceChildren();
    if (!rows.length) add($('items'), el('p', 'empty', '未找到匹配内容。可以试试章节名、模型名或切换知识板块。'));
    rows.forEach(row => {
      const key = state.kind === 'chapters' ? `c:${row.number}` : `k:${row.id}`;
      const button = el('button', 'item' + (state.selected === key ? ' selected' : '')); button.type = 'button';
      add(button, el('span', 'meta', state.kind === 'chapters' ? `CH ${String(row.number).padStart(2, '0')} / ${groups.get(row.group).title}` : `${groups.get(groupOf(row)).title} / ${chapterLabel(row.chapter)}`), el('strong', '', row.title), el('small', '', state.kind === 'chapters' ? `${row.sections.length} 个阅读检查点 · ${data.concepts.filter(c => c.chapter === row.number).length} 个概念` : row.id));
      button.addEventListener('click', () => { state.selected = key; render(); if (innerWidth < 700) $('detail').scrollIntoView({behavior:'smooth'}); });
      add($('items'), button);
    });
    renderDetail();
    syncGraphSelection();
  }
  function renderDetail() {
    const panel = $('detail'); panel.replaceChildren();
    if (!state.selected) { add(panel, add(el('div', 'detail-empty'), el('span', 'overline', 'DETAIL'), el('h3', '', '选择一张卡片'), el('p', '', '查看它的位置、知识依赖，以及对应的阅读入口。'))); return; }
    if (state.selected.startsWith('c:')) {
      const c = chapters.get(Number(state.selected.slice(2)));
      if (!c) return;
      appendHeading(panel, `CHAPTER ${String(c.number).padStart(2, '0')} / ${groups.get(c.group).title}`, c.title);
      const actions = el('div', 'detail-links'); add(actions, link('站内阅读 PDF ↗', `./reader.html#ch${c.number}`), link('直接打开 PDF ↗', c.pdf), link('章节阅读地图 ↗', c.map));
      addAgentCommand(panel, `第${c.number}章${c.title}`);
      if (codeByChapter.has(c.number)) add(actions, link(`本章代码示例（${codeByChapter.get(c.number).length}）↗`, `./code.html#ch${c.number}`));
      const graphAction = el('button', 'graph-jump', '查看本章关系图 →'); graphAction.type = 'button';
      graphAction.addEventListener('click', () => { state.graphChapter = c.number; renderGraph(); $('graph').scrollIntoView({ behavior: 'smooth' }); graphChapter.focus({ preventScroll: true }); });
      add(actions, graphAction); add(panel, actions);
      const related = data.concepts.filter(x => x.chapter === c.number);
      add(panel, el('h4', '', '相关概念'));
      if (!related.length) add(panel, el('p', '', '本章暂无独立概念卡，可从阅读地图进入。'));
      else { const list = el('ul'); related.forEach(x => { const item = el('li'); const button = el('button', 'inline-link', x.title); button.type = 'button'; button.addEventListener('click', () => { state.kind = 'concepts'; state.group = 'all'; state.query = ''; $('search').value = ''; state.selected = `k:${x.id}`; render(); }); add(list, add(item, button)); }); add(panel, list); }
      add(panel, el('h4', '', '章节阅读检查点'));
      const sections = el('ul', 'section-list'); c.sections.forEach(s => add(sections, add(el('li'), el('b', '', `${s.section} · PDF 第 ${s.pages} 页`), el('span', '', s.question)))); add(panel, sections);
    } else {
      const c = concepts.get(state.selected.slice(2)); if (!c) return;
      appendHeading(panel, `CONCEPT / ${chapterLabel(c.chapter)}`, c.title);
      add(panel, el('h4', '', '定义'), el('p', 'concept-definition', c.definition || '暂无定义'), el('h4', '', '核心判断'), el('p', '', c.summary));
      const actions = el('div', 'detail-links'); add(actions, link('阅读概念卡 ↗', c.url));
      addAgentCommand(panel, c.title);
      const citationChapter = data.chapters.filter(chapter => c.source.includes(decodeURIComponent(chapter.pdf).split('/').at(-1)))
        .sort((a, b) => c.source.indexOf(decodeURIComponent(a.pdf).split('/').at(-1)) - c.source.indexOf(decodeURIComponent(b.pdf).split('/').at(-1)))[0];
      if (citationChapter) add(actions, link('站内阅读所引 PDF ↗', `./reader.html#ch${citationChapter.number}`)); add(panel, actions);
      add(panel, el('h4', '', '相关代码讲解'));
      const examples = codeForConcept(c);
      if (examples.length) {
        const list = el('ul'); examples.forEach(row => add(list, add(el('li'), link(`${row.path.split('/').at(-1)} ↗`, row.url)))); add(panel, list);
      } else add(panel, el('p', '', '暂无确认的逐概念代码关联；可从本章代码目录按方法或文件名查找。'));
      if (codeByChapter.has(c.chapter)) add(panel, link('浏览本章所有代码 →', `./code.html#ch${c.chapter}`));
      add(panel, el('p', '', '代码讲解来自章节 README 与源码结构；本书论断以 PDF 原页为准。'));
      const prerequisites = data.edges.filter(edge => edge[1] === c.id).map(edge => concepts.get(edge[0]));
      const next = data.edges.filter(edge => edge[0] === c.id).map(edge => concepts.get(edge[1]));
      addRelations(panel, prerequisites, '无前置要求'); add(panel, el('h4', '', '由此继续')); addRelations(panel, next, '暂无后续依赖');
      add(panel, el('h4', '', '工程检查'), el('p', '', c.check), el('h4', '', '本书依据'), el('p', 'citation', c.source));
      for (const [title, detail] of [['适用与取舍', c.tradeoffs], ['如何验证', c.verification], ['容易误判', c.misjudgments]]) {
        if (detail) add(panel, el('h4', '', title), el('p', '', detail));
      }
      if (!chapters.has(c.chapter)) add(panel, el('p', '', '此概念卡关联待发布章节；请以卡片列出的已发布 PDF 引用为准。'));
    }
  }
  function addRelations(panel, rows, empty) {
    if (!rows.length) { add(panel, el('p', '', empty)); return; }
    const list = el('ul'); rows.forEach(row => { const item = el('li'); const button = el('button', 'inline-link', row.title); button.type = 'button'; button.addEventListener('click', () => { state.kind = 'concepts'; state.group = 'all'; state.query = ''; $('search').value = ''; state.selected = `k:${row.id}`; render(); }); add(list, add(item, button)); }); add(panel, list);
  }
  data.paths.forEach(path => { const button = el('button', 'tab', path.title); button.type = 'button'; button.dataset.course = path.id; button.addEventListener('click', () => { state.course = path.id; renderCourses(); }); add($('course-tabs'), button); });
  function renderCourses() {
    document.querySelectorAll('#course-tabs .tab').forEach(b => { const active = b.dataset.course === state.course; b.classList.toggle('active', active); b.setAttribute('aria-pressed', active); });
    $('course-list').replaceChildren(); const path = data.paths.find(p => p.id === state.course);
    const intro = add(el('div', 'course-intro'), el('span', '', path.audience), link('查看完整课程与验收规则 ↗', path.url));
    add($('course-list'), intro);
    path.units.forEach(unit => { const card = el('article', 'course-unit'); add(card, el('span', '', `WEEK ${String(unit.week).padStart(2, '0')} / ${unit.id.toUpperCase()}`), el('h3', '', unit.title), el('p', '', unit.deliverable), el('p', 'reading', unit.chapters.length > 6 ? `复习已学的 ${unit.chapters.length} 章 · 综合迁移` : unit.chapters.map(chapterLabel).join(' · ')));
      const button = el('button', '', '查看关联章节 →'); button.type = 'button'; button.addEventListener('click', () => { state.kind = 'chapters'; state.group = 'all'; state.query = ''; $('search').value = ''; state.selected = `c:${unit.chapters.find(n => chapters.has(n))}`; render(); $('catalog').scrollIntoView({behavior:'smooth'}); }); add(card, button); add($('course-list'), card); });
  }
  $('concept-catalog-link').addEventListener('click', () => {
    state.kind = 'concepts'; state.group = 'all'; state.query = ''; state.selected = null;
    $('search').value = ''; render();
  });
  render(); renderCourses();
})();
