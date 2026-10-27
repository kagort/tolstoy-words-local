// Небольшие помощники интерфейса: графики, отметки вхождений, дерево зависимостей, прогресс задач.

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function plotLayout(extra) {
  const text = cssVar('--text'), muted = cssVar('--muted'), border = cssVar('--border');
  const base = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: text, family: 'inherit', size: 13 },
    margin: { l: 60, r: 20, t: 30, b: 80 },
    xaxis: { gridcolor: border, linecolor: border, tickfont: { color: muted } },
    yaxis: { gridcolor: border, linecolor: border, tickfont: { color: muted } },
    legend: { orientation: 'h', y: 1.12 },
  };
  if (document.body && document.body.classList.contains('v2')) {
    // новый интерфейс: тема графиков из токенов (docs/DESIGN_TOKENS.md, раздел 7)
    const grid = cssVar('--grid') || border, surface = cssVar('--surface'), sans = cssVar('--font-sans') || 'inherit';
    const cats = [1, 2, 3, 4, 5, 6, 7, 8].map(i => cssVar('--cat-' + i)).filter(Boolean);
    const axis = { gridcolor: grid, linecolor: border, zerolinecolor: border, tickfont: { color: muted, size: 11 }, title: { font: { size: 12, color: muted } } };
    Object.assign(base, {
      font: { color: text, family: sans, size: 12 },
      xaxis: Object.assign({}, axis), yaxis: Object.assign({}, axis),
      hoverlabel: { bgcolor: surface, bordercolor: border, font: { family: sans, size: 12, color: text }, align: 'left' },
    });
    if (cats.length) base.colorway = cats;
  }
  // оси из extra дополняют оформление, а не заменяют его (иначе в тёмной теме сетка остаётся светлой)
  const out = Object.assign({}, base, extra || {});
  ['xaxis', 'yaxis'].forEach(k => { if (extra && extra[k]) out[k] = Object.assign({}, base[k], extra[k]); });
  return out;
}

// Новый интерфейс: встроенная шкала Plotly «Blues» заменяется шкалой из токенов (--surface-muted → --cat-1) на всех
// графиках сразу. У Plotly «Blues» тёмный цвет — у малых значений, поэтому reversescale меняется на обратный:
// где было «тёмное — большое», так и остаётся (и наоборот — например, в матрице расстояний «тёмное — близкие»).
function tokenScales(data) {
  if (!document.body || !document.body.classList.contains('v2')) return data;
  const seq = [[0, cssVar('--surface-muted')], [1, cssVar('--cat-1')]];
  (data || []).forEach(t => {
    if (t && t.colorscale === 'Blues') { t.colorscale = seq; t.reversescale = !t.reversescale; }
    if (t && t.marker && t.marker.colorscale === 'Blues') { t.marker.colorscale = seq; t.marker.reversescale = !t.marker.reversescale; }
  });
  return data;
}

function drawChart(id, data, layout) {
  const el = document.getElementById(id);
  if (!el || typeof Plotly === 'undefined') return;
  data = tokenScales(data);
  Plotly.newPlot(el, data, plotLayout(layout), { responsive: true, displaylogo: false, locale: 'ru',
    toImageButtonOptions: { format: 'png', filename: id, scale: 2 },
    topojsonURL: '/static/topojson/' });  // контуры карты мира лежат локально: интернет не нужен
}

// Не больше MAX_SERIES рядов на одном графике: дальше цвета неразличимы (все ряды остаются в тепловой карте и таблице).
const MAX_SERIES = 12;

// Столбцы «группа × слово (категория)» для обзора и отчёта. Подписи групп — короткие (g.label), полные — в подсказке (g.full).
// Группировка «Текст» и больше 6 групп — горизонтальные столбцы: длинные подписи читаются без наклона.
// opts.log — логарифмическая шкала (одно частое слово не сплющивает остальные).
function drawGroups(id, ov, metric, ytitle, extra, opts) {
  opts = opts || {};
  const labels = ov.groups.map(g => g.label), full = ov.groups.map(g => g.full || g.label);
  const isPeriod = ov.dim === 'period';
  const horizontal = !isPeriod && (ov.dim === 'text' || labels.length > 6);
  const sum = w => w[metric].reduce((a, b) => a + b, 0);
  const words = ov.words.length > MAX_SERIES ? [...ov.words].sort((a, b) => sum(b) - sum(a)).slice(0, MAX_SERIES) : ov.words;
  const traces = words.map(w => isPeriod
    ? { type: 'scatter', mode: 'lines+markers', name: w.word, x: labels, y: w[metric], customdata: full,
        hovertemplate: '%{customdata}<br>' + w.word + ': %{y}<extra></extra>' }
    : horizontal
      ? { type: 'bar', orientation: 'h', name: w.word, y: labels, x: w[metric], customdata: full,
          hovertemplate: '%{customdata}<br>' + w.word + ': %{x}<extra></extra>' }
      : { type: 'bar', name: w.word, x: labels, y: w[metric], customdata: full,
          hovertemplate: '%{customdata}<br>' + w.word + ': %{y}<extra></extra>' });
  const scale = opts.log ? { type: 'log' } : {};
  const layout = horizontal
    ? { barmode: 'group', xaxis: Object.assign({ title: ytitle }, scale), yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } },
        height: Math.max(320, 120 + labels.length * Math.max(22, 12 * words.length)), margin: { l: 20, r: 20, t: 50, b: 50 } }
    : { barmode: 'group', yaxis: Object.assign({ title: ytitle }, scale), xaxis: { automargin: true, tickangle: labels.length > 4 ? -30 : 0, tickfont: { size: 11 } },
        margin: { l: 60, r: 20, t: 50, b: 40 } };
  if (words.length > 4) {  // много рядов — легенда под графиком, чтобы не закрывала заголовок
    layout.legend = { orientation: 'h', x: 0, y: horizontal ? -0.08 : -0.22, yanchor: 'top' };
    layout.margin = Object.assign({}, layout.margin, { b: horizontal ? 60 + 22 * Math.ceil(words.length / 4) : 110 + 22 * Math.ceil(words.length / 4) });
    if (layout.height) layout.height += 22 * Math.ceil(words.length / 4);
  }
  if (ov.words.length > MAX_SERIES) {
    layout.annotations = [{ text: `на графике ${MAX_SERIES} самых частых из ${ov.words.length}; все — в тепловой карте и таблице`,
      xref: 'paper', yref: 'paper', x: 0, y: 1.08, showarrow: false, font: { size: 11, color: cssVar('--muted') }, xanchor: 'left' }];
  }
  const el = document.getElementById(id);
  if (el && layout.height) el.style.minHeight = layout.height + 'px';  // контейнер по высоте графика, без пустого места
  drawChart(id, traces, Object.assign(layout, extra || {}));
}

// Порядок строк по сходству профилей (жадно: следующей идёт самая похожая на предыдущую строка) —
// в тепловой карте похожие слова и тексты стоят рядом и видны блоками.
function similarityOrder(rows) {
  const n = rows.length;
  if (n < 3) return [...Array(n).keys()];
  const norm = rows.map(r => { const s = Math.sqrt(r.reduce((a, v) => a + v * v, 0)) || 1; return r.map(v => v / s); });
  const sim = (a, b) => norm[a].reduce((acc, v, k) => acc + v * norm[b][k], 0);
  const total = rows.map(r => r.reduce((a, v) => a + v, 0));
  const order = [total.indexOf(Math.max(...total))];
  const left = new Set([...Array(n).keys()].filter(i => i !== order[0]));
  while (left.size) {
    const last = order[order.length - 1];
    let best = null, bestS = -Infinity;
    left.forEach(i => { const v = sim(last, i); if (v > bestS) { bestS = v; best = i; } });
    order.push(best); left.delete(best);
  }
  return order;
}

// Тепловая карта «строки × столбцы» с подписями любой длины: при длинных или многочисленных столбцах таблица
// поворачивается (столбцы — по вертикали), высота — по числу строк, строки упорядочены по сходству.
// rows, cols — короткие подписи; colsFull — полные (подсказка); z[row][col].
function drawHeat(id, rows, cols, colsFull, z, opts) {
  opts = opts || {};
  const vertical = opts.vertical !== undefined ? opts.vertical : cols.length > 6 || cols.some(c => c.length > 18);
  let R = rows, C = cols, CF = colsFull || cols, Z = z;
  if (vertical) {  // транспонируем: столбцы становятся строками
    R = cols; C = rows; Z = cols.map((_, j) => rows.map((__, i) => z[i][j]));
  }
  const ord = opts.sort === false ? [...R.keys()] : similarityOrder(Z);
  const RF = vertical ? CF : R;
  const zz = ord.map(i => Z[i]), rr = ord.map(i => R[i]), rf = ord.map(i => RF[i]);
  const v2 = document.body && document.body.classList.contains('v2');  // новый интерфейс: шкала из токенов
  const trace = { type: 'heatmap', z: zz, x: C, y: rr, colorscale: v2 ? [[0, cssVar('--surface-muted')], [1, cssVar('--cat-1')]] : 'Blues',
    reversescale: !v2, xgap: v2 ? 1 : 0, ygap: v2 ? 1 : 0,
    customdata: vertical ? rf.map(f => C.map(() => f)) : rr.map(() => CF),
    hovertemplate: vertical ? '%{customdata} — %{x}: %{z}<extra></extra>' : '%{y} — %{customdata}: %{z}<extra></extra>' };
  const height = Math.max(200, 90 + 24 * rr.length);
  const el = document.getElementById(id);
  if (el) el.style.minHeight = height + 'px';
  drawChart(id, [trace], { height, xaxis: { side: 'top', automargin: true, tickangle: C.length > 8 ? -40 : 0, tickfont: { size: 11 } },
    yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } }, margin: { l: 20, r: 20, t: 60, b: 10 } });
}

// Тепловая карта «слово × группа» обзора.
function drawGroupHeat(id, ov, metric) {
  drawHeat(id, ov.words.map(w => w.word), ov.groups.map(g => g.label), ov.groups.map(g => g.full || g.label),
           ov.words.map(w => w[metric]), { vertical: ov.dim === 'text' || ov.groups.length > 6, sort: ov.dim !== 'period' });
}

async function setValid(hitId, value, btn) {
  const res = await fetch(`/api/hits/${hitId}/valid`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ valid: value }) });
  if (!res.ok) { alert('Не удалось сохранить отметку'); return; }
  const tr = btn.closest('tr');
  tr.classList.toggle('invalid', value === false);
  tr.querySelectorAll('[data-valid]').forEach(b => b.classList.add('secondary'));
  btn.classList.remove('secondary');
}

// Дуговая диаграмма зависимостей (SVG) для страницы предложения.
// Высота дуги — по вложенности (как в displaCy), а не по расстоянию: так длинные предложения остаются читаемыми.
function drawTree(containerId, tokens, focus, showPunct) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const ctx = document.createElement('canvas').getContext('2d');
  ctx.font = '15px sans-serif';
  const gap = 16, pad = 12;
  let x = pad;
  const pos = {};
  tokens.forEach(t => {
    const w = Math.max(ctx.measureText(t.form).width, 34);
    pos[t.i] = x + w / 2;
    x += w + gap;
  });
  const width = x + pad;
  const arcs = tokens.filter(t => t.head && (showPunct || t.upos !== 'PUNCT'))
    .map(t => ({ t, lo: Math.min(t.i, t.head), hi: Math.max(t.i, t.head), level: 1 }))
    .sort((a, b) => (a.hi - a.lo) - (b.hi - b.lo));
  arcs.forEach((a, k) => {
    const placed = arcs.slice(0, k);
    placed.forEach(b => { if (b.lo >= a.lo && b.hi <= a.hi) a.level = Math.max(a.level, b.level + 1); });
    // пересекающиеся дуги не должны лежать на одной высоте
    while (placed.some(b => b.level === a.level && a.lo < b.hi && b.lo < a.hi)) a.level += 1;
  });
  const maxLevel = Math.max(1, ...arcs.map(a => a.level));
  const levelH = 24, baseY = maxLevel * levelH + 46, height = baseY + 44;
  const text = cssVar('--text'), muted = cssVar('--muted'), accent = cssVar('--accent');
  const roleColor = { subject: cssVar('--r-subject'), predicate: cssVar('--r-predicate'), attribute: cssVar('--r-attribute'),
    object: cssVar('--r-object'), adverbial: cssVar('--r-adverbial') };
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="sans-serif">`;
  svg += `<defs><marker id="arr" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${muted}"/></marker>`;
  svg += `<marker id="arrA" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${accent}"/></marker></defs>`;
  const y0 = baseY - 20;
  tokens.filter(t => !t.head).forEach(t => {
    svg += `<line x1="${pos[t.i]}" y1="14" x2="${pos[t.i]}" y2="${y0}" stroke="${accent}" stroke-width="1.5" marker-end="url(#arrA)"/>`;
    svg += `<text x="${pos[t.i] + 5}" y="24" font-size="11" fill="${accent}">root</text>`;
  });
  arcs.forEach(a => {
    const x1 = pos[a.t.head], x2 = pos[a.t.i];
    const h = a.level * levelH;
    const hl = (a.t.i === focus || a.t.head === focus);
    const color = hl ? accent : muted;
    const r = Math.min(12, Math.abs(x2 - x1) / 2);
    const dir = x2 > x1 ? 1 : -1;
    // прямоугольная дуга со скруглёнными углами — подписи не налезают друг на друга
    svg += `<path d="M${x1},${y0} L${x1},${y0 - h + r} Q${x1},${y0 - h} ${x1 + dir * r},${y0 - h} L${x2 - dir * r},${y0 - h} Q${x2},${y0 - h} ${x2},${y0 - h + r} L${x2},${y0}" fill="none" stroke="${color}" stroke-width="${hl ? 2 : 1}" marker-end="url(#${hl ? 'arrA' : 'arr'})"/>`;
    const label = esc(a.t.rel);
    const lx = (x1 + x2) / 2, ly = y0 - h;
    const lw = label.length * 6.2 + 6;
    svg += `<rect x="${lx - lw / 2}" y="${ly - 7}" width="${lw}" height="13" fill="${cssVar('--panel')}"/>`;
    svg += `<text x="${lx}" y="${ly + 4}" font-size="11" text-anchor="middle" fill="${color}">${label}</text>`;
  });
  tokens.forEach(t => {
    const c = roleColor[t.role] || text;
    const weight = t.i === focus ? 'bold' : 'normal';
    svg += `<a href="?token=${t.i}"><text x="${pos[t.i]}" y="${baseY + 4}" font-size="15" text-anchor="middle" fill="${c}" font-weight="${weight}">${esc(t.form)}</text></a>`;
    svg += `<text x="${pos[t.i]}" y="${baseY + 21}" font-size="11" text-anchor="middle" fill="${muted}">${esc(t.upos)}</text>`;
    svg += `<text x="${pos[t.i]}" y="${baseY + 35}" font-size="10" text-anchor="middle" fill="${muted}">${t.i}</text>`;
  });
  svg += '</svg>';
  el.innerHTML = svg;
}

// Автообновление прогресса задач на странице «Задачи»
function watchJobs() {
  const box = document.getElementById('jobs-live');
  if (!box) return;
  const shown = [...box.querySelectorAll('[data-job]')]
    .filter(r => ['queued', 'running'].includes(r.dataset.status)).map(r => +r.dataset.job);
  setInterval(async () => {
    const res = await fetch('/api/jobs');
    if (!res.ok) return;
    const data = await res.json();
    const now = new Set(data.active.map(j => j.id));
    // задача завершилась или появилась новая — перерисовываем страницу целиком
    if (!data.active.length || shown.some(id => !now.has(id))) { location.reload(); return; }
    const q = document.getElementById('queue-eta');
    if (q) q.textContent = `Активных задач: ${data.queue.n}` + (data.queue.eta ? ` · вся очередь ≈ ${data.queue.eta}` : '');
    data.active.forEach(j => {
      const row = document.querySelector(`[data-job="${j.id}"]`);
      if (!row) { location.reload(); return; }
      row.querySelector('.progress > div').style.width = Math.round(j.progress * 100) + '%';
      row.querySelector('.job-msg').textContent = j.message || '';
      const eta = row.querySelector('.job-eta');
      if (eta) eta.textContent = j.eta ? '≈ ещё ' + j.eta : '';
      row.querySelector('.badge').textContent = j.status === 'running' ? 'выполняется' : 'в очереди';
    });
  }, 2500);
}
document.addEventListener('DOMContentLoaded', watchJobs);

// Всплывающее окно с соседними предложениями: наведите курсор на строку с атрибутом data-ctx
(function () {
  const cache = {};
  let box = null, timer = null, current = null, lastEvent = null;

  function popup() {
    if (!box) { box = document.createElement('div'); box.className = 'ctx-popup'; document.body.appendChild(box); }
    return box;
  }
  async function load(id) {
    if (!(id in cache)) {
      try {
        const res = await fetch(`/api/sentences/${id}/context?n=2`);
        cache[id] = res.ok ? await res.json() : null;
      } catch (e) { cache[id] = null; }
    }
    return cache[id];
  }
  function render(data, start, end) {
    const b = popup();
    b.textContent = '';
    const src = document.createElement('div');
    src.className = 'ctx-src';
    src.textContent = data.source + ' · контекст ±2 предложения';
    b.appendChild(src);
    let par = null;
    data.sentences.forEach(s => {
      if (par !== null && s.paragraph !== par) { const hr = document.createElement('div'); hr.className = 'ctx-par'; b.appendChild(hr); }
      par = s.paragraph;
      const div = document.createElement('div');
      div.className = 'ctx-sent' + (s.current ? ' current' : '');
      const id = document.createElement('span');
      id.className = 'ctx-id';
      id.textContent = `ID ${s.id} · № ${s.idx}`;
      div.appendChild(id);
      if (s.current && start !== '' && end !== '' && !isNaN(+start)) {
        div.appendChild(document.createTextNode(s.text.slice(0, +start)));
        const m = document.createElement('mark');
        m.textContent = s.text.slice(+start, +end);
        div.appendChild(m);
        div.appendChild(document.createTextNode(s.text.slice(+end)));
      } else {
        div.appendChild(document.createTextNode(s.text));
      }
      b.appendChild(div);
    });
  }
  function place(ev) {
    if (!box || box.style.display !== 'block' || !ev) return;
    const pad = 18, w = box.offsetWidth, h = box.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = Math.max(8, ev.clientX - w - pad);
    if (y + h > window.innerHeight - 8) y = Math.max(8, ev.clientY - h - pad);
    box.style.left = x + 'px';
    box.style.top = y + 'px';
  }
  function hide() { clearTimeout(timer); current = null; if (box) box.style.display = 'none'; }

  document.addEventListener('mouseover', ev => {
    const row = ev.target.closest('[data-ctx]');
    if (!row || row === current) return;
    hide();
    current = row;
    timer = setTimeout(async () => {
      const data = await load(row.dataset.ctx);
      if (!data || current !== row) return;
      render(data, row.dataset.start ?? '', row.dataset.end ?? '');
      popup().style.display = 'block';
      place(lastEvent);
    }, 350);
  });
  document.addEventListener('mousemove', ev => { lastEvent = ev; place(ev); });
  document.addEventListener('mouseout', ev => {
    if (current && !current.contains(ev.relatedTarget)) hide();
  });
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape') hide(); });
  window.addEventListener('scroll', hide, { passive: true });
})();

// ------------------------------------------------------------ панель «Выборка» и выбор слов
function selCorpusChanged(sel) {
  // отметки текстов относятся к прежнему корпусу — снимаем их и сразу применяем
  const form = document.getElementById('sel-form');
  form.querySelectorAll('input[name="texts"]').forEach(c => { c.checked = false; });
  form.submit();
}
function selDictChanged(sel) {
  sel.form.querySelectorAll('input[name="words"]').forEach(c => { c.checked = false; });
  sel.form.submit();
}
function _box(target) { return typeof target === 'string' ? document.querySelector(target) : target; }
function filterChecks(input, target) {
  const q = input.value.trim().toLowerCase();
  _box(target).querySelectorAll('[data-name]').forEach(el => {
    el.style.display = !q || el.dataset.name.includes(q) ? '' : 'none';
  });
}
function checkAll(target, on) {
  _box(target).querySelectorAll('input[type="checkbox"]').forEach(c => {
    const item = c.closest('[data-name]');
    if (!item || item.style.display !== 'none') c.checked = on;
  });
}
function checkTop(target, n) {
  const items = [..._box(target).querySelectorAll('[data-count]')];
  items.forEach(el => { el.querySelector('input').checked = false; });
  items.filter(el => +el.dataset.count > 0).sort((a, b) => b.dataset.count - a.dataset.count).slice(0, n)
    .forEach(el => { el.querySelector('input').checked = true; });
}
function checkFound(target) {
  _box(target).querySelectorAll('[data-count]').forEach(el => { el.querySelector('input').checked = +el.dataset.count > 0; });
}
// флажок в таблице частот отмечает то же слово в блоке «Слова словаря» (именно оно уходит в запрос)
function syncWord(box) {
  const twin = document.querySelector(`#word-list input[name="words"][value="${box.value}"]`);
  if (twin) twin.checked = box.checked;
}

// ------------------------------------------------------------ список предложений с подсветкой (имена, совместная встречаемость)
async function showMarked(url, title) {
  const res = await fetch(url);
  if (!res.ok) return;
  const data = await res.json();
  const box = document.getElementById('marked');
  if (!box) return;
  box.innerHTML = '';
  const list = data.sentences || [];
  for (const s of list) {
    const p = document.createElement('div'); p.className = 'example';
    const a = document.createElement('a'); a.href = `/sentences/${s.id}`; a.className = 'muted small'; a.textContent = `🌳 ${s.id} `; p.appendChild(a);
    let pos = 0;
    for (const [st, en] of s.marks || []) {
      if (st < pos) continue;
      p.appendChild(document.createTextNode(s.text.slice(pos, st)));
      const b = document.createElement('mark'); b.textContent = s.text.slice(st, en); p.appendChild(b); pos = en;
    }
    p.appendChild(document.createTextNode(s.text.slice(pos)));
    if (s.source) { const src = document.createElement('span'); src.className = 'muted small'; src.textContent = ' — ' + s.source; p.appendChild(src); }
    box.appendChild(p);
  }
  if (!list.length) box.textContent = 'Примеров нет.';
  const total = data.total && data.total > list.length ? ` (показаны ${list.length} из ${data.total})` : ` (${list.length})`;
  document.getElementById('marked-title').textContent = title + total;
  const wrap = document.getElementById('marked-box');
  wrap.style.display = '';
  wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Подсказки к метрикам: значок «?» (data-term) показывает краткое определение из глоссария (/help/term/<id>.json).
(function () {
  const cache = {};
  let pop = null, timer = null;
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  function hide() { if (pop) { pop.remove(); pop = null; } }
  async function show(el) {
    const id = el.dataset.term;
    if (!cache[id]) {
      const res = await fetch(`/help/term/${encodeURIComponent(id)}.json`);
      if (!res.ok) return;
      cache[id] = await res.json();
    }
    const t = cache[id];
    hide();
    pop = document.createElement('div');
    pop.className = 'term-pop';
    pop.innerHTML = `<strong>${esc(t.title)}</strong>${esc(t.short)}<br><a href="${t.url}">Подробнее в справке →</a>`;
    document.body.appendChild(pop);
    const r = el.getBoundingClientRect();
    const left = Math.min(window.scrollX + r.left, window.scrollX + document.documentElement.clientWidth - pop.offsetWidth - 10);
    pop.style.left = Math.max(10, left) + 'px';
    pop.style.top = (window.scrollY + r.bottom + 6) + 'px';
    pop.addEventListener('mouseenter', () => clearTimeout(timer));
    pop.addEventListener('mouseleave', () => { timer = setTimeout(hide, 250); });
  }
  document.addEventListener('mouseover', e => {
    const el = e.target.closest && e.target.closest('[data-term]');
    if (el) { clearTimeout(timer); show(el); }
  });
  document.addEventListener('mouseout', e => {
    const el = e.target.closest && e.target.closest('[data-term]');
    if (el) timer = setTimeout(hide, 250);
  });
  document.addEventListener('focusin', e => { if (e.target.dataset && e.target.dataset.term) show(e.target); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
})();

// «Вероятно, один персонаж»: отмечает имена группы в указателе и вписывает самое полное имя
function prefillMerge(names) {
  document.querySelectorAll('.merge-check').forEach(c => { c.checked = names.includes(c.value); });
  const longest = names.reduce((a, b) => (b.length > a.length ? b : a), '');
  const inp = document.getElementById('merge-canonical');
  if (inp) { inp.value = longest; inp.scrollIntoView({ behavior: 'smooth', block: 'center' }); inp.focus(); }
}

// ------------------------------------------------------------ указатель имён: поиск по мере ввода, групповые правки
function entParams() {
  const form = document.getElementById('sel-form');
  return new URLSearchParams(new FormData(form)).toString();
}
let entTimer = null, entSeq = 0;
async function entRefresh() {
  const qs = entParams(), seq = ++entSeq;
  const res = await fetch('/analysis/entities-index?' + qs);
  if (!res.ok || seq !== entSeq) return;  // ответ на устаревший ввод — не показываем
  const box = document.getElementById('ent-index');
  if (box) box.innerHTML = await res.text();
  history.replaceState(null, '', '/analysis?' + qs);
  entCount();
}
function initEntityIndex() {
  document.querySelectorAll('[data-ent-live]').forEach(el => {
    const ev = el.type === 'search' ? 'input' : 'change';
    el.addEventListener(ev, () => { clearTimeout(entTimer); entTimer = setTimeout(entRefresh, el.type === 'search' ? 300 : 0); });
    if (el.type === 'search') el.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); clearTimeout(entTimer); entRefresh(); } });
  });
}
function entFlag(flag) {
  const box = document.querySelector(`[data-ent-live][name="f"][value="${flag}"]`);
  if (box) { box.checked = true; entRefresh(); }
}
function entShowHidden() {
  const s = document.querySelector('[data-ent-live][name="hidden"]');
  if (s) { s.value = '1'; entRefresh(); }
}
function entAllNames() {
  const el = document.getElementById('ent-all');
  return el ? JSON.parse(el.textContent) : [];
}
function entCheckAll(on) {
  document.querySelectorAll('.merge-check').forEach(c => { c.checked = on; });
  const f = document.querySelector('#bulk-form [name="names_all"]');
  if (f) f.value = on ? entAllNames().join('\n') : '';
  entCount();
}
function entChecked() {
  const f = document.querySelector('#bulk-form [name="names_all"]');
  const names = new Set([...document.querySelectorAll('.merge-check:checked')].map(c => c.value));
  if (f && f.value) f.value.split('\n').forEach(n => names.add(n));
  return [...names];
}
function entCount() {
  const el = document.getElementById('ent-checked');
  const f = document.querySelector('#bulk-form [name="names_all"]');
  if (el) el.textContent = `отмечено: ${entChecked().length}` + (f && f.value ? ' (все найденные)' : '');
}
function entPrepare() {
  if (!entChecked().length) { toast('Отметьте имена в указателе'); return false; }
  return true;
}
async function entHide(names, hidden, quiet) {
  if (!names.length) { toast('Отметьте имена в указателе'); return; }
  const res = await fetch('/api/entities/hide', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names, hidden }) });
  if (!res.ok) { toast('Не удалось сохранить правку'); return; }
  await entRefresh();
  if (quiet) return;
  const what = names.length === 1 ? `«${names[0]}»` : `имён: ${names.length}`;
  toast(hidden ? `Скрыто ${what}` : `Возвращено ${what}`, hidden ? { label: 'Отменить', run: () => entHide(names, false, true) } : null);
}
// всплывающее сообщение внизу экрана (с кнопкой действия, например «Отменить»)
function toast(text, action) {
  let el = document.getElementById('toast');
  if (!el) { el = document.createElement('div'); el.id = 'toast'; el.className = 'toast'; document.body.appendChild(el); }
  el.innerHTML = '';
  el.appendChild(document.createTextNode(text + ' '));
  if (action) {
    const b = document.createElement('button'); b.className = 'small secondary'; b.textContent = action.label;
    b.onclick = () => { el.classList.remove('show'); action.run(); }; el.appendChild(b);
  }
  el.classList.add('show');
  clearTimeout(el._t); el._t = setTimeout(() => el.classList.remove('show'), 8000);
}

// ------------------------------------------------------------ новый интерфейс: тема (светлая / тёмная / как в системе)
// Графики Plotly строятся с цветами текущей темы, поэтому после смены темы страница с графиками перезагружается.
function currentTheme() { return document.documentElement.dataset.theme || 'auto'; }
function markTheme() {
  document.querySelectorAll('[data-theme-set]').forEach(b => b.classList.toggle('on', b.dataset.themeSet === currentTheme()));
}
function setTheme(t) {
  try { if (t === 'auto') localStorage.removeItem('qv-theme'); else localStorage.setItem('qv-theme', t); } catch (e) {}
  if (t === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  markTheme();
  if (document.querySelector('.js-plotly-plot')) location.reload();
}
document.addEventListener('DOMContentLoaded', () => {
  markTheme();
  document.querySelectorAll('[data-theme-set]').forEach(b => b.addEventListener('click', () => setTheme(b.dataset.themeSet)));
  document.querySelectorAll('[data-theme-cycle]').forEach(b => b.addEventListener('click', () => {
    const order = ['auto', 'light', 'dark'];
    setTheme(order[(order.indexOf(currentTheme()) + 1) % order.length]);
  }));
});
