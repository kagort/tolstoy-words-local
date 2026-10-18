// Небольшие помощники интерфейса: графики, отметки вхождений, дерево зависимостей, прогресс задач.

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function plotLayout(extra) {
  const text = cssVar('--text'), muted = cssVar('--muted'), border = cssVar('--border');
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: text, family: 'inherit', size: 13 },
    margin: { l: 60, r: 20, t: 30, b: 80 },
    xaxis: { gridcolor: border, linecolor: border, tickfont: { color: muted } },
    yaxis: { gridcolor: border, linecolor: border, tickfont: { color: muted } },
    legend: { orientation: 'h', y: 1.12 },
  }, extra || {});
}

function drawChart(id, data, layout) {
  const el = document.getElementById(id);
  if (!el || typeof Plotly === 'undefined') return;
  Plotly.newPlot(el, data, plotLayout(layout), { responsive: true, displaylogo: false, locale: 'ru',
    toImageButtonOptions: { format: 'png', filename: id, scale: 2 } });
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
    data.active.forEach(j => {
      const row = document.querySelector(`[data-job="${j.id}"]`);
      if (!row) { location.reload(); return; }
      row.querySelector('.progress > div').style.width = Math.round(j.progress * 100) + '%';
      row.querySelector('.job-msg').textContent = j.message || '';
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
