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

// Дуговая диаграмма зависимостей (SVG) для страницы предложения
function drawTree(containerId, tokens, focus) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const ctx = document.createElement('canvas').getContext('2d');
  ctx.font = '15px sans-serif';
  const gap = 18, pad = 10;
  let x = pad;
  const pos = {};
  tokens.forEach(t => {
    const w = Math.max(ctx.measureText(t.form).width, ctx.measureText(t.upos).width, 30);
    pos[t.i] = { x: x + w / 2, w };
    x += w + gap;
  });
  const width = x + pad;
  let maxDist = 1;
  tokens.forEach(t => { if (t.head) maxDist = Math.max(maxDist, Math.abs(t.head - t.i)); });
  const levelH = 26, baseY = Math.min(maxDist, 12) * levelH + 40;
  const height = baseY + 50;
  const text = cssVar('--text'), muted = cssVar('--muted'), accent = cssVar('--accent');
  const roleColor = { subject: cssVar('--r-subject'), predicate: cssVar('--r-predicate'), attribute: cssVar('--r-attribute'),
    object: cssVar('--r-object'), adverbial: cssVar('--r-adverbial') };
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="sans-serif">`;
  svg += `<defs><marker id="arr" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${muted}"/></marker></defs>`;
  tokens.forEach(t => {
    if (!t.head) {
      svg += `<line x1="${pos[t.i].x}" y1="10" x2="${pos[t.i].x}" y2="${baseY - 18}" stroke="${muted}" marker-end="url(#arr)"/>`;
      svg += `<text x="${pos[t.i].x + 4}" y="22" font-size="11" fill="${muted}">root</text>`;
      return;
    }
    const x1 = pos[t.head].x, x2 = pos[t.i].x;
    const h = Math.min(Math.abs(t.head - t.i), 12) * levelH;
    const y = baseY - 18;
    const hl = (t.i === focus || t.head === focus);
    const color = hl ? accent : muted;
    svg += `<path d="M${x1},${y} C${x1},${y - h} ${x2},${y - h} ${x2},${y}" fill="none" stroke="${color}" stroke-width="${hl ? 2 : 1}" marker-end="url(#arr)"/>`;
    svg += `<text x="${(x1 + x2) / 2}" y="${y - h * 0.75 - 3}" font-size="11" text-anchor="middle" fill="${color}">${t.rel}</text>`;
  });
  tokens.forEach(t => {
    const c = roleColor[t.role] || text;
    const weight = t.i === focus ? 'bold' : 'normal';
    svg += `<text x="${pos[t.i].x}" y="${baseY + 4}" font-size="15" text-anchor="middle" fill="${c}" font-weight="${weight}">${t.form.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</text>`;
    svg += `<text x="${pos[t.i].x}" y="${baseY + 22}" font-size="11" text-anchor="middle" fill="${muted}">${t.upos}</text>`;
  });
  svg += '</svg>';
  el.innerHTML = svg;
}

// Автообновление прогресса задач на странице «Задачи»
function watchJobs() {
  const box = document.getElementById('jobs-live');
  if (!box) return;
  setInterval(async () => {
    const res = await fetch('/api/jobs');
    if (!res.ok) return;
    const data = await res.json();
    if (!data.active.length) { location.reload(); return; }
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
