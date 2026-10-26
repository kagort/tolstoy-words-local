// Графики дашборда v2 (services/dashboard_v2.py, шаблон _dashboard_v2_body.html). Цвета — только из токенов
// docs/DESIGN_TOKENS.md: категории --cat-1..8 (девятая и дальше — --inactive), расхождения --div-0..4, роли --r-*.
// Используется и на странице приложения, и в автономном HTML-файле (встраивается целиком).

function dv2Color(idx) { return idx === null || idx === undefined ? cssVar('--inactive') : cssVar('--cat-' + (idx + 1)); }
function dv2Diverging() { return [0, 1, 2, 3, 4].map(i => [i / 4, cssVar('--div-' + i)]); }
function dv2Sequential() { return [[0, cssVar('--surface-muted')], [1, cssVar('--cat-1')]]; }
function dv2Fmt(v, d) { return Number(v).toLocaleString('ru-RU', { maximumFractionDigits: d === undefined ? 1 : d }); }
function dv2Height(id, h) { const el = document.getElementById(id); if (el) el.style.minHeight = h + 'px'; return h; }

// 1. Покрытие: категории (или слова) по числу вхождений — горизонтальные столбцы в цветах категорий.
function dv2Coverage(v) {
  let rows = v.has_cats ? v.categories.map(c => ({ name: c.name, n: c.count, color: c.color }))
    : (v.core ? [...v.core.points].sort((a, b) => b.count - a.count).slice(0, 20).map(p => ({ name: p.word, n: p.count, color: 0 })) : []);
  if (!rows.length) return;
  if (rows.length > 16) {  // длинный хвост категорий — одной строкой
    const rest = rows.slice(15);
    rows = rows.slice(0, 15).concat([{ name: `остальные (${rest.length})`, n: rest.reduce((a, r) => a + r.n, 0), color: null }]);
  }
  const h = dv2Height('dv2-cov', Math.max(220, 60 + 26 * rows.length));
  drawChart('dv2-cov', [{ type: 'bar', orientation: 'h', y: rows.map(r => r.name), x: rows.map(r => r.n),
    marker: { color: rows.map(r => dv2Color(r.color)) }, hovertemplate: '%{y}: %{x} вхожд.<extra></extra>' }],
    { height: h, xaxis: { title: { text: 'вхождений' } }, yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } },
      margin: { l: 20, r: 20, t: 10, b: 44 }, bargap: 0.35 });
}

// 2. Ядро и периферия: ipm (лог. шкала) × DP; вверху справа — частые и равномерные.
function dv2Core(c) {
  if (!c) return;
  const pts = c.points, ipms = pts.map(p => p.ipm).sort((a, b) => a - b), med = ipms[Math.floor(ipms.length / 2)];
  // в легенде — только категории в цвете; серые (девятая и дальше) — одной строкой «остальные категории»
  const groups = {};
  pts.forEach(p => { const k = p.color === null || p.color === undefined ? (p.cat ? '\u0000' : '') : p.cat; (groups[k] = groups[k] || []).push(p); });
  const keys = Object.keys(groups).sort((a, b) => (groups[a][0].color ?? 99) - (groups[b][0].color ?? 99));
  const traces = keys.map(k => {
    const g = groups[k];
    return { type: 'scatter', mode: 'markers+text', name: k === '\u0000' ? 'остальные категории' : (k || 'слова'), x: g.map(p => p.ipm), y: g.map(p => p.dp),
      text: g.map(p => p.label ? p.word : ''), textposition: 'top center', textfont: { size: 11, color: cssVar('--text') },
      customdata: g.map(p => [p.word, p.count, p.texts]), showlegend: !!k,
      marker: { size: g.map(p => 6 + 3 * Math.log10(1 + p.count)), color: dv2Color(g[0].color), opacity: 0.85,
                line: { width: 0.5, color: cssVar('--surface') } },
      hovertemplate: '<b>%{customdata[0]}</b><br>ipm %{x}<br>DP %{y}<br>вхождений %{customdata[1]}, текстов %{customdata[2]}<extra></extra>' };
  });
  const muted = cssVar('--text-muted'), border = cssVar('--border');
  drawChart('dv2-core', traces, { height: dv2Height('dv2-core', 440),
    xaxis: { type: 'log', dtick: 1, title: { text: 'частота, ipm (логарифмическая шкала)' } },
    yaxis: { title: { text: 'DP: 0 — равномерно, 1 — в одном тексте' }, range: [1.05, -0.05] },
    shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: med, x1: med, y0: 0, y1: 1, line: { color: border, dash: 'dot', width: 1 } },
             { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: border, dash: 'dot', width: 1 } }],
    annotations: [{ text: 'ядро', xref: 'paper', yref: 'paper', x: 0.99, y: 0.99, showarrow: false, xanchor: 'right', font: { size: 12, color: muted } },
                  { text: 'периферия', xref: 'paper', yref: 'paper', x: 0.01, y: 0.01, showarrow: false, xanchor: 'left', yanchor: 'bottom', font: { size: 12, color: muted } }],
    legend: { orientation: 'h', x: 0, y: -0.18, yanchor: 'top' }, margin: { l: 60, r: 20, t: 10, b: 90 } });
}

// 3. Профиль категорий: радар (2–6 групп), тепловая карта отклонений (больше 6), столбцы (одна группа).
function dv2Profile(p) {
  if (!p) return;
  if (p.form === 'radar') {
    const theta = [...p.cats, p.cats[0]];
    const traces = p.groups.map(g => ({ type: 'scatterpolar', name: g, r: [...p.share[g], p.share[g][0]], theta, mode: 'lines+markers',
      line: { width: 2 }, marker: { size: 5 }, hovertemplate: g + '<br>%{theta}: %{r} %<extra></extra>' }));
    traces.push({ type: 'scatterpolar', name: 'среднее', r: [...p.avg, p.avg[0]], theta, mode: 'lines',
      line: { width: 1.5, dash: 'dash', color: cssVar('--inactive') }, hovertemplate: 'среднее<br>%{theta}: %{r} %<extra></extra>' });
    const grid = cssVar('--grid'), muted = cssVar('--text-muted');
    drawChart('dv2-prof', traces, { height: dv2Height('dv2-prof', 480),
      polar: { bgcolor: 'rgba(0,0,0,0)', radialaxis: { gridcolor: grid, linecolor: grid, tickfont: { size: 10, color: muted }, ticksuffix: ' %' },
               angularaxis: { gridcolor: grid, linecolor: grid, tickfont: { size: 11 } } },
      legend: { orientation: 'h', x: 0, y: -0.08, yanchor: 'top' }, margin: { l: 70, r: 70, t: 30, b: 60 } });
  } else if (p.form === 'heatmap') {
    const z = p.groups.map(g => p.z[g]);
    const h = dv2Height('dv2-prof', Math.max(260, 120 + 24 * p.groups.length));
    drawChart('dv2-prof', [{ type: 'heatmap', z, x: p.cats, y: p.groups, zmid: 0, zmin: -2.5, zmax: 2.5, colorscale: dv2Diverging(),
      customdata: p.groups.map(g => p.ipm[g]), colorbar: { title: { text: 'z' }, thickness: 10 },
      hovertemplate: '%{y} — %{x}<br>z = %{z}<br>ipm %{customdata}<extra></extra>' }],
      { height: h, xaxis: { side: 'top', automargin: true, tickangle: p.cats.length > 5 ? -30 : 0, tickfont: { size: 11 } },
        yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } }, margin: { l: 20, r: 20, t: 80, b: 10 } });
  } else {
    const g = p.groups[0];
    drawChart('dv2-prof', [{ type: 'bar', x: p.cats, y: p.share[g], marker: { color: p.cats.map((_, i) => dv2Color(i)) },
      hovertemplate: '%{x}: %{y} %<extra></extra>' }], { yaxis: { ticksuffix: ' %' }, xaxis: { automargin: true, tickangle: -20 } });
  }
}

// 4. Характерные слова группы: log-ratio (вправо — чаще в группе, влево — реже), бледные — незначимые (G² < порога).
function dv2Distinct(d, group) {
  if (!d) return;
  const g = group || d.groups[0], items = d.by_group[g].items;
  const el = document.getElementById('dv2-dist');
  if (!items.length) { if (el) el.innerHTML = '<p class="muted small">Для этой группы значимых различий нет.</p>'; return; }
  const pos = cssVar('--div-4'), neg = cssVar('--div-0'), y = items.map(x => x.word);
  const color = items.map(x => x.lr > 0 ? pos : neg), op = items.map(x => x.ll >= d.sig ? 1 : 0.4);
  const h = dv2Height('dv2-dist', Math.max(240, 60 + 24 * items.length));
  drawChart('dv2-dist', [
    { type: 'scatter', mode: 'lines', x: items.flatMap(x => [0, x.lr, null]), y: items.flatMap(x => [x.word, x.word, null]),
      line: { color: cssVar('--border'), width: 2 }, hoverinfo: 'skip', showlegend: false },
    { type: 'scatter', mode: 'markers', x: items.map(x => x.lr), y, marker: { size: 10, color, opacity: op }, showlegend: false,
      customdata: items.map(x => [x.a, x.b, x.ll]),
      hovertemplate: '<b>%{y}</b><br>log-ratio %{x}<br>в группе %{customdata[0]}, в остальных %{customdata[1]}<br>G² %{customdata[2]}<extra></extra>' }],
    { height: h, xaxis: { title: { text: '← реже в группе   log-ratio   чаще в группе →' }, zeroline: true },
      yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } }, margin: { l: 20, r: 20, t: 10, b: 50 } });
}

// 4а. Две группы — рассеяние в стиле Scattertext: по осям ipm в каждой группе, у диагонали — общие слова.
function dv2Pair(pair) {
  if (!pair || !pair.points.length) return;
  const pts = pair.points, lr = pts.map(p => Math.log2(p.x / p.y));
  const order = [...pts.keys()].sort((a, b) => Math.abs(lr[b]) - Math.abs(lr[a]));
  const label = new Set(order.slice(0, 14));
  const pos = cssVar('--div-4'), neg = cssVar('--div-0'), mid = cssVar('--inactive');
  const all = pts.flatMap(p => [p.x, p.y]), lo = Math.min(...all), hi = Math.max(...all);
  drawChart('dv2-pair', [{ type: 'scatter', mode: 'markers+text', x: pts.map(p => p.x), y: pts.map(p => p.y),
    text: pts.map((p, i) => label.has(i) ? p.word : ''), textposition: 'top center', textfont: { size: 10 },
    marker: { size: pts.map(p => 5 + 2 * Math.log10(1 + p.n)), color: lr.map(v => v > 0.5 ? pos : v < -0.5 ? neg : mid), opacity: 0.8 },
    customdata: pts.map(p => [p.word, p.n]), hovertemplate: '<b>%{customdata[0]}</b><br>' + pair.a + ': %{x} ipm<br>' + pair.b + ': %{y} ipm<extra></extra>' }],
    { height: dv2Height('dv2-pair', 460), xaxis: { type: 'log', title: { text: pair.a + ', ipm' } }, yaxis: { type: 'log', title: { text: pair.b + ', ipm' } },
      shapes: [{ type: 'line', x0: lo, y0: lo, x1: hi, y1: hi, line: { color: cssVar('--border'), dash: 'dot' } }], margin: { l: 60, r: 20, t: 10, b: 50 } });
}

// 5. Время: ipm по десятилетиям.
function dv2Time(t) {
  if (!t) return;
  drawChart('dv2-time', t.series.map(s => ({ type: 'scatter', mode: 'lines+markers', name: s.name, x: t.periods, y: s.ipm,
    connectgaps: false, hovertemplate: s.name + ' · %{x}: %{y} ipm<extra></extra>' })),
    { height: dv2Height('dv2-time', 380), yaxis: { title: { text: 'ipm' }, rangemode: 'tozero' },
      legend: { orientation: 'h', x: 0, y: -0.15, yanchor: 'top' }, margin: { l: 60, r: 20, t: 10, b: 80 } });
}

// 6. Синтаксические роли: 100 % столбцы по категориям или словам.
function dv2Syntax(s) {
  if (!s) return;
  const keys = ['--r-subject', '--r-predicate', '--r-object', '--r-attribute', '--r-adverbial', '--inactive'];
  const names = s.rows.map(r => r.name);
  const traces = s.roles.map((role, k) => ({ type: 'bar', orientation: 'h', name: role, y: names, x: s.rows.map(r => r.pct[k]),
    marker: { color: cssVar(keys[k]) }, customdata: s.rows.map(r => r.total),
    hovertemplate: '%{y}<br>' + role + ': %{x} %<br>(всего %{customdata})<extra></extra>' }));
  drawChart('dv2-syn', traces, { barmode: 'stack', height: dv2Height('dv2-syn', Math.max(220, 110 + 30 * names.length)),
    xaxis: { range: [0, 100], ticksuffix: ' %' }, yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } },
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', traceorder: 'normal' }, margin: { l: 20, r: 20, t: 40, b: 40 }, bargap: 0.3 });
}

// 7. Сеть совместной встречаемости (в одном предложении).
function dv2Network(n) {
  if (!n) return;
  const at = {}; n.nodes.forEach(x => { at[x.id] = x; });
  const maxW = Math.max(...n.edges.map(e => e.w));
  const edges = [1, 2, 3].map(level => {
    const es = n.edges.filter(e => Math.ceil(3 * e.w / maxW) === level);
    return { type: 'scatter', mode: 'lines', hoverinfo: 'skip', showlegend: false,
      x: es.flatMap(e => [at[e.a].x, at[e.b].x, null]), y: es.flatMap(e => [at[e.a].y, at[e.b].y, null]),
      line: { width: level * 1.2, color: cssVar('--border') } };
  });
  const maxN = Math.max(...n.nodes.map(x => x.n));
  const nodes = { type: 'scatter', mode: 'markers+text', x: n.nodes.map(x => x.x), y: n.nodes.map(x => x.y), text: n.nodes.map(x => x.word),
    textposition: 'top center', textfont: { size: 11 }, showlegend: false, customdata: n.nodes.map(x => [x.word, x.n, x.cat]),
    marker: { size: n.nodes.map(x => 8 + 22 * Math.sqrt(x.n / maxN)), color: n.nodes.map(x => dv2Color(x.color)),
              line: { width: 1, color: cssVar('--surface') } },
    hovertemplate: '<b>%{customdata[0]}</b><br>предложений: %{customdata[1]}<br>%{customdata[2]}<extra></extra>' };
  const off = { visible: false, showgrid: false, zeroline: false };
  drawChart('dv2-net', [...edges, nodes], { height: dv2Height('dv2-net', 500), xaxis: off, yaxis: off, margin: { l: 10, r: 10, t: 10, b: 10 } });
}

// 8. Персонажи × категории (тепловая карта) и места, окрашенные преобладающей категорией.
function dv2Names(nm) {
  if (!nm) return;
  if (nm.persons.length) {
    const h = dv2Height('dv2-per', Math.max(220, 110 + 24 * nm.persons.length));
    drawChart('dv2-per', [{ type: 'heatmap', z: nm.z, x: nm.keys, y: nm.persons, colorscale: dv2Sequential(), colorbar: { thickness: 10 },
      hovertemplate: '%{y} — %{x}: %{z} предл.<extra></extra>' }],
      { height: h, xaxis: { side: 'top', automargin: true, tickangle: nm.keys.length > 5 ? -30 : 0, tickfont: { size: 11 } },
        yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } }, margin: { l: 20, r: 20, t: 80, b: 10 } });
  }
  if (nm.places && nm.places.length) {
    const groups = {};
    nm.places.forEach(p => { (groups[p.cat] = groups[p.cat] || []).push(p); });
    const maxN = Math.max(...nm.places.map(p => p.n));
    drawChart('dv2-geo', Object.keys(groups).map(k => ({ type: 'scattergeo', mode: 'markers', name: k,
      lat: groups[k].map(p => p.lat), lon: groups[k].map(p => p.lon), customdata: groups[k].map(p => [p.name, p.n]),
      marker: { size: groups[k].map(p => 7 + 20 * Math.sqrt(p.n / maxN)), color: dv2Color(groups[k][0].color), opacity: 0.85,
                line: { width: 0.5, color: cssVar('--surface') } },
      hovertemplate: '<b>%{customdata[0]}</b><br>' + k + ': %{customdata[1]}<extra></extra>' })),
      { geo: { resolution: 110, showcountries: true, countrycolor: cssVar('--border'), showland: true, landcolor: cssVar('--surface-muted'),
               showocean: true, oceancolor: cssVar('--surface'), coastlinecolor: cssVar('--border'), projection: { type: 'natural earth', rotation: { lon: 10 } },
               lataxis: { range: [-58, 84] }, bgcolor: 'rgba(0,0,0,0)' },
        height: dv2Height('dv2-geo', 420), legend: { orientation: 'h', x: 0, y: -0.02, yanchor: 'top' }, margin: { l: 0, r: 0, t: 0, b: 30 } });
  }
}

// Один текст: плотность слов словаря по отрезкам текста; отдельные слова — включаются в легенде.
function dv2TextMap(tm) {
  if (!tm) return;
  const x = [...Array(tm.segments).keys()].map(k => k + 1);
  drawChart('dv2-map', [{ type: 'scatter', mode: 'lines', name: 'все слова словаря', x, y: tm.all.ipm, line: { width: 2.5, color: cssVar('--cat-1') },
    fill: 'tozeroy', fillcolor: 'rgba(0,0,0,0.04)' },
    ...tm.rows.map(r => ({ type: 'scatter', mode: 'lines', name: r.label, x, y: r.ipm, line: { width: 1 }, visible: 'legendonly' }))],
    { height: dv2Height('dv2-map', 360), xaxis: { title: { text: 'отрезок текста (от начала к концу)' } }, yaxis: { title: { text: 'ipm' } },
      legend: { orientation: 'h', x: 0, y: -0.22, yanchor: 'top' }, margin: { l: 60, r: 20, t: 10, b: 90 } });
}

function drawDashV2(v) {
  if (!v) return;
  dv2TextMap(v.textmap); dv2Coverage(v); dv2Core(v.core); dv2Profile(v.profile); dv2Distinct(v.distinct); dv2Pair(v.distinct && v.distinct.pair);
  dv2Time(v.time); dv2Syntax(v.syntax); dv2Network(v.network); dv2Names(v.names);
  // оглавление: подсветка раздела, который сейчас на экране
  const links = document.querySelectorAll('.dv2-toc a[href^="#"]');
  if (!links.length || !('IntersectionObserver' in window)) return;
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id));
  }), { rootMargin: '-20% 0px -70% 0px' });
  document.querySelectorAll('.dv2-sec[id]').forEach(s => io.observe(s));
}
