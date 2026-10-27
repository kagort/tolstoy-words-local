// Графики вкладок «Анализа» в новом интерфейсе (волна «Визуализация», шаг 5; docs/DESIGN_TOKENS.md).
// Подключается только в base_v2.html; шаблоны вызывают их, если есть window.QV2, иначе — прежние графики (классический вид).
// Цвета — только токены: категории --cat-1..8 (дальше --inactive), расхождения --div-0..4, роли --r-*.

window.QV2 = (function () {
  const v = name => cssVar(name);
  const cat = i => (i === null || i === undefined || i >= 8) ? v('--inactive') : v('--cat-' + (i + 1));
  const diverging = () => [0, 1, 2, 3, 4].map(i => [i / 4, v('--div-' + i)]);
  const sequential = () => [[0, v('--surface-muted')], [1, v('--cat-1')]];
  const height = (id, h) => { const el = document.getElementById(id); if (el) el.style.minHeight = h + 'px'; return h; };
  const fmt = x => Number(x).toLocaleString('ru-RU', { maximumFractionDigits: 1 });

  // порядок категорий по сумме — цвет закреплён за категорией (первые 8 в цвете, остальные серые)
  function catIndex(rows, catKey, valKey) {
    const tot = {};
    rows.forEach(r => { const c = r[catKey] || ''; tot[c] = (tot[c] || 0) + (r[valKey] || 0); });
    const order = Object.keys(tot).filter(c => c).sort((a, b) => tot[b] - tot[a]);
    const idx = {}; order.forEach((c, i) => { idx[c] = i; });
    return idx;
  }

  // Точечная диаграмма Кливленда (lollipop): ранжирование, прямые подписи значений, цвет — категория.
  function lollipop(id, rows, xtitle, opts) {
    opts = opts || {};
    if (!rows.length) return;
    const h = height(id, Math.max(240, 50 + 24 * rows.length));
    const y = rows.map(r => r.label);
    drawChart(id, [
      { type: 'scatter', mode: 'lines', x: rows.flatMap(r => [0, r.value, null]), y: rows.flatMap(r => [r.label, r.label, null]),
        line: { color: v('--grid'), width: 2 }, hoverinfo: 'skip', showlegend: false },
      { type: 'scatter', mode: 'markers+text', x: rows.map(r => r.value), y, text: rows.map(r => fmt(r.value)), textposition: 'middle right',
        textfont: { size: 10, color: v('--text-muted') }, cliponaxis: false, showlegend: false,
        marker: { size: 10, color: rows.map(r => r.color || v('--cat-1')), line: { width: 1, color: v('--surface') } },
        customdata: rows.map(r => r.hover || ''), hovertemplate: '<b>%{y}</b><br>%{customdata}<extra></extra>' }],
      { height: h, xaxis: { title: { text: xtitle }, rangemode: 'tozero', type: opts.log ? 'log' : 'linear' },
        yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } }, margin: { l: 20, r: 50, t: 10, b: 44 } });
  }

  // Treemap «категория → слова»: площадь — вхождения; цвет — категория.
  function treemap(id, rows) {
    rows = rows.filter(r => r.count > 0);
    if (!rows.length) return;
    const idx = catIndex(rows, 'category', 'count');
    const cats = [...new Set(rows.map(r => r.category || 'без категории'))];
    const ids = [], labels = [], parents = [], values = [], colors = [], hover = [];
    cats.forEach(c => {
      const sum = rows.filter(r => (r.category || 'без категории') === c).reduce((a, r) => a + r.count, 0);
      ids.push('c:' + c); labels.push(c); parents.push(''); values.push(sum); colors.push(cat(idx[c])); hover.push(sum + ' вхожд.');
    });
    rows.forEach(r => {
      const c = r.category || 'без категории';
      ids.push('w:' + c + ':' + r.word); labels.push(r.word); parents.push('c:' + c); values.push(r.count);
      colors.push(cat(idx[c])); hover.push(`${r.count} вхожд., ${fmt(r.ipm)} ipm`);
    });
    drawChart(id, [{ type: 'treemap', ids, labels, parents, values, branchvalues: 'total', customdata: hover,
      marker: { colors, line: { color: v('--surface'), width: 1 } }, textfont: { family: v('--font-sans'), color: '#fff' },
      tiling: { pad: 2 }, pathbar: { visible: false }, hovertemplate: '<b>%{label}</b><br>%{customdata}<extra></extra>' }],
      { height: height(id, 460), margin: { l: 0, r: 0, t: 0, b: 0 } });
  }

  // «Обзор»: группы × слова. До 6 групп — точки Кливленда (цвет — группа, линия от минимума к максимуму);
  // больше 6 — полоса точек (beeswarm-подобная: точка — группа, ромб — медиана); периоды — линии (прежний drawGroups).
  function groups(id, ov, metric, xtitle, log) {
    if (ov.dim === 'period') return drawGroups(id, ov, metric, xtitle, {}, { log });
    const words = ov.words.slice(0, 25), G = ov.groups, many = G.length > 6;
    const y = words.map(w => w.word);
    const h = height(id, Math.max(260, 70 + (many ? 34 : 28) * words.length));
    const traces = [{ type: 'scatter', mode: 'lines', hoverinfo: 'skip', showlegend: false, line: { color: v('--grid'), width: many ? 8 : 3 },
      x: words.flatMap(w => { const vals = w[metric].filter(x => x !== null); return [Math.min(...vals), Math.max(...vals), null]; }),
      y: words.flatMap(w => [w.word, w.word, null]) }];
    if (!many) {
      G.forEach((g, k) => traces.push({ type: 'scatter', mode: 'markers', name: g.label, x: words.map(w => w[metric][k]), y,
        marker: { size: 11, color: cat(k), line: { width: 1, color: v('--surface') } },
        hovertemplate: (g.full || g.label) + '<br>%{y}: %{x}<extra></extra>' }));
    } else {
      // вертикальный разброс точек внутри строки, чтобы совпадающие значения не слипались
      const pts = { x: [], y: [], text: [] };
      words.forEach((w, wi) => G.forEach((g, k) => {
        if (w[metric][k] === null) return;
        pts.x.push(w[metric][k]); pts.y.push(wi + ((k * 0.6180339) % 1 - 0.5) * 0.55); pts.text.push(`${g.full || g.label}<br>${w.word}: ${w[metric][k]}`);
      }));
      traces[0].y = words.flatMap((w, wi) => [wi, wi, null]);
      traces.push({ type: 'scatter', mode: 'markers', name: ov.dim_label, x: pts.x, y: pts.y, text: pts.text, showlegend: false,
        marker: { size: 7, color: v('--cat-1'), opacity: 0.55 }, hovertemplate: '%{text}<extra></extra>' });
      const med = words.map(w => { const s = w[metric].filter(x => x !== null).sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : null; });
      traces.push({ type: 'scatter', mode: 'markers', name: 'медиана', x: med, y: words.map((_, wi) => wi),
        marker: { symbol: 'diamond', size: 11, color: v('--cat-2'), line: { width: 1, color: v('--surface') } },
        hovertemplate: 'медиана: %{x}<extra></extra>' });
    }
    drawChart(id, traces, { height: h, xaxis: { title: { text: xtitle }, type: log ? 'log' : 'linear', rangemode: 'tozero' },
      yaxis: many ? { tickvals: words.map((_, wi) => wi), ticktext: y, autorange: 'reversed', automargin: true, zeroline: false, tickfont: { size: 11 } }
        : { automargin: true, autorange: 'reversed', tickfont: { size: 11 } },
      legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom' }, margin: { l: 20, r: 20, t: 40, b: 44 } });
  }

  // Тепловая карта отклонений: для каждого слова — z-оценка по группам (красное — выше среднего, синее — ниже);
  // в подсказке — само значение. Строки — слова, столбцы — группы.
  function devHeat(id, ov, metric) {
    const words = ov.words, G = ov.groups;
    const z = words.map(w => {
      const vals = w[metric].map(x => x || 0), mu = vals.reduce((a, b) => a + b, 0) / vals.length;
      const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mu) ** 2, 0) / vals.length) || 1;
      return vals.map(x => Math.round((x - mu) / sd * 100) / 100);
    });
    const vertical = G.length > 6 || G.some(g => (g.label || '').length > 18);  // много групп или длинные названия — группы по вертикали
    const xs = vertical ? words.map(w => w.word) : G.map(g => g.label), ys = vertical ? G.map(g => g.label) : words.map(w => w.word);
    const zz = vertical ? G.map((_, k) => z.map(r => r[k])) : z;
    const raw = vertical ? G.map((_, k) => words.map(w => w[metric][k])) : words.map(w => w[metric]);
    const h = height(id, Math.max(220, 110 + 24 * ys.length));
    drawChart(id, [{ type: 'heatmap', x: xs, y: ys, z: zz, customdata: raw, zmid: 0, zmin: -2.5, zmax: 2.5, colorscale: diverging(),
      xgap: 1, ygap: 1, colorbar: { title: { text: 'z' }, thickness: 10 },
      hovertemplate: '%{y} — %{x}<br>значение: %{customdata}<br>отклонение z = %{z}<extra></extra>' }],
      { height: h, xaxis: { side: 'top', automargin: true, tickangle: xs.length > 6 ? -35 : 0, tickfont: { size: 11 } },
        yaxis: { automargin: true, autorange: 'reversed', tickfont: { size: 11 } }, margin: { l: 20, r: 20, t: 70, b: 10 } });
  }

  // Сравнение корпусов в стиле Scattertext: x — ipm в B, y — ipm в A (лог. шкалы), диагональ — одинаково часто;
  // цвет — где значимо чаще (A — тёплый, B — холодный), подписаны самые характерные.
  function scatterCompare(id, rows, nameA, nameB) {
    rows = rows.filter(r => r.count_a + r.count_b > 0);
    if (!rows.length) return;
    const pa = cssVar('--div-4'), pb = cssVar('--div-0'), mid = cssVar('--inactive');
    const vals = rows.flatMap(r => [r.ipm_a, r.ipm_b]).filter(x => x > 0);
    const floor = Math.max(Math.min(...vals) / 2, 0.01), hi = Math.max(...vals) * 1.3;
    // подписи: самые значимые, но у края (слово есть только в одном корпусе) — не больше 4, иначе подписи слипаются
    const ranked = rows.filter(r => r.significant).sort((a, b) => b.ll - a.ll);
    const edge = ranked.filter(r => !r.ipm_a || !r.ipm_b).slice(0, 4), inner = ranked.filter(r => r.ipm_a && r.ipm_b).slice(0, 12);
    const sig = [...edge, ...inner].map(r => r.word);
    // нули — у края оси с небольшим разбросом (детерминированным), чтобы точки не сливались в одну
    const spread = (val, i) => val > 0 ? val : floor * Math.pow(10, 0.35 * ((i * 0.6180339) % 1));
    drawChart(id, [{ type: 'scatter', mode: 'markers+text', x: rows.map((r, i) => spread(r.ipm_b, i)), y: rows.map((r, i) => spread(r.ipm_a, i + 7)),
      text: rows.map(r => sig.includes(r.word) ? r.word : ''), textposition: 'top center', textfont: { size: 10 },
      marker: { size: rows.map(r => 6 + 2 * Math.log10(1 + r.count_a + r.count_b)), opacity: 0.85,
                color: rows.map(r => !r.significant ? mid : (r.log_ratio > 0 ? pa : pb)), line: { width: 0.5, color: v('--surface') } },
      customdata: rows.map(r => [r.word, r.count_a, r.count_b, r.ll, r.log_ratio]),
      hovertemplate: '<b>%{customdata[0]}</b><br>A: %{y} ipm (%{customdata[1]})<br>B: %{x} ipm (%{customdata[2]})<br>G² %{customdata[3]}, log-ratio %{customdata[4]}<extra></extra>' }],
      { height: height(id, 480), xaxis: { type: 'log', title: { text: `ipm в B — ${nameB}` }, range: [Math.log10(floor), Math.log10(hi)] },
        yaxis: { type: 'log', title: { text: `ipm в A — ${nameA}` }, range: [Math.log10(floor), Math.log10(hi)] },
        shapes: [{ type: 'line', x0: floor, y0: floor, x1: hi, y1: hi, line: { color: v('--border'), dash: 'dot' } }],
        annotations: [{ text: 'чаще в A', xref: 'paper', yref: 'paper', x: 0.12, y: 0.98, showarrow: false, xanchor: 'left', font: { color: pa, size: 12 } },
                      { text: 'чаще в B', xref: 'paper', yref: 'paper', x: 0.98, y: 0.12, showarrow: false, xanchor: 'right', font: { color: pb, size: 12 } },
                      { text: 'нет в B', xref: 'x', yref: 'paper', x: Math.log10(floor * 1.5), y: 1, showarrow: false, yanchor: 'bottom', font: { color: v('--text-muted'), size: 10 } },
                      { text: 'нет в A', xref: 'paper', yref: 'y', x: 1, y: Math.log10(floor * 1.5), showarrow: false, xanchor: 'right', font: { color: v('--text-muted'), size: 10 } }],
        margin: { l: 60, r: 20, t: 10, b: 50 } });
  }

  // Полосы 100 %: доли типов предложений (со словами словаря / все предложения).
  function bars100(id, series, cats) {
    const last = series[series.length - 1].values;
    cats = [...cats].sort((a, b) => (last[b] || 0) - (last[a] || 0));  // крупные доли — первыми
    const traces = cats.map((c, k) => ({ type: 'bar', orientation: 'h', name: c, y: series.map(s => s.name),
      x: series.map(s => s.values[c] || 0), marker: { color: cat(k) }, hovertemplate: '%{y}<br>' + c + ': %{x} %<extra></extra>' }));
    drawChart(id, traces, { barmode: 'stack', height: height(id, 120 + 44 * series.length), xaxis: { range: [0, 100], ticksuffix: ' %' },
      yaxis: { automargin: true, autorange: 'reversed' }, legend: { orientation: 'h', x: 0, y: 1.05, yanchor: 'bottom', traceorder: 'normal' },
      margin: { l: 20, r: 20, t: 40, b: 30 }, bargap: 0.35 });
  }

  // Распределение длины предложений — две линии вместо сгруппированных столбцов.
  function lengthLines(id, h, a, nameH) {
    const keys = [...new Set([...Object.keys(h), ...Object.keys(a)].map(Number))].sort((x, y) => x - y);
    const lab = keys.map(k => k >= 100 ? '100+' : `${k}–${k + 4}`);
    drawChart(id, [
      { type: 'scatter', mode: 'lines+markers', name: nameH, x: lab, y: keys.map(k => h[k] || 0), line: { color: v('--cat-1'), width: 2.5 }, marker: { size: 5 } },
      { type: 'scatter', mode: 'lines', name: 'все предложения', x: lab, y: keys.map(k => a[k] || 0), line: { color: v('--inactive'), width: 2, dash: 'dot' } }],
      { xaxis: { title: { text: 'длина предложения, слов' } }, yaxis: { title: { text: '% предложений' }, rangemode: 'tozero' },
        legend: { orientation: 'h', x: 0, y: 1.08 }, margin: { l: 60, r: 20, t: 30, b: 60 } });
  }

  return { cat, catIndex, lollipop, treemap, groups, devHeat, scatterCompare, bars100, lengthLines };
})();
