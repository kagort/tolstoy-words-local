# Бриф для дизайнера: визуализация Quotverba

Промты для генератора изображений (GPT Image): несколько мудбордов, затем макеты дашборда и отдельных графиков.
План, ради которого это делается, — «Волна «Визуализация»» в [`PROGRESS.md`](PROGRESS.md).

**Как пользоваться.** Промты написаны по-английски: так генератор точнее понимает дизайнерские термины. Подписи на
графиках — по-русски, в кавычках; генератор иногда искажает кириллицу. Если подписи получились нечитаемыми, попросите
перерисовать с более короткими подписями или считайте их условными: важны композиция, цвет, типографика и форма
графиков. Генерируйте по одному изображению на промт, формат 1536×1024 (альбомный), для телефона — 1024×1536.
Сначала промт 0 (контекст) — вставьте его перед каждым следующим или в начало диалога.

---

## Промт 0. Контекст (вставлять первым)

```
You are a senior data-visualization and product designer. Design for "Quotverba" — a web platform for corpus
linguistics of Russian texts used by philologists, historians and political-discourse researchers.

What the product does: users upload Russian texts (classic literature — Tolstoy, Chekhov, Sholokhov; historical
sources of the 18th–19th c.; political speeches), build a corpus with metadata (author, year, genre), upload a
dictionary of words grouped into categories (e.g. "СОСЛОВИЯ", "УГРОЗА", "ЗАПАХИ"), and analyze how the dictionary
lives in the corpus: frequencies per million words (ipm), dispersion, syntactic roles, collocations, keywords,
characters (named entities), places on a map, authorship (stylometry). Results are exported as a self-contained
HTML dashboard that works offline.

Users: academic researchers, not data scientists. They need clarity, trust and the path "number → quote from the text".
The interface language is Russian (Cyrillic). Charts are implemented with Plotly.js, so designs must be buildable with
standard chart types (scatter, bar, heatmap, scatterpolar/radar, treemap, sunburst, box/violin, small-multiple grids,
scattergeo maps, network drawn as scatter + lines). No 3D, no decorative effects that hide data.

Current UI (to evolve, not to copy): system sans-serif, light gray background #f7f7f5, white panels with 1px borders
#d8dbe0, accent blue #2f5fb3, muted text #656d76; dark theme exists. Brand idea for the logo: a fish whose body is a
Russian quotation mark «» (guillemet) — "quot" = quotation; wordmark "Quotverba".

Design principles to follow: data-ink first; direct labels instead of legends; color only for meaning (diverging scale
for "more / less than usual", max 8 categorical colors, everything else gray with one highlight); sort by value or
similarity, never alphabetically; "top N + others"; same scale on neighboring charts; colorblind-safe palettes;
light and dark themes; readable at 1280 px desktop and 390 px phone.
```

---

## Промт 1. Четыре мудборда (одно изображение на направление)

Сгенерируйте четыре изображения, подставляя направление A, B, C или D.

```
Create a moodboard (one 1536x1024 board) for Quotverba, design direction: <DIRECTION>.
The board must contain, arranged as a clean grid with small captions:
1) color palette: 8 categorical swatches + a diverging scale (cool ← neutral → warm) + 3 neutrals, each with HEX codes;
   show it on both a light and a dark background;
2) typography: heading font, body font, numeric/tabular font, with a sample "Частоты по категориям словаря" and the
   numbers "12 345 ipm · 4,6 %";
3) four small chart swatches in this style: a heatmap of deviations (z-scores), a lollipop ranking, a radar chart
   comparing 3 authors over 8 categories, a beeswarm of dots;
4) UI fragments: a filter panel with chips "Корпус", "Авторы", "Годы", "Словарь"; a KPI tile "58 текстов"; a tooltip
   showing a quote with the keyword highlighted;
5) the logo idea: a fish formed by the guillemet «», plus the wordmark "Quotverba", small and large;
6) 3–4 texture/mood references (photos or illustrations) that explain the feeling of the direction.
Keep it elegant and practical; this is a working research tool, not a poster.
```

Направления:
- **A — «Академическая типографика».** Spirit of a scholarly journal and Edward Tufte: ink on warm white paper, serif
  headings (e.g. Literata/PT Serif), restrained palette of deep blue, ochre, oxblood and gray; hairline rules; lots of air.
- **B — «Архив и рукопись».** For historical sources: subtle paper tone, sepia and iron-gall ink colors, manuscript
  margins and folio marks ("л. 2 об.") as a design motif, but charts stay crisp and modern; no fake grunge.
- **C — «Обсерватория».** Calm dark-first analytics: deep navy/graphite background, luminous but not neon accents
  (teal, amber, coral, violet), precise grid, monospaced numbers; must also have a light variant.
- **D — «Литературная карта».** Cartographic: muted map colors (sage, sand, slate, terracotta), contour-line textures,
  compass/legend motifs; charts feel like thematic maps; strong for geography and networks.

---

## Промт 2. Дашборд v2 целиком (после выбора направления)

```
Using design direction <A/B/C/D> from the moodboard, design a full desktop page (1536x1024, show the top part; then
generate a second image with the lower part) of the Quotverba HTML dashboard "Корпус «Русская проза XIX века» ×
словарь «Запахи»". Layout: left sticky table of contents; header with title, selection description ("58 текстов,
12 авторов, 1830–1910") and buttons "Скачать HTML", "PDF". Sections, each with a one-sentence takeaway written above
the chart:
1. "Покрытие": KPI tiles (найдено 214 из 260 слов; 1,8 % текста; 46 «мёртвых» слов) + a small treemap of categories.
2. "Ядро и периферия словаря": scatter, x = frequency (log), y = dispersion DP 0–1, labeled outlier words
   ("запах", "аромат", "вонь", "дым", "ладан").
3. "Профиль категорий": small-multiple radar charts, one per author (12), same scale, 8 axes.
4. "Характерные слова авторов": diverging lollipop chart "чаще ← → реже, чем у остальных" for the selected author.
5. "Время": small multiples, one sparkline per category, 1830–1910.
6. "Синтаксис": 100 % stacked horizontal bars of syntactic roles (подлежащее, дополнение, определение, обстоятельство)
   per category.
7. "Сеть слов": force-directed network of co-occurring dictionary words, nodes colored by category.
8. "Персонажи и места": heatmap "персонаж × категория" and a small map with places colored by dominant category.
9. "Качество данных": compact checklist with warning icons.
Use real-looking Russian labels and plausible numbers. Make hierarchy obvious: takeaway text > chart > details.
```

---

## Промты 3. Отдельные графики (для каждого — светлая и тёмная версия)

Добавляйте в конце каждого: `Style: direction <A/B/C/D>. Light theme.` и затем то же с `Dark theme.`

**3.1. Тепловая карта отклонений**
```
Design a heatmap of z-score deviations: rows = 60 Russian texts (short labels like "Толстой Л. Н. · Война и мир"),
columns = 10 dictionary categories ("ЗАПАХИ ЦВЕТОВ", "ДЫМ И ГАРЬ", "ЕДА", "ТЕЛО"...). Diverging scale: blue = less than
corpus average, white = average, red = more. Rows ordered by similarity with a thin dendrogram on the left; author
groups marked by a colored bar. Hover card example: "Чехов А. П. · Степь — ДЫМ И ГАРЬ: +2,4σ (310 ipm при среднем 95)".
Make 60 rows readable: small but legible labels, row height 14–16 px, sticky column headers.
```

**3.2. Beeswarm: одно слово по сотням текстов**
```
Design a beeswarm chart: each dot = one text (300 texts), x axis = ipm of the word "аромат" (log scale), dots grouped
in rows by genre ("роман", "повесть", "рассказ", "очерк"), color = period. Outliers labeled directly
("Бунин · Антоновские яблоки"). Add a thin box-and-whisker overlay per row. Title: "«аромат» в 300 текстах".
```

**3.3. Сравнение двух корпусов (Scattertext)**
```
Design a Scattertext-style scatter comparing two corpora: x = frequency in "Толстой" (log), y = frequency in
"Достоевский" (log), diagonal line of parity, words far from the diagonal labeled; upper-left region tinted for words
typical of Dostoevsky, lower-right for Tolstoy; color = dictionary category; side lists "Характерно для Толстого" /
"Характерно для Достоевского" with top 10 words. Avoid label overlap.
```

**3.4. Ранжирование (lollipop) и расхождение**
```
Design two variants: (a) a sorted lollipop chart of top 25 dictionary words by ipm with direct numeric labels;
(b) a diverging lollipop "чаще / реже, чем в остальном корпусе" with log-ratio on x, significance marked by filled vs
hollow dots, words sorted by effect size.
```

**3.5. Радар-профили**
```
Design (a) a radar chart comparing 3 authors over 8 categories with normalized values (share of category in the
author's dictionary hits), consistent axis order grouped by meaning, translucent fills, direct labels;
(b) a grid of 12 small radar charts (one per author) with identical scales and a highlighted "corpus average" outline
in gray in each. Explain normalization in a one-line caption.
```

**3.6. Состав словаря**
```
Design a treemap and an alternative sunburst of a dictionary: category → words, area = frequency, color = how
characteristic the word is for the current selection (diverging). Show breadcrumbs for drill-down and a hover card
with "ipm", "текстов", "пример цитаты".
```

**3.7. Время и место в тексте**
```
Design (a) small multiples: 10 mini line charts (one per category) on the same y scale, x = years 1830–1910, with the
category name and a tiny trend arrow; (b) a ridgeline chart showing where in one novel each category is dense
(x = position in text 0–100 %, one ridge per category), with peaks annotated by chapter numbers.
```

**3.8. Сеть слов и персонажей**
```
Design a force-directed network of dictionary words that co-occur in the same sentence: node size = frequency,
color = category, edge width = co-occurrence count; community clusters gently outlined; labels for the top 30 nodes.
Second variant: bipartite view "персонажи ↔ категории" (characters on the left, categories on the right).
```

**3.9. Карта мест**
```
Design a map panel for places mentioned in a corpus (European Russia, the Caucasus, Western Europe): proportional
circles = mentions, color = dominant dictionary category near the place; a toggle "места / страны" where countries are
shaded by total mentions; a small table of top places; note "границы современные". Muted basemap, no satellite.
```

**3.10. Мобильная версия**
```
Design the same dashboard section "Профиль категорий" for a 390 px phone: radar becomes a ranked lollipop list,
heatmap becomes a scrollable list with inline color chips; sticky filter chip bar at top.
```

---

## Что вернуть от дизайнера

1. Четыре мудборда (A–D) — выбрать одно направление (или смешать: например, A для отчётов и C для тёмной темы).
2. Дашборд v2 в выбранном направлении (2 изображения: верх и низ страницы).
3. Макеты графиков 3.1–3.9 в светлой и тёмной теме; 3.10 — мобильный вариант.
4. Итог в текстовом виде, чтобы перенести в код: палитры с HEX (категориальная, двухцветная шкала, нейтральные — для
   обеих тем), шрифты, размеры подписей, толщины линий, радиусы, отступы.

Из этого делается: токены цветов и шрифтов в `app/static/app.css`, шаблоны оформления графиков в `app/static/app.js`
(общая функция оформления для Plotly), макет дашборда в `app/templates/_dashboard_body.html`.
