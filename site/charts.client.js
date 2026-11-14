// Draws the [data-chart] figures at their real width, so text stays readable on phones.
// The tables on the page hold every value, so the charts only add to them.
(function () {
  document.documentElement.classList.add("js");
  const NS = "http://www.w3.org/2000/svg";
  const HEIGHT = 280;

  function node(parent, tag, attrs, text) {
    const el = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text !== undefined) el.textContent = text;
    parent.appendChild(el);
    return el;
  }

  const full = (v) => v.toLocaleString("en-US");
  const compact = (v) => (v >= 1000 ? `${v / 1000}K` : String(v));

  // Clean y-axis ticks: about four steps of 1, 2, 2.5 or 5 times a power of ten.
  function ticks(max) {
    const raw = max / 4;
    const power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw);
    const out = [];
    for (let v = 0; v < max + step; v += step) out.push(v);
    return out;
  }

  function showTip(tip, figure, title, rows, x, y) {
    tip.replaceChildren();
    const heading = document.createElement("div");
    heading.className = "tip-title";
    heading.textContent = title;
    tip.appendChild(heading);
    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "tip-row";
      const key = document.createElement("i");
      key.style.background = `var(--series-${row.slot})`;
      const value = document.createElement("strong");
      value.textContent = row.value;
      const name = document.createElement("span");
      name.textContent = row.name;
      line.append(key, value, name);
      tip.appendChild(line);
    }
    tip.hidden = false;
    const left = Math.min(Math.max(0, x + 12), figure.clientWidth - tip.offsetWidth);
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(0, y - tip.offsetHeight - 8)}px`;
  }

  function hoverable(target, show, hide) {
    target.setAttribute("tabindex", "0");
    target.addEventListener("pointerenter", show);
    target.addEventListener("pointermove", show);
    target.addEventListener("focus", show);
    target.addEventListener("pointerleave", hide);
    target.addEventListener("blur", hide);
  }

  function axes(svg, area, top, labels, xOf) {
    for (const value of ticks(top)) {
      const y = area.y(value);
      node(svg, "line", { x1: area.left, x2: area.right, y1: y, y2: y, class: value ? "grid" : "axis" });
      node(svg, "text", { x: area.left - 8, y: y + 4, "text-anchor": "end", class: "tick" }, compact(value));
    }
    labels.forEach((label, i) => {
      node(svg, "text", { x: xOf(i), y: area.bottom + 20, "text-anchor": "middle", class: "tick" }, label);
    });
  }

  function drawLine(svg, spec, width, tip, figure) {
    const area = plot(spec, width, 72);
    const step = (area.right - area.left) / (spec.labels.length - 1);
    const xOf = (i) => area.left + i * step;
    const top = Math.max(...ticks(maxOf(spec)));
    area.y = scale(area, top);
    axes(svg, area, top, spec.labels, xOf);
    const hair = node(svg, "line", { y1: area.top, y2: area.bottom, class: "crosshair", visibility: "hidden" });

    const last = spec.labels.length - 1;
    // Series names sit at the right end of each line; push them apart when lines end close together.
    const ends = spec.series.map((series, s) => ({ s, y: area.y(series.values[last]) + 4 }))
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 14);
    const peaks = []; // value labels already drawn, so close ones are skipped (the tooltip has every value)

    spec.series.forEach((series, s) => {
      const slot = s + 1;
      const points = series.values.map((v, i) => `${xOf(i)},${area.y(v)}`).join(" ");
      node(svg, "polyline", { points, class: "series-line", style: `stroke:var(--series-${slot})` });
      series.values.forEach((v, i) => {
        node(svg, "circle", { cx: xOf(i), cy: area.y(v), r: 4, class: "dot", style: `fill:var(--series-${slot})` });
      });
      node(svg, "text", { x: xOf(last) + 10, y: ends.find((e) => e.s === s).y, class: "label" }, series.name);
    });
    // Peak values, highest series first: when two would collide, the one sitting on top keeps its label.
    spec.series.map((series) => Math.max(...series.values)).map((v, s) => ({ v, s }))
      .sort((a, b) => b.v - a.v)
      .forEach(({ v, s }) => {
        const at = { x: xOf(spec.series[s].values.indexOf(v)), y: area.y(v) - 12 };
        if (peaks.every((p) => Math.abs(p.x - at.x) > 40 || Math.abs(p.y - at.y) > 16)) {
          peaks.push(at);
          node(svg, "text", { x: at.x, y: at.y, "text-anchor": "middle", class: "value" }, full(v));
        }
      });

    spec.labels.forEach((label, i) => {
      const band = node(svg, "rect", {
        x: Math.max(0, xOf(i) - step / 2), width: step, y: area.top, height: area.bottom - area.top, class: "hit",
        "aria-label": `${label} ${spec.xName}: ${spec.series.map((s) => `${s.name} ${full(s.values[i])}`).join(", ")}`,
      });
      hoverable(band, () => {
        hair.setAttribute("x1", xOf(i));
        hair.setAttribute("x2", xOf(i));
        hair.setAttribute("visibility", "visible");
        const rows = spec.series.map((s, n) => ({ slot: n + 1, name: s.name, value: full(s.values[i]) }));
        showTip(tip, figure, `${label} ${spec.xName}`, rows, xOf(i), area.top + 60);
      }, () => {
        hair.setAttribute("visibility", "hidden");
        tip.hidden = true;
      });
    });
  }

  // Flat at the baseline, 4px rounded at the data end.
  function column(x, y, w, bottom) {
    const r = Math.min(4, bottom - y, w / 2);
    return `M${x},${bottom}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${bottom}Z`;
  }

  function drawBars(svg, spec, width, tip, figure) {
    const area = plot(spec, width, 8);
    const band = (area.right - area.left) / spec.labels.length;
    const xOf = (i) => area.left + band * (i + 0.5);
    const top = Math.max(...ticks(maxOf(spec)));
    area.y = scale(area, top);
    const w = Math.min(24, (band * 0.8 - 2) / spec.series.length);
    axes(svg, area, top, spec.labels, xOf);

    spec.labels.forEach((label, i) => {
      const start = xOf(i) - (w * spec.series.length + 2 * (spec.series.length - 1)) / 2;
      spec.series.forEach((series, s) => {
        const value = series.values[i];
        const x = start + s * (w + 2);
        const y = area.y(value);
        const mark = node(svg, "path", { d: column(x, y, w, area.bottom), class: "bar", style: `fill:var(--series-${s + 1})` });
        if (series.labelled) {
          node(svg, "text", { x: x + w / 2, y: y - 6, "text-anchor": "middle", class: "value" }, full(value));
        }
        const hit = node(svg, "rect", {
          x: x - 1, width: w + 2, y: area.top, height: area.bottom - area.top, class: "hit",
          "aria-label": `${label}, ${series.name}: ${full(value)} ${spec.unit}`,
        });
        hoverable(hit, () => {
          mark.classList.add("lifted");
          showTip(tip, figure, label, [{ slot: s + 1, name: series.name, value: `${full(value)} ${spec.unit}` }], x, y);
        }, () => {
          mark.classList.remove("lifted");
          tip.hidden = true;
        });
      });
    });
  }

  const maxOf = (spec) => Math.max(...spec.series.flatMap((s) => s.values));
  const plot = (spec, width, right) => ({ left: 44, right: width - right, top: 24, bottom: HEIGHT - 30 });
  const scale = (area, top) => (v) => area.bottom - (v / top) * (area.bottom - area.top);

  function render(figure, spec) {
    const width = figure.clientWidth;
    const holder = figure.querySelector(".plot");
    holder.replaceChildren();
    const svg = node(holder, "svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}` });
    const tip = figure.querySelector(".tip");
    tip.hidden = true;
    (spec.type === "line" ? drawLine : drawBars)(svg, spec, width, tip, figure);
  }

  for (const figure of document.querySelectorAll("[data-chart]")) {
    const spec = JSON.parse(figure.dataset.chart);
    const legend = figure.querySelector(".legend");
    spec.series.forEach((series, s) => {
      const item = document.createElement("span");
      const key = document.createElement("i");
      key.className = spec.type === "line" ? "key-line" : "key-box";
      key.style.background = `var(--series-${s + 1})`;
      item.append(key, document.createTextNode(series.name));
      legend.appendChild(item);
    });
    let lastWidth = 0;
    new ResizeObserver(() => {
      if (figure.clientWidth !== lastWidth) {
        lastWidth = figure.clientWidth;
        render(figure, spec);
      }
    }).observe(figure);
  }
})();
