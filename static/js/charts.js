/* ==========================================================================
   charts.js - tiny dependency-free SVG chart library
   Supported types: donut, bars, grouped-bars, radar, line
   ========================================================================== */

(function (global) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const PALETTE = [
    "#22d37a", "#38bdf8", "#a78bfa", "#fb7185",
    "#fbbf24", "#a3e635", "#f472b6", "#60a5fa",
  ];

  function create(tag, attrs = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function parseData(element) {
    const raw = element.dataset.payload || element.dataset.items;
    if (!raw) return [];
    try {
      return JSON.parse(raw);
    } catch (error) {
      console.warn("Невалидни данни за диаграма", error);
      return [];
    }
  }

  function formatValue(value, decimals = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) return "-";
    return Number(value).toFixed(decimals).replace(/\.0+$/, "");
  }

  function emptyState(element, message) {
    element.innerHTML = `<p class="result-count" style="padding:1.5rem;text-align:center">${message}</p>`;
  }

  /* --------------------------------------------------------------- donut --- */
  function renderDonut(element, items) {
    if (!items.length) return emptyState(element, "Няма данни за показване.");

    const total = items.reduce((sum, item) => sum + Number(item.value || 0), 0);
    const size = 220;
    const radius = 84;
    const stroke = 30;
    const center = size / 2;
    const circumference = 2 * Math.PI * radius;
    let offset = 0;

    const svg = create("svg", {
      class: "chart",
      viewBox: `0 0 ${size} ${size}`,
      role: "img",
      "aria-label": "Пръстенова диаграма",
    });

    items.forEach((item, index) => {
      const value = Number(item.value || 0);
      const share = total ? value / total : 0;
      const color = item.color || PALETTE[index % PALETTE.length];
      const circle = create("circle", {
        cx: center,
        cy: center,
        r: radius,
        fill: "none",
        stroke: color,
        "stroke-width": stroke,
        "stroke-dasharray": `${(share * circumference).toFixed(2)} ${circumference.toFixed(2)}`,
        "stroke-dashoffset": (-offset * circumference).toFixed(2),
        transform: `rotate(-90 ${center} ${center})`,
      });
      circle.appendChild(create("title", {})).textContent =
        `${item.label}: ${value} (${Math.round(share * 100)}%)`;
      svg.appendChild(circle);
      offset += share;
    });

    const totalText = create("text", {
      x: center, y: center - 2, "text-anchor": "middle",
      fill: "#e9eff8", "font-size": "30", "font-weight": "700",
    });
    totalText.textContent = total;
    svg.appendChild(totalText);

    const totalLabel = create("text", {
      x: center, y: center + 20, "text-anchor": "middle",
      fill: "#8494ab", "font-size": "11",
    });
    totalLabel.textContent = "общо";
    svg.appendChild(totalLabel);

    element.innerHTML = "";
    element.appendChild(svg);

    const legend = document.createElement("div");
    legend.className = "legend";
    items.forEach((item, index) => {
      const color = item.color || PALETTE[index % PALETTE.length];
      const entry = document.createElement("span");
      entry.innerHTML = `<i style="background:${color}"></i>${item.label} · ${item.value}`;
      legend.appendChild(entry);
    });
    element.appendChild(legend);
  }

  /* ---------------------------------------------------------------- bars --- */
  function renderBars(element, items, unit) {
    if (!items.length) return emptyState(element, "Няма данни за показване.");

    const width = 640;
    const height = 250;
    const padding = { top: 26, right: 16, bottom: 42, left: 40 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxValue = Math.max(...items.map((item) => Number(item.value || 0)), 1);
    const slot = plotWidth / items.length;
    const barWidth = Math.min(52, slot * 0.6);

    const svg = create("svg", {
      class: "chart",
      viewBox: `0 0 ${width} ${height}`,
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": "Столбовидна диаграма",
    });

    const ticks = 4;
    for (let index = 0; index <= ticks; index += 1) {
      const value = (maxValue / ticks) * index;
      const y = padding.top + plotHeight - (value / maxValue) * plotHeight;
      svg.appendChild(create("line", {
        x1: padding.left, y1: y, x2: width - padding.right, y2: y, class: "grid-line",
      }));
      const label = create("text", {
        x: padding.left - 8, y: y + 4, "text-anchor": "end", class: "axis-label",
      });
      label.textContent = formatValue(value, 0);
      svg.appendChild(label);
    }

    items.forEach((item, index) => {
      const value = Number(item.value || 0);
      const barHeight = (value / maxValue) * plotHeight;
      const x = padding.left + slot * index + (slot - barWidth) / 2;
      const y = padding.top + plotHeight - barHeight;
      const color = item.color || PALETTE[index % PALETTE.length];

      const bar = create("rect", {
        x, y, width: barWidth, height: barHeight, fill: color, class: "bar",
      });
      bar.appendChild(create("title", {})).textContent =
        `${item.label}: ${formatValue(value)}${unit ? " " + unit : ""}`;
      svg.appendChild(bar);

      const valueLabel = create("text", {
        x: x + barWidth / 2, y: y - 7, "text-anchor": "middle", class: "value-label",
      });
      valueLabel.textContent = formatValue(value);
      svg.appendChild(valueLabel);

      const nameLabel = create("text", {
        x: x + barWidth / 2, y: height - padding.bottom + 18, "text-anchor": "middle", class: "axis-label",
      });
      nameLabel.textContent = item.label.length > 12 ? `${item.label.slice(0, 11)}…` : item.label;
      svg.appendChild(nameLabel);
    });

    element.innerHTML = "";
    element.appendChild(svg);
  }

  /* -------------------------------------------------------- grouped bars --- */
  function renderGroupedBars(element, items) {
    if (!items.length) return emptyState(element, "Няма данни за показване.");

    const seriesKeys = Object.keys(items[0]).filter((key) => key !== "label");
    const width = 640;
    const height = 250;
    const padding = { top: 26, right: 16, bottom: 42, left: 40 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxValue = Math.max(
      1,
      ...items.flatMap((item) => seriesKeys.map((key) => Number(item[key] || 0))),
    );
    const slot = plotWidth / items.length;
    const groupWidth = Math.min(slot * 0.72, 90);
    const barWidth = groupWidth / seriesKeys.length;

    const svg = create("svg", {
      class: "chart",
      viewBox: `0 0 ${width} ${height}`,
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": "Групирана столбовидна диаграма",
    });

    const ticks = 4;
    for (let index = 0; index <= ticks; index += 1) {
      const value = (maxValue / ticks) * index;
      const y = padding.top + plotHeight - (value / maxValue) * plotHeight;
      svg.appendChild(create("line", {
        x1: padding.left, y1: y, x2: width - padding.right, y2: y, class: "grid-line",
      }));
      const label = create("text", {
        x: padding.left - 8, y: y + 4, "text-anchor": "end", class: "axis-label",
      });
      label.textContent = formatValue(value, 0);
      svg.appendChild(label);
    }

    items.forEach((item, index) => {
      const groupX = padding.left + slot * index + (slot - groupWidth) / 2;
      seriesKeys.forEach((key, seriesIndex) => {
        const value = Number(item[key] || 0);
        const barHeight = (value / maxValue) * plotHeight;
        const x = groupX + seriesIndex * barWidth;
        const y = padding.top + plotHeight - barHeight;
        const color = PALETTE[seriesIndex % PALETTE.length];
        const bar = create("rect", {
          x: x + 1, y, width: Math.max(barWidth - 3, 4), height: barHeight, fill: color, class: "bar",
        });
        bar.appendChild(create("title", {})).textContent = `${item.label} · ${key}: ${value}`;
        svg.appendChild(bar);
      });

      const label = create("text", {
        x: groupX + groupWidth / 2, y: height - padding.bottom + 18, "text-anchor": "middle", class: "axis-label",
      });
      label.textContent = item.label;
      svg.appendChild(label);
    });

    element.innerHTML = "";
    element.appendChild(svg);

    const legend = document.createElement("div");
    legend.className = "legend";
    seriesKeys.forEach((key, index) => {
      const entry = document.createElement("span");
      const friendly = key === "goals" ? "Голове" : key === "assists" ? "Асисти" : key;
      entry.innerHTML = `<i style="background:${PALETTE[index % PALETTE.length]}"></i>${friendly}`;
      legend.appendChild(entry);
    });
    element.appendChild(legend);
  }

  /* ---------------------------------------------------------------- line --- */
  function renderLine(element, items, seriesKeys) {
    if (!items.length) return emptyState(element, "Няма данни за показване.");

    const keys = seriesKeys || Object.keys(items[0]).filter((key) => key !== "label");
    const width = 640;
    const height = 240;
    const padding = { top: 24, right: 20, bottom: 38, left: 38 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxValue = Math.max(
      1,
      ...items.flatMap((item) => keys.map((key) => Number(item[key] || 0))),
    );

    const svg = create("svg", {
      class: "chart",
      viewBox: `0 0 ${width} ${height}`,
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": "Линейна диаграма",
    });

    const ticks = 4;
    for (let index = 0; index <= ticks; index += 1) {
      const value = (maxValue / ticks) * index;
      const y = padding.top + plotHeight - (value / maxValue) * plotHeight;
      svg.appendChild(create("line", {
        x1: padding.left, y1: y, x2: width - padding.right, y2: y, class: "grid-line",
      }));
      const label = create("text", {
        x: padding.left - 8, y: y + 4, "text-anchor": "end", class: "axis-label",
      });
      label.textContent = formatValue(value, 0);
      svg.appendChild(label);
    }

    const step = items.length > 1 ? plotWidth / (items.length - 1) : 0;

    keys.forEach((key, seriesIndex) => {
      const color = PALETTE[seriesIndex % PALETTE.length];
      const points = items.map((item, index) => {
        const x = padding.left + step * index + (items.length === 1 ? plotWidth / 2 : 0);
        const y = padding.top + plotHeight - (Number(item[key] || 0) / maxValue) * plotHeight;
        return [x, y];
      });

      const areaPath = points.length
        ? `M ${points[0][0]} ${padding.top + plotHeight} L ${points
            .map((point) => point.join(" "))
            .join(" L ")} L ${points[points.length - 1][0]} ${padding.top + plotHeight} Z`
        : "";
      if (areaPath) {
        svg.appendChild(create("path", { d: areaPath, fill: color, class: "series-area" }));
      }

      const line = create("path", {
        d: `M ${points.map((point) => point.join(" ")).join(" L ")}`,
        stroke: color,
        class: "series-line",
      });
      svg.appendChild(line);

      points.forEach((point, index) => {
        const dot = create("circle", { cx: point[0], cy: point[1], r: 4, fill: color });
        dot.appendChild(create("title", {})).textContent =
          `${items[index].label} · ${key}: ${items[index][key]}`;
        svg.appendChild(dot);
      });
    });

    items.forEach((item, index) => {
      const label = create("text", {
        x: padding.left + step * index + (items.length === 1 ? plotWidth / 2 : 0),
        y: height - padding.bottom + 18,
        "text-anchor": "middle",
        class: "axis-label",
      });
      label.textContent = item.label;
      svg.appendChild(label);
    });

    element.innerHTML = "";
    element.appendChild(svg);

    const legend = document.createElement("div");
    legend.className = "legend";
    keys.forEach((key, index) => {
      const entry = document.createElement("span");
      entry.innerHTML = `<i style="background:${PALETTE[index % PALETTE.length]}"></i>${key}`;
      legend.appendChild(entry);
    });
    element.appendChild(legend);
  }

  /* --------------------------------------------------------------- radar --- */
  function renderRadar(element, data, maxValue) {
    const isMulti = data && Array.isArray(data.series) && data.series.length;
    const labels = isMulti ? data.labels : (data || []).map((item) => item.label);
    const series = isMulti
      ? data.series
      : [{ name: "Играч", color: PALETTE[0], values: (data || []).map((item) => Number(item.value || 0)) }];

    if (!labels.length) return emptyState(element, "Няма данни за показване.");

    const size = 340;
    const center = size / 2;
    const radius = 118;
    const max = Number(maxValue || 100);
    const count = labels.length;
    const angle = (index) => (Math.PI * 2 * index) / count - Math.PI / 2;

    const svg = create("svg", {
      class: "chart",
      viewBox: `0 0 ${size} ${size}`,
      role: "img",
      "aria-label": "Радарна диаграма",
    });

    const rings = 4;
    for (let ring = rings; ring >= 1; ring -= 1) {
      const points = [];
      for (let index = 0; index < count; index += 1) {
        const r = (radius * ring) / rings;
        points.push(`${center + Math.cos(angle(index)) * r},${center + Math.sin(angle(index)) * r}`);
      }
      svg.appendChild(create("polygon", {
        points: points.join(" "),
        fill: ring === rings ? "rgba(148,163,184,0.05)" : "none",
        stroke: "rgba(148,163,184,0.16)",
        "stroke-width": 1,
      }));
    }

    for (let index = 0; index < count; index += 1) {
      svg.appendChild(create("line", {
        x1: center, y1: center,
        x2: center + Math.cos(angle(index)) * radius,
        y2: center + Math.sin(angle(index)) * radius,
        class: "grid-line",
      }));
      const labelX = center + Math.cos(angle(index)) * (radius + 24);
      const labelY = center + Math.sin(angle(index)) * (radius + 24);
      const label = create("text", {
        x: labelX, y: labelY + 4, "text-anchor": "middle", class: "axis-label",
      });
      label.textContent = labels[index];
      svg.appendChild(label);
    }

    series.forEach((entry) => {
      const color = entry.color || PALETTE[0];
      const points = entry.values.map((value, index) => {
        const r = (Math.max(0, Math.min(Number(value || 0), max)) / max) * radius;
        return `${center + Math.cos(angle(index)) * r},${center + Math.sin(angle(index)) * r}`;
      });
      svg.appendChild(create("polygon", {
        points: points.join(" "),
        fill: color,
        "fill-opacity": 0.16,
        stroke: color,
        "stroke-width": 2,
      }));
      points.forEach((point, index) => {
        const [x, y] = point.split(",").map(Number);
        const dot = create("circle", { cx: x, cy: y, r: 3.2, fill: color });
        dot.appendChild(create("title", {})).textContent =
          `${entry.name} · ${labels[index]}: ${entry.values[index]}`;
        svg.appendChild(dot);
      });
    });

    element.innerHTML = "";
    element.appendChild(svg);
  }

  /* ------------------------------------------------------------ dispatch --- */
  const RENDERERS = {
    donut: (element, data) => renderDonut(element, data),
    bars: (element, data) => renderBars(element, data, element.dataset.unit),
    "grouped-bars": (element, data) => renderGroupedBars(element, data),
    line: (element, data) => renderLine(element, data),
    radar: (element, data) => renderRadar(element, data, element.dataset.max),
  };

  function renderAll(root = document) {
    root.querySelectorAll("[data-chart]").forEach((element) => {
      const renderer = RENDERERS[element.dataset.chart];
      if (!renderer) return;
      renderer(element, parseData(element));
    });
  }

  global.FIPCharts = { renderAll, renderDonut, renderBars, renderGroupedBars, renderRadar, renderLine };

  document.addEventListener("DOMContentLoaded", () => renderAll());
})(window);
