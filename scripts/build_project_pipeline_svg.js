#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, "paper");
const SVG_PATH = path.join(OUT_DIR, "project_pipeline_js.svg");
const PNG_PATH = path.join(OUT_DIR, "project_pipeline_js.png");

function requireTool(name) {
  try {
    return require(name);
  } catch (err) {
    return require(path.join(ROOT, ".cache/js-tools/node_modules", name));
  }
}

const THREE = requireTool("three");

const W = 2400;
const H = 1050;

const C = {
  ink: "#152238",
  text: "#2B3440",
  muted: "#64748B",
  faint: "#EEF2F7",
  line: "#C9D3DF",
  bg: "#FBFCFE",
  panel: "#FFFFFF",
  blue: "#2556A8",
  blueSoft: "#EAF1FF",
  teal: "#0C766E",
  tealSoft: "#E7F6F3",
  amber: "#A76318",
  amberSoft: "#FFF3DE",
  violet: "#6546A3",
  violetSoft: "#F2EDFF",
  red: "#B33A3A",
  redSoft: "#FFF0F0",
  green: "#2F7D4E",
  greenSoft: "#EAF7EF",
  gold: "#B48A00",
  goldSoft: "#FFF8D8",
};

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function wrap(text, max) {
  const lines = [];
  for (const raw of String(text).split("\n")) {
    const words = raw.trim().split(/\s+/).filter(Boolean);
    let line = "";
    for (const word of words) {
      const next = line ? `${line} ${word}` : word;
      if (next.length > max && line) {
        lines.push(line);
        line = word;
      } else {
        line = next;
      }
    }
    if (line) lines.push(line);
  }
  return lines;
}

function text(x, y, content, opts = {}) {
  const {
    size = 24,
    weight = 500,
    color = C.text,
    anchor = "start",
    lineHeight = 1.22,
    family = "Inter, Arial, Helvetica, sans-serif",
  } = opts;
  const lines = Array.isArray(content) ? content : [content];
  const tspans = lines
    .map((line, i) => {
      const dy = i === 0 ? 0 : size * lineHeight;
      return `<tspan x="${x}" dy="${dy}">${esc(line)}</tspan>`;
    })
    .join("");
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${color}">${tspans}</text>`;
}

function roundedRect(x, y, w, h, r, fill, stroke = "none", sw = 0, extra = "") {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" ${extra}/>`;
}

function shade(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.max(0, Math.min(255, (n >> 16) + amount));
  const g = Math.max(0, Math.min(255, ((n >> 8) & 255) + amount));
  const b = Math.max(0, Math.min(255, (n & 255) + amount));
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
}

function project3(v) {
  return {
    x: v.x + v.z * 0.52,
    y: v.y - v.z * 0.28,
  };
}

function polygon(points, fill, stroke = "none", sw = 0) {
  const coords = points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  return `<polygon points="${coords}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function slab3d(x, y, w, h, depth, color) {
  const A = project3(new THREE.Vector3(x, y, 0));
  const B = project3(new THREE.Vector3(x + w, y, 0));
  const C = project3(new THREE.Vector3(x + w, y + h, 0));
  const D = project3(new THREE.Vector3(x, y + h, 0));
  const Ab = project3(new THREE.Vector3(x, y, depth));
  const Bb = project3(new THREE.Vector3(x + w, y, depth));
  const Cb = project3(new THREE.Vector3(x + w, y + h, depth));
  const Db = project3(new THREE.Vector3(x, y + h, depth));

  return `
    <g filter="url(#depthShadow)">
      ${polygon([A, B, Bb, Ab], shade(color, 38), "rgba(255,255,255,0.35)", 1)}
      ${polygon([B, C, Cb, Bb], shade(color, -28))}
      ${polygon([D, C, Cb, Db], shade(color, -46))}
      ${polygon([A, B, C, D], color)}
    </g>
  `;
}

function stage({ x, y, w, h, no, title, subtitle, accent, fill, rows }) {
  const innerX = x + 34;
  const rowStart = y + (rows.length === 3 ? 174 : 152);
  const top = `
    <g filter="url(#cardShadow)">
      ${roundedRect(x, y, w, h, 22, fill, "#D7DFEA", 2)}
      ${roundedRect(x, y, 12, h, 6, accent)}
    </g>
    ${roundedRect(innerX, y + 30, 54, 34, 17, accent)}
    ${text(innerX + 27, y + 54, no, { size: 20, weight: 850, color: "#FFFFFF", anchor: "middle" })}
    ${text(innerX + 72, y + 55, title, { size: 29, weight: 850, color: C.ink })}
    ${text(innerX, y + 94, wrap(subtitle, 34), { size: 18, weight: 600, color: C.muted, lineHeight: 1.25 })}
  `;

  const rowSvg = rows
    .map((row, i) => {
      const ry = rowStart + i * 58;
      return `
        ${roundedRect(innerX, ry, w - 68, 42, 12, "#FFFFFF", "#E1E8F0", 1.5)}
        ${roundedRect(innerX + 14, ry + 12, 18, 18, 9, row.color || accent)}
        ${text(innerX + 46, ry + 27, row.label, { size: 17, weight: 750, color: C.text })}
        ${row.value ? text(x + w - 42, ry + 27, row.value, { size: 17, weight: 850, color: row.color || accent, anchor: "end" }) : ""}
      `;
    })
    .join("");
  return `<g>${top}${rowSvg}</g>`;
}

function arrow(x1, y1, x2, y2, color = C.line) {
  return `<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="${color}" stroke-width="4" stroke-linecap="round" marker-end="url(#smallArrow)"/>`;
}

function capsule(x, y, w, label, color) {
  return `
    <g>
      ${roundedRect(x, y, w, 42, 21, "#FFFFFF", color, 1.8)}
      ${text(x + w / 2, y + 27, label, { size: 17, weight: 760, color, anchor: "middle" })}
    </g>
  `;
}

function metricBox(x, y, w, title, body, color, fill) {
  const h = 174;
  return `
    <g filter="url(#cardShadow)">
      ${roundedRect(x, y, w, h, 20, fill, "#D7DFEA", 2)}
      ${roundedRect(x + 26, y + 28, 42, 42, 12, color)}
      ${text(x + 47, y + 57, "+", { size: 28, weight: 850, color: "#FFFFFF", anchor: "middle" })}
      ${text(x + 86, y + 56, title, { size: 25, weight: 850, color: C.ink })}
      ${text(x + 26, y + 98, wrap(body, 54), { size: 17, weight: 560, color: C.text, lineHeight: 1.24 })}
    </g>
  `;
}

const stages = [
  {
    x: 100,
    y: 120,
    w: 392,
    h: 438,
    no: "01",
    title: "Data",
    subtitle: "Resolved binary questions from the forecasting corpus.",
    accent: C.blue,
    fill: C.blueSoft,
    rows: [
      { label: "Metaculus-style questions", value: "1,580", color: C.blue },
      { label: "Yes outcomes", value: "556", color: C.green },
      { label: "No outcomes", value: "1,024", color: C.red },
      { label: "News articles/question", value: "3", color: C.blue },
    ],
  },
  {
    x: 562,
    y: 120,
    w: 392,
    h: 438,
    no: "02",
    title: "Prompt Design",
    subtitle: "Baseline plus controlled rationale interventions.",
    accent: C.teal,
    fill: C.tealSoft,
    rows: [
      { label: "Neutral baseline", value: "V0", color: C.teal },
      { label: "Structured variants", value: "V1-V8", color: C.teal },
      { label: "Forecast components", value: "8", color: C.teal },
    ],
  },
  {
    x: 1024,
    y: 120,
    w: 392,
    h: 438,
    no: "03",
    title: "Generation",
    subtitle: "Model and decoding sweep over every prompt condition.",
    accent: C.amber,
    fill: C.amberSoft,
    rows: [
      { label: "Target models", value: "3", color: C.amber },
      { label: "Temperatures", value: "6", color: C.amber },
      { label: "Prompt conditions", value: "9", color: C.amber },
    ],
  },
  {
    x: 1486,
    y: 120,
    w: 392,
    h: 438,
    no: "04",
    title: "Outputs",
    subtitle: "Each run emits a forecast and explanation fields.",
    accent: C.violet,
    fill: C.violetSoft,
    rows: [
      { label: "Binary prediction", value: "yes/no", color: C.violet },
      { label: "Probability estimate", value: "p", color: C.violet },
      { label: "Natural rationale", value: "text", color: C.violet },
      { label: "Variant component", value: "field", color: C.violet },
    ],
  },
  {
    x: 1948,
    y: 120,
    w: 352,
    h: 438,
    no: "05",
    title: "Evaluation",
    subtitle: "Reliability, explanation quality, and attribution analysis.",
    accent: C.red,
    fill: C.redSoft,
    rows: [
      { label: "Accuracy", value: "", color: C.red },
      { label: "Brier score", value: "", color: C.red },
      { label: "ECE", value: "", color: C.red },
      { label: "LLM-judge scores", value: "", color: C.red },
    ],
  },
];

const variants = [
  ["V1", "Predicted event"],
  ["V2", "Key attribute"],
  ["V3", "Reasoning type"],
  ["V4", "Credibility"],
  ["V5", "Key conditions"],
  ["V6", "Step-by-step"],
  ["V7", "Uncertainty"],
  ["V8", "Temporal anchors"],
];

const variantChips = variants
  .map(([v, label], i) => {
    const col = i % 4;
    const row = Math.floor(i / 4);
    const x = 560 + col * 320;
    const y = 646 + row * 58;
    return capsule(x, y, 270, `${v}  ${label}`, C.teal);
  })
  .join("");

const stageSvg = stages.map(stage).join("");
const depthSvg = `
  <g opacity="0.22">
    ${slab3d(126, 562, 320, 24, 66, C.blue)}
    ${slab3d(588, 562, 320, 24, 66, C.teal)}
    ${slab3d(1050, 562, 320, 24, 66, C.amber)}
    ${slab3d(1512, 562, 320, 24, 66, C.violet)}
    ${slab3d(1974, 562, 280, 24, 66, C.red)}
  </g>
`;
const stageArrows = [
  arrow(506, 339, 548, 339),
  arrow(968, 339, 1010, 339),
  arrow(1430, 339, 1472, 339),
  arrow(1892, 339, 1934, 339),
].join("");

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Structured LLM rationale evaluation pipeline for forecasting">
  <defs>
    <filter id="cardShadow" x="-12%" y="-12%" width="124%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#0F172A" flood-opacity="0.09"/>
    </filter>
    <filter id="depthShadow" x="-20%" y="-40%" width="145%" height="190%">
      <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#0F172A" flood-opacity="0.12"/>
    </filter>
    <marker id="smallArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 8 4 L 0 8 z" fill="${C.line}"/>
    </marker>
    <linearGradient id="headerGrad" x1="0" x2="1">
      <stop offset="0%" stop-color="#152238"/>
      <stop offset="100%" stop-color="#2556A8"/>
    </linearGradient>
  </defs>

  ${roundedRect(0, 0, W, H, 0, C.bg)}
  ${roundedRect(64, 50, 2272, 960, 34, "#FFFFFF", "#E5EBF2", 2)}

  ${depthSvg}
  ${stageSvg}
  ${stageArrows}

  ${text(560, 620, "Rationale interventions beside the neutral baseline", { size: 25, weight: 850, color: C.ink })}
  ${variantChips}

  ${metricBox(100, 810, 660, "Forecast Reliability", "Accuracy, Brier score, and expected calibration error are computed against resolved labels.", C.blue, C.blueSoft)}
  ${metricBox(870, 810, 660, "Rationale Quality", "Gemma-4-31B-it and Kimi-K2.5 judge plausibility, completeness, source consistency, non-hallucination, informativeness, and conciseness.", C.violet, C.violetSoft)}
  ${metricBox(1640, 810, 660, "Diagnostic Attribution", "SHAP models estimate which judged rationale attributes are most predictive of forecast correctness.", C.green, C.greenSoft)}
</svg>
`;

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(SVG_PATH, svg);

  try {
    const sharp = requireTool("sharp");
    await sharp(Buffer.from(svg), { density: 220 }).png().toFile(PNG_PATH);
    console.log(SVG_PATH);
    console.log(PNG_PATH);
  } catch (err) {
    console.log(SVG_PATH);
    console.warn(`PNG export skipped: ${err.message}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
