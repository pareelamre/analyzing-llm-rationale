#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, "paper");
const SVG_PATH = path.join(OUT_DIR, "system_flow.svg");

const W = 1320;
const H = 520;

const C = {
  ink: "#151515",
  text: "#333333",
  muted: "#666666",
  line: "#7A7A7A",
  hairline: "#CFCFCF",
  fill: "#FAFAFA",
  accent: "#B87518",
  accentFill: "#F9EFE1",
};

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function wrap(value, maxChars) {
  const lines = [];
  const words = String(value).trim().split(/\s+/).filter(Boolean);
  let line = "";
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (next.length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function text(x, y, value, opts = {}) {
  const {
    size = 16,
    weight = 400,
    color = C.text,
    anchor = "start",
    lineHeight = 1.25,
    family = "Arial, Helvetica, sans-serif",
  } = opts;
  const lines = Array.isArray(value) ? value : [value];
  const tspans = lines
    .map((line, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : size * lineHeight}">${esc(line)}</tspan>`)
    .join("");
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${color}">${tspans}</text>`;
}

function rect(x, y, w, h, opts = {}) {
  const fill = opts.fill || "#FFFFFF";
  const stroke = opts.stroke || C.line;
  const sw = opts.sw || 1.3;
  const r = opts.r || 0;
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function line(x1, y1, x2, y2, opts = {}) {
  const stroke = opts.stroke || C.line;
  const sw = opts.sw || 1.2;
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function arrow(x1, y1, x2, y2) {
  return `<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="${C.line}" stroke-width="1.5" marker-end="url(#arrow)"/>`;
}

function elbow(points) {
  const d = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");
  return `<path d="${d}" fill="none" stroke="${C.line}" stroke-width="1.5" stroke-linejoin="round" marker-end="url(#arrow)"/>`;
}

function box(x, y, w, h, title, subtitle, opts = {}) {
  const fill = opts.fill || "#FFFFFF";
  const titleColor = opts.titleColor || C.ink;
  return `
    ${rect(x, y, w, h, { fill, stroke: C.line, sw: 1.2 })}
    ${text(x + w / 2, y + 29, title, { size: 16, weight: 700, color: titleColor, anchor: "middle" })}
    ${line(x + 18, y + 42, x + w - 18, y + 42, { stroke: C.hairline, sw: 1 })}
    ${text(x + w / 2, y + 67, wrap(subtitle, Math.floor(w / 8.2)), {
      size: 13,
      color: C.text,
      anchor: "middle",
      lineHeight: 1.2,
    })}
  `;
}

function smallBox(x, y, w, title, subtitle) {
  return `
    ${rect(x, y, w, 58, { fill: "#FFFFFF", stroke: C.line, sw: 1.1 })}
    ${text(x + w / 2, y + 22, title, { size: 13, weight: 700, color: C.ink, anchor: "middle" })}
    ${text(x + w / 2, y + 43, subtitle, { size: 12, color: C.muted, anchor: "middle" })}
  `;
}

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Structured rationale forecasting system flow">
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 8 4 L 0 8 z" fill="${C.line}"/>
    </marker>
  </defs>

  ${rect(0, 0, W, H, { fill: "#FFFFFF", stroke: "#FFFFFF", sw: 0 })}

  ${text(W / 2, 38, "Structured rationale forecasting pipeline", { size: 24, weight: 700, color: C.ink, anchor: "middle" })}
  ${text(W / 2, 63, "The question and evidence are fixed; only the requested rationale component, model, and temperature vary.", { size: 14, color: C.muted, anchor: "middle" })}

  ${line(52, 92, 1268, 92, { stroke: C.line, sw: 1 })}

  ${box(70, 132, 180, 108, "Dataset", "Resolved binary questions with evidence summaries", { fill: C.fill })}
  ${box(322, 132, 198, 108, "Prompt variant", "V0 baseline or one V1-V8 rationale intervention", { fill: C.accentFill, titleColor: C.accent })}
  ${box(592, 132, 176, 108, "LLM run", "model x temperature sweep with retries", { fill: C.fill })}
  ${box(840, 132, 184, 108, "Parsed output", "answer, confidence, rationale, optional field", { fill: C.fill })}

  ${arrow(255, 186, 316, 186)}
  ${arrow(525, 186, 586, 186)}
  ${arrow(773, 186, 834, 186)}

  ${rect(1084, 124, 166, 124, { fill: "#FFFFFF", stroke: C.line, sw: 1.2 })}
  ${text(1167, 153, "Run files", { size: 16, weight: 700, color: C.ink, anchor: "middle" })}
  ${line(1102, 166, 1232, 166, { stroke: C.hairline, sw: 1 })}
  ${text(1167, 192, "results_variant*.json", { size: 13, color: C.text, anchor: "middle" })}
  ${text(1167, 214, "metadata + error logs", { size: 13, color: C.text, anchor: "middle" })}
  ${arrow(1029, 186, 1078, 186)}

  ${line(52, 292, 1268, 292, { stroke: C.hairline, sw: 1 })}
  ${text(70, 322, "Evaluation", { size: 18, weight: 700, color: C.accent })}

  ${smallBox(184, 342, 214, "Forecast reliability", "accuracy, Brier, ECE")}
  ${smallBox(472, 342, 214, "Rationale quality", "LLM judges + proxy scores")}
  ${smallBox(760, 342, 214, "Attribution", "SHAP over rationale attributes")}
  ${smallBox(1048, 342, 214, "Comparison", "models, temperatures, V0-V8")}

  ${elbow([[1167, 252], [1167, 296], [291, 296], [291, 336]])}
  ${elbow([[1167, 252], [1167, 296], [579, 296], [579, 336]])}
  ${elbow([[1167, 252], [1167, 296], [867, 296], [867, 336]])}
  ${elbow([[1167, 252], [1167, 296], [1155, 296], [1155, 336]])}

  ${line(52, 452, 1268, 452, { stroke: C.line, sw: 1 })}
  ${text(W / 2, 482, "Main question: when do rationale interventions improve explanation quality without degrading forecast calibration?", {
    size: 15,
    weight: 700,
    color: C.text,
    anchor: "middle",
  })}
</svg>
`;

function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(SVG_PATH, svg, "utf8");
  console.log(SVG_PATH);
}

main();
