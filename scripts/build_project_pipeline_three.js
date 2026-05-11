#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, "paper");
const SVG_PATH = path.join(OUT_DIR, "project_pipeline_acl_three.svg");
const PNG_PATH = path.join(OUT_DIR, "project_pipeline_acl_three.png");

function requireTool(name) {
  try {
    return require(name);
  } catch (err) {
    return require(path.join(ROOT, ".cache/js-tools/node_modules", name));
  }
}

const THREE = requireTool("three");
const sharp = requireTool("sharp");

const W = 2400;
const H = 1320;

const C = {
  bg: "#F8FAFC",
  paper: "#FFFFFF",
  ink: "#111827",
  text: "#273447",
  muted: "#64748B",
  line: "#CAD5E2",
  lineStrong: "#AEBBCC",
  blue: "#2458A8",
  blueSoft: "#E9F1FF",
  teal: "#087B75",
  tealSoft: "#E7F7F4",
  amber: "#B76B12",
  amberSoft: "#FFF2D9",
  violet: "#6551B5",
  violetSoft: "#F0EDFF",
  rose: "#B73C45",
  roseSoft: "#FFF0F2",
  green: "#2D7D4C",
  greenSoft: "#EAF7EF",
  slate: "#475569",
  slateSoft: "#F1F5F9",
};

const stageColors = [C.blue, C.teal, C.amber, C.violet, C.rose];
const stageSoft = [C.blueSoft, C.tealSoft, C.amberSoft, C.violetSoft, C.roseSoft];

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function wrap(text, maxChars) {
  const out = [];
  for (const raw of String(text).split("\n")) {
    const words = raw.trim().split(/\s+/).filter(Boolean);
    let line = "";
    for (const word of words) {
      const next = line ? `${line} ${word}` : word;
      if (next.length > maxChars && line) {
        out.push(line);
        line = word;
      } else {
        line = next;
      }
    }
    if (line) out.push(line);
  }
  return out;
}

function text(x, y, content, opts = {}) {
  const {
    size = 22,
    weight = 520,
    color = C.text,
    anchor = "start",
    lineHeight = 1.24,
    family = "Inter, Arial, Helvetica, sans-serif",
  } = opts;
  const lines = Array.isArray(content) ? content : [content];
  const tspans = lines
    .map((line, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : size * lineHeight}">${esc(line)}</tspan>`)
    .join("");
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${color}">${tspans}</text>`;
}

function rect(x, y, w, h, r, fill, stroke = "none", sw = 0, extra = "") {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" ${extra}/>`;
}

function poly(points, fill, stroke = "none", sw = 0, extra = "") {
  const d = points.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
  return `<polygon points="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" ${extra}/>`;
}

function shade(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  let r = (n >> 16) + amount;
  let g = ((n >> 8) & 255) + amount;
  let b = (n & 255) + amount;
  r = Math.max(0, Math.min(255, r));
  g = Math.max(0, Math.min(255, g));
  b = Math.max(0, Math.min(255, b));
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
}

const projectionBasis = {
  x: new THREE.Vector2(1, 0),
  y: new THREE.Vector2(0, -0.95),
  z: new THREE.Vector2(0.42, 0.24),
};

function project(v, originX = 1200, originY = 642, scale = 165) {
  const p = v.clone();
  return {
    x: originX + (p.x * projectionBasis.x.x + p.y * projectionBasis.y.x + p.z * projectionBasis.z.x) * scale,
    y: originY + (p.x * projectionBasis.x.y + p.y * projectionBasis.y.y + p.z * projectionBasis.z.y) * scale,
  };
}

function cuboid(cx, cz, opts = {}) {
  const {
    y = 0,
    w = 1.55,
    d = 0.9,
    h = 0.18,
    fill = C.blue,
    originX = 1200,
    originY = 642,
    scale = 165,
  } = opts;

  const V = {
    tnw: new THREE.Vector3(cx - w / 2, y, cz - d / 2),
    tne: new THREE.Vector3(cx + w / 2, y, cz - d / 2),
    tse: new THREE.Vector3(cx + w / 2, y, cz + d / 2),
    tsw: new THREE.Vector3(cx - w / 2, y, cz + d / 2),
    bnw: new THREE.Vector3(cx - w / 2, y - h, cz - d / 2),
    bne: new THREE.Vector3(cx + w / 2, y - h, cz - d / 2),
    bse: new THREE.Vector3(cx + w / 2, y - h, cz + d / 2),
    bsw: new THREE.Vector3(cx - w / 2, y - h, cz + d / 2),
  };
  const P = Object.fromEntries(Object.entries(V).map(([k, v]) => [k, project(v, originX, originY, scale)]));
  return `
    <g filter="url(#softDepth)">
      ${poly([P.tnw, P.tne, P.tse, P.tsw], fill, "rgba(255,255,255,0.25)", 1)}
      ${poly([P.tsw, P.tse, P.bse, P.bsw], shade(fill, -32))}
      ${poly([P.tne, P.tse, P.bse, P.bne], shade(fill, -48))}
    </g>
  `;
}

function backbone() {
  const centers = [-4.8, -2.4, 0, 2.4, 4.8];
  const slabs = centers.map((x, i) => cuboid(x, 0, { fill: stageColors[i], y: 0, h: 0.22 })).join("");
  const links = [];
  for (let i = 0; i < centers.length - 1; i += 1) {
    const a = centers[i] + 0.78;
    const b = centers[i + 1] - 0.78;
    const pts = [
      project(new THREE.Vector3(a, -0.05, -0.28)),
      project(new THREE.Vector3(b, -0.05, -0.28)),
      project(new THREE.Vector3(b, -0.05, 0.28)),
      project(new THREE.Vector3(a, -0.05, 0.28)),
    ];
    links.push(poly(pts, "#D7DEE8", "#C4CEDA", 1));
  }
  return `<g opacity="0.42">${links.join("")}${slabs}</g>`;
}

function bulletRow(x, y, w, label, value, color) {
  return `
    ${rect(x, y, w, 40, 10, "#FFFFFF", "#DCE5EF", 1.5)}
    <circle cx="${x + 22}" cy="${y + 20}" r="8" fill="${color}"/>
    ${text(x + 42, y + 26, label, { size: 16, weight: 760, color: C.text })}
    ${value ? text(x + w - 16, y + 26, value, { size: 16, weight: 850, color, anchor: "end" }) : ""}
  `;
}

function stageCard(stage) {
  const { x, y, w, h, no, title, sub, color, fill, rows } = stage;
  const rowY = y + 150;
  return `
    <g filter="url(#cardShadow)">
      ${rect(x, y, w, h, 22, fill, "#D7E0EB", 2)}
      ${rect(x, y, 12, h, 6, color)}
    </g>
    ${rect(x + 30, y + 28, 54, 34, 17, color)}
    ${text(x + 57, y + 52, no, { size: 18, weight: 850, color: "#FFFFFF", anchor: "middle" })}
    ${text(x + 100, y + 53, title, { size: 27, weight: 850, color: C.ink })}
    ${text(x + 30, y + 94, wrap(sub, 35), { size: 17, weight: 650, color: C.muted, lineHeight: 1.22 })}
    ${rows.map((r, i) => bulletRow(x + 30, rowY + i * 50, w - 60, r[0], r[1], r[2] || color)).join("")}
  `;
}

function arrow(x1, y1, x2, y2) {
  return `<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="${C.lineStrong}" stroke-width="4" stroke-linecap="round" marker-end="url(#arrow)"/>`;
}

function pill(x, y, w, label, color) {
  return `
    <g>
      ${rect(x, y, w, 40, 20, "#FFFFFF", color, 1.8)}
      ${text(x + w / 2, y + 26, label, { size: 15, weight: 780, color, anchor: "middle" })}
    </g>
  `;
}

function metricBox(x, y, w, title, rows, color, fill) {
  return `
    <g filter="url(#cardShadow)">
      ${rect(x, y, w, 188, 22, fill, "#D7E0EB", 2)}
      ${rect(x + 28, y + 30, 44, 44, 12, color)}
      ${text(x + 50, y + 60, "+", { size: 28, weight: 850, color: "#FFFFFF", anchor: "middle" })}
      ${text(x + 90, y + 59, title, { size: 25, weight: 850, color: C.ink })}
      ${rows
        .map(
          (r, i) => `
            <circle cx="${x + 34}" cy="${y + 101 + i * 28}" r="4" fill="${color}"/>
            ${text(x + 48, y + 107 + i * 28, r, { size: 17, weight: 640, color: C.text })}
          `,
        )
        .join("")}
    </g>
  `;
}

const stages = [
  {
    x: 92,
    y: 250,
    w: 390,
    h: 430,
    no: "01",
    title: "Data",
    sub: "Resolved binary forecasting questions with evidence.",
    color: C.blue,
    fill: C.blueSoft,
    rows: [
      ["Questions", "1,580"],
      ["Yes / no labels", "556 / 1,024"],
      ["Evidence summaries", "0-3"],
      ["Mean articles", "2.14"],
    ],
  },
  {
    x: 552,
    y: 250,
    w: 390,
    h: 430,
    no: "02",
    title: "Intervention",
    sub: "Only the requested rationale component is varied.",
    color: C.teal,
    fill: C.tealSoft,
    rows: [
      ["Neutral baseline", "V0"],
      ["Structured variants", "V1-V8"],
      ["Forecast components", "8"],
      ["Question/evidence", "fixed"],
    ],
  },
  {
    x: 1012,
    y: 250,
    w: 390,
    h: 430,
    no: "03",
    title: "Generation",
    sub: "Prompted model sweep across decoding settings.",
    color: C.amber,
    fill: C.amberSoft,
    rows: [
      ["Target models", "3"],
      ["Temperatures", "6"],
      ["Prompt conditions", "9"],
      ["Rationales", "logged"],
    ],
  },
  {
    x: 1472,
    y: 250,
    w: 390,
    h: 430,
    no: "04",
    title: "Forecast",
    sub: "Each run returns probability and rationale text.",
    color: C.violet,
    fill: C.violetSoft,
    rows: [
      ["Binary answer", "yes/no"],
      ["Probability", "p"],
      ["Rationale", "text"],
      ["Component field", "structured"],
    ],
  },
  {
    x: 1932,
    y: 250,
    w: 376,
    h: 430,
    no: "05",
    title: "Analysis",
    sub: "Reliability and explanation quality are evaluated jointly.",
    color: C.rose,
    fill: C.roseSoft,
    rows: [
      ["Accuracy / Brier", ""],
      ["ECE", ""],
      ["LLM judges", ""],
      ["SHAP attribution", ""],
    ],
  },
];

const variantLabels = [
  "V1 Predicted event",
  "V2 Key attribute",
  "V3 Reasoning type",
  "V4 Credibility",
  "V5 Key conditions",
  "V6 Step-by-step",
  "V7 Uncertainty",
  "V8 Temporal anchors",
];

const variantGrid = variantLabels
  .map((label, i) => {
    const col = i % 4;
    const row = Math.floor(i / 4);
    return pill(552 + col * 316, 736 + row * 54, 270, label, C.teal);
  })
  .join("");

const stageCards = stages.map(stageCard).join("");
const arrows = [
  arrow(494, 465, 540, 465),
  arrow(954, 465, 1000, 465),
  arrow(1414, 465, 1460, 465),
  arrow(1874, 465, 1920, 465),
].join("");

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="ACL paper pipeline figure for structured LLM rationale evaluation in forecasting">
  <defs>
    <filter id="cardShadow" x="-12%" y="-12%" width="126%" height="140%">
      <feDropShadow dx="0" dy="9" stdDeviation="12" flood-color="#0F172A" flood-opacity="0.10"/>
    </filter>
    <filter id="softDepth" x="-30%" y="-30%" width="160%" height="180%">
      <feDropShadow dx="0" dy="12" stdDeviation="14" flood-color="#0F172A" flood-opacity="0.13"/>
    </filter>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 9 4.5 L 0 9 z" fill="${C.lineStrong}"/>
    </marker>
  </defs>

  ${rect(0, 0, W, H, 0, C.bg)}
  ${rect(54, 54, 2292, 1212, 34, C.paper, "#E1E8F0", 2)}

  ${text(100, 122, "Experimental pipeline", { size: 29, weight: 850, color: C.blue })}
  ${text(100, 176, "Structured rationale components as controlled forecasting interventions.", { size: 33, weight: 830, color: C.ink })}
  ${text(100, 218, "Questions and evidence are fixed; the prompted rationale component is varied; reliability and explanation quality are measured.", { size: 21, weight: 560, color: C.muted })}

  ${backbone()}
  ${stageCards}
  ${arrows}

  ${text(552, 714, "Prompt variants", { size: 24, weight: 850, color: C.ink })}
  ${variantGrid}

  ${metricBox(92, 916, 648, "Forecast reliability", ["accuracy", "Brier score", "expected calibration error"], C.blue, C.blueSoft)}
  ${metricBox(876, 916, 648, "Rationale quality", ["plausibility, completeness", "source consistency, non-hallucination", "informativeness, conciseness"], C.violet, C.violetSoft)}
  ${metricBox(1660, 916, 648, "Attribution analysis", ["SHAP over judge attributes", "features predictive of correctness", "component-level diagnostics"], C.green, C.greenSoft)}

  ${rect(92, 1164, 2216, 58, 29, C.slateSoft, "#D7E0EB", 2)}
  ${text(1200, 1201, "Core comparison: forecast reliability versus judged explanation quality across V0-V8.", { size: 23, weight: 800, color: C.ink, anchor: "middle" })}
</svg>
`;

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(SVG_PATH, svg);
  await sharp(Buffer.from(svg), { density: 220 }).png().toFile(PNG_PATH);
  console.log(SVG_PATH);
  console.log(PNG_PATH);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
