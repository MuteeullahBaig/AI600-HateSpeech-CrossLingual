/**
 * AI-600 Hate Speech Detection – ISE-Hate Nastaliq Urdu Project
 * Presentation v4 — rebuilt for correct project content
 */

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" × 5.625"
pres.author = "Muteeullah Baig, Rana Muhammad Hamza";
pres.title = "Cross-Lingual Hate Speech Detection";

// ─────────────────────────────────────────────
// Color palette (Ocean Executive)
// ─────────────────────────────────────────────
const C = {
  navy:   "065A82",   // primary background / headers
  teal:   "0D9488",   // accent / highlights
  tealLt: "14B8A6",   // lighter teal
  ice:    "E0F2FE",   // light content background
  white:  "FFFFFF",
  dark:   "0F172A",   // body text
  mid:    "334155",   // secondary text / table rows
  muted:  "64748B",   // captions
  hateRed:"DC2626",   // hate class markers
  nonGrn: "16A34A",   // non-hate class markers
  rowAlt: "F0F9FF",   // alternating table row
};

const makeShadow = () => ({
  type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 3, angle: 135
});

// ─────────────────────────────────────────────
// SLIDE 1 – Title
// ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // accent bar left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: 5.625,
    fill: { color: C.teal }, line: { color: C.teal }
  });

  // top decorative strip
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.18, y: 0, w: 9.82, h: 0.08,
    fill: { color: C.tealLt }, line: { color: C.tealLt }
  });

  // main title
  s.addText("Cross-Lingual Hate Speech Detection", {
    x: 0.45, y: 1.0, w: 9.2, h: 1.1,
    fontSize: 36, bold: true, color: C.white,
    fontFace: "Calibri", margin: 0
  });

  // subtitle
  s.addText("English to Nastaliq Urdu Transfer Learning", {
    x: 0.45, y: 2.15, w: 9.2, h: 0.65,
    fontSize: 22, bold: false, color: C.tealLt,
    fontFace: "Calibri", margin: 0
  });

  // divider
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 2.92, w: 9.1, h: 0.04,
    fill: { color: "1C7293" }, line: { color: "1C7293" }
  });

  // authors
  s.addText("Muteeullah Baig  ·  Rana Muhammad Hamza", {
    x: 0.45, y: 3.1, w: 9.2, h: 0.45,
    fontSize: 15, color: "CADCFC", fontFace: "Calibri", margin: 0
  });

  // course info
  s.addText("AI-600 Deep Learning  ·  LUMS Spring 2026  ·  Project 6", {
    x: 0.45, y: 3.55, w: 9.2, h: 0.35,
    fontSize: 13, color: "94A3B8", fontFace: "Calibri", margin: 0
  });

  // bottom tag
  s.addText("Cross-Lingual Hate Speech Detection", {
    x: 0.45, y: 4.95, w: 9.1, h: 0.45,
    fontSize: 10, color: "64748B", fontFace: "Calibri",
    align: "right", margin: 0
  });
}

// ─────────────────────────────────────────────
// SLIDE 2 – Problem Statement
// ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.white };

  // header bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.78,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Problem Statement", {
    x: 0.4, y: 0, w: 8.5, h: 0.78,
    fontSize: 22, bold: true, color: C.white, fontFace: "Calibri",
    valign: "middle", margin: 0
  });
  s.addText("Slide 1 / 5", {
    x: 8.5, y: 0, w: 1.3, h: 0.78,
    fontSize: 10, color: "94A3B8", fontFace: "Calibri",
    valign: "middle", align: "right", margin: 0
  });

  // left column – Why Nastaliq Urdu?
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.9, w: 4.3, h: 1.35,
    fill: { color: C.ice }, line: { color: "BAE6FD", pt: 1 },
    shadow: makeShadow()
  });
  s.addText("Why Nastaliq Urdu?", {
    x: 0.5, y: 0.95, w: 3.9, h: 0.38,
    fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri", margin: 0
  });
  s.addText([
    { text: "Arabic-script Urdu spoken by 230M people", options: { bullet: true, breakLine: true } },
    { text: "ISE-Hate: first large annotated Urdu hate corpus (21,759 tweets)", options: { bullet: true, breakLine: true } },
    { text: "Nastaliq in XLM-R's 100-language pre-training; Roman Urdu is not", options: { bullet: true } },
  ], {
    x: 0.5, y: 1.33, w: 3.9, h: 0.87,
    fontSize: 10.5, color: C.mid, fontFace: "Calibri", margin: 0
  });

  // right column – Study Overview
  s.addShape(pres.shapes.RECTANGLE, {
    x: 4.8, y: 0.9, w: 4.9, h: 1.35,
    fill: { color: C.ice }, line: { color: "BAE6FD", pt: 1 },
    shadow: makeShadow()
  });
  s.addText("Study Overview", {
    x: 5.0, y: 0.95, w: 4.5, h: 0.38,
    fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri", margin: 0
  });
  s.addText([
    { text: "Fine-tune multilingual transformers on English hate speech data", options: { bullet: true, breakLine: true } },
    { text: "Evaluate zero-shot & few-shot transfer to Nastaliq Urdu (ISE-Hate)", options: { bullet: true, breakLine: true } },
    { text: "Compare neural cross-lingual transfer vs. classical ML baselines", options: { bullet: true } },
  ], {
    x: 5.0, y: 1.33, w: 4.55, h: 0.87,
    fontSize: 10.5, color: C.mid, fontFace: "Calibri", margin: 0
  });

  const rqs = [
    { tag: "RQ1", txt: "How does model capacity (mBERT → XLM-R-large) affect EN performance and Urdu zero-shot transfer?" },
    { tag: "RQ2", txt: "Is zero-shot failure due to class-imbalance prior or true domain gap?" },
    { tag: "RQ3", txt: "Do K-shot examples enable stable few-shot transfer to Nastaliq Urdu?" },
    { tag: "RQ4", txt: "Does the model show differential FPR across hate sub-categories (Level-2 fairness)?" },
  ];

  rqs.forEach((rq, i) => {
    const yStart = 2.45 + i * 0.7;
    // tag pill
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.3, y: yStart, w: 0.62, h: 0.42,
      fill: { color: C.teal }, line: { color: C.teal }, rectRadius: 0.08
    });
    s.addText(rq.tag, {
      x: 0.3, y: yStart, w: 0.62, h: 0.42,
      fontSize: 10.5, bold: true, color: C.white,
      fontFace: "Calibri", align: "center", valign: "middle", margin: 0
    });
    s.addText(rq.txt, {
      x: 1.05, y: yStart, w: 8.65, h: 0.42,
      fontSize: 11, color: C.mid, fontFace: "Calibri",
      valign: "middle", margin: 0
    });
  });

  // footer
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.33, w: 10, h: 0.3,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("AI-600  ·  Cross-Lingual Hate Speech Detection  ·  LUMS Spring 2026", {
    x: 0.3, y: 5.33, w: 9.4, h: 0.3,
    fontSize: 8.5, color: "94A3B8", fontFace: "Calibri",
    valign: "middle", align: "center", margin: 0
  });
}

// ─────────────────────────────────────────────
// SLIDE 3 – Dataset Description
// ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.white };

  // header
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.78,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Dataset Description", {
    x: 0.4, y: 0, w: 8.5, h: 0.78,
    fontSize: 22, bold: true, color: C.white, fontFace: "Calibri",
    valign: "middle", margin: 0
  });
  s.addText("Slide 2 / 5", {
    x: 8.5, y: 0, w: 1.3, h: 0.78,
    fontSize: 10, color: "94A3B8", fontFace: "Calibri",
    valign: "middle", align: "right", margin: 0
  });

  // SOURCE label
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.88, w: 4.3, h: 0.38,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  s.addText("SOURCE  ·  English (Davidson et al. 2017)", {
    x: 0.3, y: 0.88, w: 4.3, h: 0.38,
    fontSize: 11.5, bold: true, color: C.white, fontFace: "Calibri",
    align: "center", valign: "middle", margin: 0
  });

  // TARGET label
  s.addShape(pres.shapes.RECTANGLE, {
    x: 4.9, y: 0.88, w: 4.8, h: 0.38,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("TARGET  ·  ISE-Hate (Akram et al. 2023, Nastaliq Urdu)", {
    x: 4.9, y: 0.88, w: 4.8, h: 0.38,
    fontSize: 11.5, bold: true, color: C.white, fontFace: "Calibri",
    align: "center", valign: "middle", margin: 0
  });

  // Source dataset box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.3, w: 4.3, h: 2.8,
    fill: { color: C.ice }, line: { color: "BAE6FD", pt: 1 },
    shadow: makeShadow()
  });

  const srcRows = [
    ["Platform", "Twitter / X"],
    ["Train",    "19,826 samples"],
    ["Dev",      "2,477 samples"],
    ["Test",     "2,477 samples"],
    ["Hate %",   "5.2%  (severely imbalanced)"],
    ["Labels",   "Binarised: hate vs. non-hate"],
    ["HuggingFace", "tdavidson/hate_speech_offensive"],
  ];

  srcRows.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.white : C.rowAlt;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.35, y: 1.35 + i * 0.37, w: 4.2, h: 0.35,
      fill: { color: bg }, line: { color: bg }
    });
    s.addText(k, {
      x: 0.4, y: 1.37 + i * 0.37, w: 1.5, h: 0.31,
      fontSize: 10, bold: true, color: C.navy, fontFace: "Calibri",
      valign: "middle", margin: 0
    });
    s.addText(v, {
      x: 1.95, y: 1.37 + i * 0.37, w: 2.55, h: 0.31,
      fontSize: 10, color: C.mid, fontFace: "Calibri",
      valign: "middle", margin: 0
    });
  });

  // Target dataset box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 4.9, y: 1.3, w: 4.8, h: 2.8,
    fill: { color: C.ice }, line: { color: "BAE6FD", pt: 1 },
    shadow: makeShadow()
  });

  const tgtRows = [
    ["Platform",   "Twitter / X (Nastaliq script)"],
    ["Train",      "15,231 samples"],
    ["Dev",        "2,176 samples"],
    ["Test",       "4,352 samples"],
    ["Hate %",     "39.8%  (more balanced)"],
    ["Cohen's κ",  "0.71  (substantial agreement)"],
    ["Level-2",    "Interfaith / Sectarian / Ethnic / Other"],
  ];

  tgtRows.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.white : C.rowAlt;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 4.95, y: 1.35 + i * 0.37, w: 4.7, h: 0.35,
      fill: { color: bg }, line: { color: bg }
    });
    s.addText(k, {
      x: 5.0, y: 1.37 + i * 0.37, w: 1.65, h: 0.31,
      fontSize: 10, bold: true, color: C.navy, fontFace: "Calibri",
      valign: "middle", margin: 0
    });
    s.addText(v, {
      x: 6.7, y: 1.37 + i * 0.37, w: 2.9, h: 0.31,
      fontSize: 10, color: C.mid, fontFace: "Calibri",
      valign: "middle", margin: 0
    });
  });

  // key challenge note
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 4.2, w: 9.4, h: 0.72,
    fill: { color: "FFF7ED" }, line: { color: "FED7AA", pt: 1 }
  });
  s.addText("Key challenge: EN training data is 94.8% non-hate — the model learns a strong non-hate prior that dominates zero-shot Urdu inference", {
    x: 0.5, y: 4.26, w: 9.1, h: 0.58,
    fontSize: 11, color: "92400E", fontFace: "Calibri",
    valign: "middle", margin: 0
  });

  // footer
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.33, w: 10, h: 0.3,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("AI-600  ·  Cross-Lingual Hate Speech Detection  ·  LUMS Spring 2026", {
    x: 0.3, y: 5.33, w: 9.4, h: 0.3,
    fontSize: 8.5, color: "94A3B8", fontFace: "Calibri",
    valign: "middle", align: "center", margin: 0
  });
}

// ─────────────────────────────────────────────
// SLIDE 4 – Methodology
// ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.white };

  // header
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.78,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Methodology", {
    x: 0.4, y: 0, w: 8.5, h: 0.78,
    fontSize: 22, bold: true, color: C.white, fontFace: "Calibri",
    valign: "middle", margin: 0
  });
  s.addText("Slide 3 / 5", {
    x: 8.5, y: 0, w: 1.3, h: 0.78,
    fontSize: 10, color: "94A3B8", fontFace: "Calibri",
    valign: "middle", align: "right", margin: 0
  });

  // model backbone cards
  const models = [
    { name: "mBERT",         params: "179M params · 12 layers",  role: "Baseline",   color: "6B7280" },
    { name: "XLM-R-base",    params: "278M params · 12 layers",  role: "Ablation",   color: "0D9488" },
    { name: "XLM-R-large ★", params: "560M params · 24 layers", role: "Primary",    color: C.navy  },
  ];

  models.forEach((m, i) => {
    const x = 0.3 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 0.88, w: 3.0, h: 0.95,
      fill: { color: m.color }, line: { color: m.color },
      shadow: makeShadow()
    });
    s.addText(m.name, {
      x: x + 0.1, y: 0.90, w: 2.8, h: 0.38,
      fontSize: 14, bold: true, color: C.white, fontFace: "Calibri",
      align: "center", valign: "middle", margin: 0
    });
    s.addText(m.params, {
      x: x + 0.1, y: 1.27, w: 2.8, h: 0.24,
      fontSize: 9.5, color: "D1FAE5", fontFace: "Calibri",
      align: "center", margin: 0
    });
    s.addText(m.role, {
      x: x + 0.1, y: 1.5, w: 2.8, h: 0.22,
      fontSize: 9, color: "A5F3FC", fontFace: "Calibri",
      align: "center", margin: 0
    });
  });

  // transfer protocols header
  s.addText("Transfer Protocols", {
    x: 0.3, y: 1.94, w: 3.5, h: 0.32,
    fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri", margin: 0
  });

  const protocols = [
    {
      num: "1", title: "Zero-Shot",
      desc: "EN checkpoint evaluated directly on Urdu. No Urdu labels at inference time.\nModel biased to predict non-hate (94.8% EN prior) → low recall (0.080), not low precision (0.885)."
    },
    {
      num: "2", title: "Few-Shot (K = 8 / 16 / 32 / 64)",
      desc: "Stratified Urdu samples added to fine-tuning. Class weights [1.0, 2.5] for Urdu's 40% hate rate.\nMonotonic improvement K=8 (hate F1=0.590) → K=64 (0.694). No collapse on Nastaliq script."
    },
    {
      num: "3", title: "Classical ML (TF-IDF + LR / RF)",
      desc: "Unigram+bigram TF-IDF (50K features) + Logistic Regression / Random Forest.\nFuzzyWuzzy lexicon provides zero benefit: Latin lexicon vs Arabic-script Nastaliq = script mismatch."
    },
    {
      num: "4", title: "Supervised Fine-Tuning",
      desc: "XLM-R-large fine-tuned on full ISE-Hate train set (15,231 samples). Starts from EN checkpoint.\nAchieves hate F1 = 0.796 — upper bound, modest +0.049 over TF-IDF+LR at higher FPR cost."
    },
  ];

  protocols.forEach((p, i) => {
    const yBase = 2.32 + i * 0.72;
    const bg = i % 2 === 0 ? C.ice : C.white;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y: yBase, w: 9.4, h: 0.66,
      fill: { color: bg }, line: { color: "E2E8F0", pt: 1 }
    });
    // number circle
    s.addShape(pres.shapes.OVAL, {
      x: 0.38, y: yBase + 0.1, w: 0.42, h: 0.42,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText(p.num, {
      x: 0.38, y: yBase + 0.1, w: 0.42, h: 0.42,
      fontSize: 11, bold: true, color: C.white, fontFace: "Calibri",
      align: "center", valign: "middle", margin: 0
    });
    s.addText(p.title, {
      x: 0.9, y: yBase + 0.04, w: 2.5, h: 0.32,
      fontSize: 11, bold: true, color: C.navy, fontFace: "Calibri",
      valign: "middle", margin: 0
    });
    s.addText(p.desc, {
      x: 0.9, y: yBase + 0.32, w: 8.7, h: 0.3,
      fontSize: 9, color: C.mid, fontFace: "Calibri",
      valign: "top", margin: 0
    });
  });

  // footer
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.33, w: 10, h: 0.3,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("AI-600  ·  Cross-Lingual Hate Speech Detection  ·  LUMS Spring 2026", {
    x: 0.3, y: 5.33, w: 9.4, h: 0.3,
    fontSize: 8.5, color: "94A3B8", fontFace: "Calibri",
    valign: "middle", align: "center", margin: 0
  });
}

// ─────────────────────────────────────────────
// SLIDE 5 – Results & Comparison
// ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.white };

  // header
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.78,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Results & Comparison with Baselines", {
    x: 0.4, y: 0, w: 8.5, h: 0.78,
    fontSize: 22, bold: true, color: C.white, fontFace: "Calibri",
    valign: "middle", margin: 0
  });
  s.addText("Slide 4 / 5", {
    x: 8.5, y: 0, w: 1.3, h: 0.78,
    fontSize: 10, color: "94A3B8", fontFace: "Calibri",
    valign: "middle", align: "right", margin: 0
  });

  // subtitle
  s.addText("ISE-Hate Test Set  ·  4,352 Nastaliq Urdu samples", {
    x: 0.3, y: 0.85, w: 6.0, h: 0.3,
    fontSize: 11, italic: true, color: C.muted, fontFace: "Calibri", margin: 0
  });

  // results table header
  const cols = ["Method", "K", "Hate F1", "Macro F1", "FPR"];
  const colW = [3.5, 0.7, 1.3, 1.3, 1.1];
  const tableX = 0.3;

  // header row
  let cx = tableX;
  colW.forEach((w, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: 1.2, w: w, h: 0.38,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    s.addText(cols[i], {
      x: cx + 0.04, y: 1.2, w: w - 0.04, h: 0.38,
      fontSize: 10.5, bold: true, color: C.white, fontFace: "Calibri",
      valign: "middle", margin: 0
    });
    cx += w;
  });

  const rows = [
    { cells: ["Zero-shot (XLM-R-large)", "0",    "0.147", "0.462", "0.046"], highlight: false, hateColor: C.hateRed },
    { cells: ["Few-shot K=8",            "8",    "0.590", "0.630", "0.264"], highlight: false, hateColor: C.mid },
    { cells: ["Few-shot K=16",           "16",   "0.619", "0.652", "0.252"], highlight: false, hateColor: C.mid },
    { cells: ["Few-shot K=32",           "32",   "0.649", "0.676", "0.230"], highlight: false, hateColor: C.mid },
    { cells: ["Few-shot K=64",           "64",   "0.694", "0.719", "0.205"], highlight: false, hateColor: C.mid },
    { cells: ["TF-IDF + LR (classical)", "all",  "0.747", "0.753", "0.154"], highlight: false, hateColor: C.teal },
    { cells: ["XLM-R-large (supervised)", "all", "0.796", "0.816", "0.228"], highlight: true,  hateColor: C.navy },
  ];

  rows.forEach((row, i) => {
    const bg = row.highlight ? "ECFDF5" : (i % 2 === 0 ? C.white : C.rowAlt);
    cx = tableX;
    colW.forEach((w, j) => {
      s.addShape(pres.shapes.RECTANGLE, {
        x: cx, y: 1.6 + i * 0.37, w: w, h: 0.35,
        fill: { color: bg }, line: { color: "E2E8F0", pt: 0.5 }
      });
      const isHateCol = j === 2;
      const isFprCol  = j === 4;
      let textColor = C.mid;
      if (isHateCol) textColor = row.hateColor;
      if (isFprCol && row.highlight) textColor = C.hateRed;
      const isBold = (row.highlight && (j === 2 || j === 3)) || (j === 0 && row.highlight);
      s.addText(row.cells[j], {
        x: cx + 0.06, y: 1.62 + i * 0.37, w: w - 0.06, h: 0.31,
        fontSize: 10, bold: isBold, color: textColor, fontFace: "Calibri",
        valign: "middle", margin: 0
      });
      cx += w;
    });
  });

  // callout stats at bottom
  const callouts = [
    { val: "0.147", lbl: "Zero-shot\nHate F1", sub: "low recall failure", col: C.hateRed },
    { val: "0.747", lbl: "TF-IDF+LR\nHate F1", sub: "beats all few-shot", col: C.teal },
    { val: "0.796", lbl: "Supervised\nHate F1", sub: "project best", col: C.navy },
  ];

  callouts.forEach((c, i) => {
    const x = 0.3 + i * 3.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 4.28, w: 3.0, h: 0.95,
      fill: { color: C.ice }, line: { color: "BAE6FD", pt: 1 },
      shadow: makeShadow()
    });
    s.addText(c.val, {
      x: x + 0.05, y: 4.3, w: 1.05, h: 0.85,
      fontSize: 30, bold: true, color: c.col, fontFace: "Calibri",
      align: "center", valign: "middle", margin: 0
    });
    s.addText(c.lbl, {
      x: x + 1.1, y: 4.32, w: 1.6, h: 0.45,
      fontSize: 9.5, bold: true, color: C.dark, fontFace: "Calibri",
      valign: "top", margin: 0
    });
    s.addText(c.sub, {
      x: x + 1.1, y: 4.76, w: 1.6, h: 0.35,
      fontSize: 8.5, italic: true, color: C.muted, fontFace: "Calibri",
      valign: "top", margin: 0
    });
  });

  // footer
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.33, w: 10, h: 0.3,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("AI-600  ·  Cross-Lingual Hate Speech Detection  ·  LUMS Spring 2026", {
    x: 0.3, y: 5.33, w: 9.4, h: 0.3,
    fontSize: 8.5, color: "94A3B8", fontFace: "Calibri",
    valign: "middle", align: "center", margin: 0
  });
}

// ─────────────────────────────────────────────
// SLIDE 6 – Key Findings
// ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // accent bar left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: 5.625,
    fill: { color: C.teal }, line: { color: C.teal }
  });

  s.addText("Key Findings", {
    x: 0.38, y: 0.15, w: 9.0, h: 0.6,
    fontSize: 26, bold: true, color: C.white, fontFace: "Calibri", margin: 0
  });
  s.addText("Slide 5 / 5", {
    x: 8.5, y: 0.15, w: 1.3, h: 0.6,
    fontSize: 10, color: "94A3B8", fontFace: "Calibri",
    align: "right", margin: 0
  });

  const findings = [
    {
      num: "1",
      title: "Zero-shot fails due to class-imbalance prior — not domain gap",
      body: "EN training: 94.8% non-hate → strong non-hate prior. Recall=0.080, Precision=0.885 at zero-shot. XLM-R-large is BEST at zero-shot (hate F1=0.147) — larger models transfer better."
    },
    {
      num: "2",
      title: "Few-shot transfer is stable and monotonically improving",
      body: "Hate F1 rises K=8 (0.590) → K=16 (0.619) → K=32 (0.649) → K=64 (0.694). No collapse on Nastaliq Urdu. Class weights [1.0, 2.5] calibrated to Urdu's 40% hate rate."
    },
    {
      num: "3",
      title: "Classical ML beats all few-shot neural methods",
      body: "TF-IDF+LR on full data: hate F1=0.747. FuzzyWuzzy adds zero benefit — Latin lexicon vs. Arabic-script Nastaliq is a script mismatch. Random Forest achieves lowest FPR (0.056)."
    },
    {
      num: "4",
      title: "Supervised XLM-R-large: strong performance, high FPR cost",
      body: "Hate F1=0.796 (+0.049 over TF-IDF+LR) but FPR rises 0.154→0.228. Dominant signal is lexical (TF-IDF+LR nearly matches at 560× less compute). Modest gain may not justify cost."
    },
    {
      num: "5",
      title: "20× Level-2 fairness disparity at zero-shot",
      body: "Interfaith hate recall=28.2% vs. Other recall=1.4%. Annotator bias from Davidson AMT workers baked into EN weights — fires on English-pattern hate, blind to native Urdu idiom hate."
    },
  ];

  findings.forEach((f, i) => {
    const x = i < 3 ? 0.38 : 0.38 + 3.15 * (i - 3);
    const y = i < 3 ? 0.92 + i * 1.35 : 0.92;
    const w = i < 3 ? 9.3 : 4.45;

    // Actually, 5 items — use 3+2 layout
    // Recalculate based on layout: top 3 rows in left column, bottom 2 in right? No...
    // Let's do straight vertical 5 cards at uniform y spacing
  });

  // 5 horizontal cards, each occupying ~0.83" height
  findings.forEach((f, i) => {
    const yCard = 0.9 + i * 0.92;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.38, y: yCard, w: 9.3, h: 0.82,
      fill: { color: "0C3D57" }, line: { color: "1C7293", pt: 1 }
    });
    // number
    s.addShape(pres.shapes.OVAL, {
      x: 0.48, y: yCard + 0.15, w: 0.42, h: 0.42,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText(f.num, {
      x: 0.48, y: yCard + 0.15, w: 0.42, h: 0.42,
      fontSize: 11, bold: true, color: C.white, fontFace: "Calibri",
      align: "center", valign: "middle", margin: 0
    });
    // title
    s.addText(f.title, {
      x: 1.02, y: yCard + 0.05, w: 8.56, h: 0.34,
      fontSize: 11, bold: true, color: C.tealLt, fontFace: "Calibri",
      valign: "middle", margin: 0
    });
    // body
    s.addText(f.body, {
      x: 1.02, y: yCard + 0.38, w: 8.56, h: 0.38,
      fontSize: 9.5, color: "CBD5E1", fontFace: "Calibri",
      valign: "top", margin: 0
    });
  });

  // GitHub link bottom
  s.addText("github.com/MuteeullahBaig/AI600-HateSpeech-CrossLingual", {
    x: 0.38, y: 5.48, w: 9.3, h: 0.2,
    fontSize: 8.5, color: C.tealLt, fontFace: "Calibri",
    align: "center", margin: 0
  });
}

// ─────────────────────────────────────────────
// Write file
// ─────────────────────────────────────────────
pres.writeFile({ fileName: "D:\\Deep Learning Project\\report\\AI600_HateSpeech_Presentation_v4.pptx" })
  .then(() => console.log("✅  Saved: report/AI600_HateSpeech_Presentation_v4.pptx"))
  .catch(err => console.error("❌  Error:", err));
