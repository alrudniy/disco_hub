// Generates Discovery_Hub_Pipeline_Description.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, TableOfContents, PageNumber, Header, Footer, PageBreak,
  ExternalHyperlink,
} = require("docx");

// ---------- palette ----------
const INK = "1A1A1A", ACCENT = "2E6DA4", LIGHT = "5B6B7B";
const CODE_BG = "F4F5F7", HEAD_BG = "DCE6F1", CALL_BG = "EAF3FB", CALL_BD = "9CC3E6";
const WARN_BG = "FBF1E6", WARN_BD = "E6C79C";
const CONTENT_W = 9360;

const tline = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const tborders = { top: tline, bottom: tline, left: tline, right: tline };

// ---------- helpers ----------
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });

function P(runs, opts = {}) {
  const children = (Array.isArray(runs) ? runs : [runs]).map((r) =>
    typeof r === "string" ? new TextRun(r) : r);
  return new Paragraph({ spacing: { after: 120, line: 276 }, children, ...opts });
}
const B = (t) => new TextRun({ text: t, bold: true });
const code = (t) => new TextRun({ text: t, font: "Consolas", size: 19, color: "B5179E" });

function bullet(runs) {
  const children = (Array.isArray(runs) ? runs : [runs]).map((r) =>
    typeof r === "string" ? new TextRun(r) : r);
  return new Paragraph({ numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 276 }, children });
}
function num(runs, ref = "numbers") {
  const children = (Array.isArray(runs) ? runs : [runs]).map((r) =>
    typeof r === "string" ? new TextRun(r) : r);
  return new Paragraph({ numbering: { reference: ref, level: 0 },
    spacing: { after: 80, line: 276 }, children });
}

// shaded monospace code block as a single-cell table
function codeBlock(lines) {
  const paras = lines.map((ln, i) =>
    new Paragraph({
      spacing: { after: i === lines.length - 1 ? 0 : 20, line: 240 },
      children: [new TextRun({ text: ln === "" ? " " : ln, font: "Consolas", size: 18, color: "2B2B2B" })],
    }));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: tborders, width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { fill: CODE_BG, type: ShadingType.CLEAR },
      margins: { top: 120, bottom: 120, left: 160, right: 160 },
      children: paras,
    })] })],
  });
}

// callout box (info or warn)
function callout(titleText, bodyParas, warn = false) {
  const head = new Paragraph({ spacing: { after: 60 },
    children: [new TextRun({ text: titleText, bold: true, color: warn ? "8A5A1E" : "23527C" })] });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: { top: { style: BorderStyle.SINGLE, size: 1, color: warn ? WARN_BD : CALL_BD },
                 bottom: { style: BorderStyle.SINGLE, size: 1, color: warn ? WARN_BD : CALL_BD },
                 left: { style: BorderStyle.SINGLE, size: 18, color: warn ? WARN_BD : CALL_BD },
                 right: { style: BorderStyle.SINGLE, size: 1, color: warn ? WARN_BD : CALL_BD } },
      width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { fill: warn ? WARN_BG : CALL_BG, type: ShadingType.CLEAR },
      margins: { top: 140, bottom: 140, left: 200, right: 200 },
      children: [head, ...bodyParas],
    })] })],
  });
}

// generic table; headers: [str], rows: [[cell,...]], widths sum to CONTENT_W
// each cell can be a string or an array of runs
function makeTable(headers, rows, widths) {
  const mkCellChildren = (c) => {
    if (Array.isArray(c)) return [new Paragraph({ spacing: { after: 0, line: 252 }, children: c })];
    return [new Paragraph({ spacing: { after: 0, line: 252 }, children: [new TextRun(String(c))] })];
  };
  const headerRow = new TableRow({ tableHeader: true, children: headers.map((h, i) =>
    new TableCell({ borders: tborders, width: { size: widths[i], type: WidthType.DXA },
      shading: { fill: HEAD_BG, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: h, bold: true })] })] })) });
  const bodyRows = rows.map((r) => new TableRow({ children: r.map((c, i) =>
    new TableCell({ borders: tborders, width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 70, bottom: 70, left: 120, right: 120 },
      children: mkCellChildren(c) })) }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths, rows: [headerRow, ...bodyRows] });
}
const spacer = (after = 120) => new Paragraph({ spacing: { after }, children: [] });

// ---------- document body ----------
const body = [];

// Title block
body.push(new Paragraph({ spacing: { after: 40 },
  children: [new TextRun({ text: "Discovery Hub", bold: true, size: 52, color: INK })] }));
body.push(new Paragraph({ spacing: { after: 40 },
  children: [new TextRun({ text: "End-to-End Pipeline — Description", bold: true, size: 30, color: ACCENT })] }));
body.push(new Paragraph({ spacing: { after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 6 } },
  children: [new TextRun({ text: "Nine numbered stages implementing the three-layer architecture, each with a deterministic mock mode.", italics: true, size: 22, color: LIGHT })] }));

body.push(P([
  "This pipeline implements the three-layer Discovery Hub architecture (Evidence Integration → Retrieval & Ranking → Explanation & Workflow) as nine runnable stages. Every stage has a deterministic ",
  code("--mock"),
  " mode, so the whole system runs, tests, and demonstrates its reproducibility without a GPU or any network download — the same way ",
  code("discovery_finetune"),
  " is validated against real files with a mock embedder. The real-mode code (live APIs, GPU models, R-GCN, vLLM) is present in each stage, gated behind lazy imports.",
]));

// TOC
body.push(spacer(80));
body.push(new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 24, color: INK })], spacing: { after: 100 } }));
body.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// 1. Architecture mapping
body.push(H1("1. How the code maps to the three-layer architecture"));
body.push(makeTable(
  ["Deck layer", "What it does", "Stages"],
  [
    [[B("Layer 1 — Evidence Integration")], "Ingest patents, publications, trials & org signals into a translation-oriented knowledge graph (technologies, inventors, organizations, experts, facilities)", [code("01"), new TextRun(" download · "), code("02"), new TextRun(" parse · "), code("03"), new TextRun(" graph")]],
    [[B("Layer 2 — Retrieval & Ranking")], "Hybrid semantic search + graph signal + neural rerank; where the existing discovery_finetune package lives", [code("04"), new TextRun(" embed · "), code("05"), new TextRun(" index · "), code("06"), new TextRun(" R-GCN · "), code("07"), new TextRun(" rank")]],
    [[B("Layer 3 — Explanation & Workflow")], "Strict-RAG, multi-agent pipeline producing cited, provenance-stamped, confidence-scored recommendations with human-in-the-loop", [code("08"), new TextRun(" multi-agent RAG")]],
    [[B("Cross-cutting — Reproducibility QA")], "Measures and reports run-to-run stability at every stage (the investor's question)", [code("09"), new TextRun(" stability harness")]],
  ],
  [2300, 5060, 2000]));
body.push(spacer(100));
body.push(P([
  "The existing ", code("discovery_finetune"), " work (schema normalization, synthetic-query generation, hard-negative mining, MNRL fine-tuning of Qwen3-Embedding-0.6B, Recall@k / MRR@10) is the ", B("retrieval layer"),
  " inside this fuller vision. Stages ", code("02"), ", ", code("04"), ", and ", code("07"),
  " are where it plugs in; the net-new work is the knowledge graph (", code("03"), "/", code("06"), ") and the multi-agent explanation layer (", code("08"), ").",
]));

// 2. Nine stages
body.push(H1("2. The nine stages, in execution order"));
body.push(P(["Run them in numeric order; each consumes the previous stage's artifacts under ", code("$DH_DATA_ROOT"), " (default ", code("./data"), ")."]));
body.push(makeTable(
  ["#", "Script", "Input → Output", "Compute"],
  [
    ["01", [code("01_download_data.py")], "source APIs → raw/*.jsonl", "Anvil CPU*"],
    ["02", [code("02_parse_normalize.py")], "raw/* → normalized/docs.jsonl", "Anvil CPU*"],
    ["03", [code("03_build_graph.py")], "docs → graph/{nodes,edges}.jsonl", "Anvil CPU*"],
    ["04", [code("04_generate_embeddings.py")], "docs → embeddings/doc_vectors.npy", [B("Anvil GPU")]],
    ["05", [code("05_build_index.py")], "vectors → index/faiss.index", "Anvil → Drew"],
    ["06", [code("06_train_rgcn.py")], "graph → artifacts/rgcn_node_emb.npy", [B("Anvil GPU")]],
    ["07", [code("07_retrieve_rank.py")], "index + R-GCN → ranked candidates", [B("Drew")]],
    ["08", [code("08_multiagent_rag.py")], "candidates → cited recommendations", [B("Drew")]],
    ["09", [code("09_stability_harness.py")], "runs 07/08 K times → report", "either"],
  ],
  [560, 2640, 4160, 2000]));
body.push(new Paragraph({ spacing: { before: 80, after: 160 },
  children: [new TextRun({ text: "* For the MVP biomedical slice (< ~200 GB) stages 01–06 run fine on Drew too; full scale (~2–3 TB working set) needs Anvil.", italics: true, size: 20, color: LIGHT })] }));

body.push(H3("What each stage does"));
const stageItems = [
  ["01 — Download. ", "Pulls records from ClinicalTrials.gov v2, OpenAlex, SBIR, USPTO/PatentsView, and AUTM. Mock mode generates a deterministic synthetic corpus shaped like each real schema (nested protocolSection for trials, flat dicts for USPTO, OpenAlex inverted-index abstracts). For full-scale OpenAlex, use the S3 snapshot, not the REST API."],
  ["02 — Parse & normalize. ", "One parser per source converges every schema onto the unified DiscoveryDoc with a canonical embedding_text. Includes the AUTM noise filter. This is the contract that lets the students' data-collection work proceed in parallel."],
  ["03 — Build graph. ", "Constructs the heterogeneous KG: technology nodes (keyed by doc_id — each invention is unique) plus deduplicated inventor / organization / expert / facility nodes, connected by typed relations. Validates with networkx."],
  ["04 — Embeddings. ", "Batch-embeds every embedding_text. The Anvil A100/H100 job at full scale; ~30–100 A100-hours for 40–50M abstracts. Mock embedder is bit-identical across machines."],
  ["05 — Index. ", "Builds a FAISS IndexFlatIP (exact, deterministic) — chosen for the reproducibility story; a numpy fallback runs if FAISS is absent. Built on Anvil, shipped to Drew."],
  ["06 — R-GCN. ", "Learns relational node embeddings. Real path: PyTorch Geometric RGCNConv with link prediction. Mock path: deterministic numpy message-passing. With neighbor sampling this fits a 12 GB GPU; a run is ~1–6 GPU-hours."],
  ["07 — Retrieve & rank. ", "Two-stage: text recall (FAISS) → blend R-GCN graph similarity → rerank (BGE cross-encoder in real mode, blended-score sort in mock). Returns candidates with supporting evidence. Importable Retriever class."],
  ["08 — Multi-agent RAG. ", "Five agents in sequence — retrieval, expertise-gap, reranking, policy-safety, explanation — as a state dict that maps 1:1 onto LangGraph nodes. Strict RAG: every claim carries a citation and provenance; a policy gate refuses to assert below the confidence threshold and routes to human review."],
  ["09 — Stability harness. ", "Runs the same queries K times and reports embedding drift, retrieval Jaccard / Kendall-τ, and LLM exact-match / semantic-equivalence / citation-set Jaccard. Emits reports/stability_report.{json,md}."],
];
stageItems.forEach((s) => body.push(bullet([B(s[0]), new TextRun(s[1])])));

// 3. Two run modes
body.push(H1("3. Two run modes"));
body.push(P([B("Mock mode (--mock) — "), "deterministic, no GPU, no network. Synthetic data, hash-seeded embeddings, numpy message-passing for the R-GCN, and a templated strict-RAG explainer. This is what CI and the smoke test use, and what proves the reproducibility claims. Everything is bit-reproducible across machines."]));
body.push(P([B("Real mode (default) — "), "live source APIs; Qwen3-Embedding-0.6B via sentence-transformers; PyTorch Geometric R-GCN; BGE-reranker-v2-m3; a self-hosted 7–8B LLM served by vLLM (OpenAI-compatible) on Drew. Heavy dependencies are imported lazily inside the functions that need them, so the package imports and the mock pipeline runs even when torch / PyG / vLLM are absent."]));
body.push(P(["The seam between modes is a single ", code("mock"), " flag threaded through the embedder factory, the R-GCN trainer, the reranker, and the explainer — so the control flow and data contracts are ", B("identical"), " in both modes. You validate the plumbing in mock, then flip the flag."]));

// 4. Compute placement
body.push(H1("4. Compute placement (Anvil vs Drew)"));
body.push(P([B("The pattern is Anvil = the factory, Drew = the storefront.")]));
body.push(bullet([B("Anvil "), new TextRun("(batch HPC, A100 40 GB / H100 80 GB, ~6,000 GPU-hr allocation): bulk ingestion / parsing / graph-building (CPU, RAM-heavy), embedding millions of docs (04), R-GCN training (06), and any 4B-model LoRA fine-tune. Batch jobs — exactly what a Slurm scheduler is for. The full set of training jobs is comfortably inside the allocation (~1,000–2,000 GPU-hr with re-runs).")]));
body.push(bullet([B("Drew "), new TextRun("(2× 12 GB GPUs, always-on): the persistent retrieval service (07), the multi-agent RAG with a vLLM-served 7–8B model at 4-bit (08), and the demo. 12 GB caps the local LLM to ~7–8B; for anything heavier, call an API or use Anvil's Composable Subsystem for a hosted endpoint.")]));
body.push(P(["Artifacts produced on Anvil (", code("doc_vectors.npy"), ", ", code("faiss.index"), ", ", code("rgcn_node_emb.npy"), ", fine-tuned encoder) are rsync'd / Globus'd to Drew for serving. Because every path is under ", code("$DH_DATA_ROOT"), ", the same code points at Anvil scratch or Drew local disk with no edits."]));

// 5. Determinism
body.push(H1("5. The determinism & stability story (the investor's question)"));
body.push(P([B("\u201CAsk the same question twice — do you get the same output?\u201D"), " The honest, defensible answer is built into the code:"]));
body.push(callout("How to read the answer", [
  P([B("Byte-level determinism is not guaranteed across hardware/library changes"), " — even at temperature 0 — because floating-point addition is non-associative and standard GPU kernels are not batch-invariant (a request's output depends on how many other requests were batched with it). This is real and we don't paper over it."]),
  P([B("Semantic / functional stability is achievable and measured."), " On a fixed stack, embeddings and exact (FAISS Flat) retrieval are deterministic; the evidence and citations a recommendation rests on are stable even if the prose wording varies."], { spacing: { after: 0, line: 276 } }),
]));
body.push(spacer(80));
body.push(P([code("discovery_hub/determinism.py"), " sets every knob we control (Python / NumPy / torch seeds, ", code("torch.use_deterministic_algorithms"), ", deterministic cuDNN, disabled TF32, ", code("CUBLAS_WORKSPACE_CONFIG=:4096:8"), ") and records exactly what was in force. ", code("09_stability_harness.py"), " then measures stability and writes a report whose headline metric is ", B("citation-set Jaccard"), " — the property that makes a recommendation auditable."]));
body.push(P(["On the mock stack the harness reports perfect stability:"]));
body.push(makeTable(
  ["Stage", "Metric", "Mock result", "Ideal"],
  [
    ["Embedding", "max cosine drift", "≈ 6e-8", "~0"],
    ["Embedding", "exact match across runs", "true", "true"],
    ["Retrieval", "top-k set Jaccard", "1.000", "1.000"],
    ["Retrieval", "Kendall's τ (rank)", "1.000", "1.000"],
    ["Explanation", "citation-set Jaccard", "1.000", "1.000"],
    ["Explanation", "semantic equivalence", "1.000", "high"],
  ],
  [1900, 3360, 2100, 2000]));
body.push(spacer(100));
body.push(P(["On a real GPU it surfaces the true drift, honestly. The pitch to the investor is ", B("not"), " \u201Cour model is perfectly deterministic\u201D (false for any GPU LLM). It is \u201Cwe ", B("measure and report"), " stability at every stage, the evidence and citations are reproducible, and we can make even the prose byte-identical with batch-invariant kernels at a throughput cost if a customer requires it.\u201D A response cache keyed on input hash means identical inputs return identical outputs in production — usually the real question being asked."]));

// 6. Repo layout
body.push(H1("6. Repository layout"));
body.push(codeBlock([
  "discovery_hub_pipeline/",
  "├── 01_download_data.py        Layer 1: ingest raw records (mock + real APIs)",
  "├── 02_parse_normalize.py      Layer 1: heterogeneous schemas → DiscoveryDoc",
  "├── 03_build_graph.py          Layer 1: heterogeneous knowledge graph",
  "├── 04_generate_embeddings.py  Layer 2: batch embed (Anvil GPU job)",
  "├── 05_build_index.py          Layer 2: FAISS exact index (→ ship to Drew)",
  "├── 06_train_rgcn.py           Layer 2: R-GCN (PyG real / numpy mock)",
  "├── 07_retrieve_rank.py        Layer 2: two-stage retrieve + rerank (Drew)",
  "├── 08_multiagent_rag.py       Layer 3: multi-agent strict-RAG (Drew)",
  "├── 09_stability_harness.py    QA: reproducibility measurement + report",
  "├── discovery_hub/             shared library (imported by every stage)",
  "│   ├── config.py              profiles, paths, source specs, compute targets",
  "│   ├── schema.py              DiscoveryDoc, graph vocab, JSONL I/O",
  "│   ├── determinism.py         seed/flag control + honest guarantees report",
  "│   ├── embedding.py           MockEmbedder + SentenceTransformerEmbedder",
  "│   └── mock.py                deterministic synthetic-data generators",
  "├── tests/test_pipeline_smoke.py   end-to-end mock test + determinism asserts",
  "├── Makefile                   make mock / make real / make smoke",
  "└── requirements.txt           core (light) + optional GPU deps by stage",
]));

// 7. Quickstart
body.push(H1("7. Quickstart"));
body.push(codeBlock([
  "# install the light core (numpy, scipy, networkx, requests, faiss-cpu, pytest)",
  "pip install -r requirements.txt",
  "",
  "# run the entire pipeline end-to-end with deterministic mocks (no GPU/network)",
  "make mock",
  "",
  "# confirm the determinism assertions pass",
  "make smoke",
  "",
  "# try a single query through the full stack",
  'python 08_multiagent_rag.py --mock --query "EGFR inhibitor for oncology"',
  "",
  "# produce the investor-facing stability report",
  "python 09_stability_harness.py --mock --runs 5   # -> data/reports/stability_report.md",
  "",
  "# --- going real (install the optional deps for the stages you run) ---",
  "# on Anvil:  01 ; 02 ; 03 ; 04 --batch-size 64 ; 05 ; 06 --epochs 20",
  '# on Drew :  07 --query "..." ; 08 --query "..." ; 09 --runs 5',
]));
body.push(P(["Override the data location and master seed with env vars: ", code("DH_DATA_ROOT"), ", ", code("DH_SEED"), ", ", code("DH_EMBED_MODEL"), ", ", code("DH_LLM_MODEL"), "."]));

// 8. Mocked vs production + next steps
body.push(H1("8. What is mocked vs production-ready, and next steps"));
body.push(P([B("Production-ready now: "), "the data contracts (DiscoveryDoc, graph schema, JSONL I/O), the parsers for all five sources, the FAISS index, the two-stage retrieval control flow, the multi-agent state machine, the determinism controls, and the stability harness. These are the same in mock and real mode."]));
body.push(P([B("Mocked (swap in real components): "), "the embedder (→ Qwen3-Embedding-0.6B), the R-GCN trainer (→ PyG RGCNConv with your contrastive / link-prediction objective and neighbor sampling), the reranker (→ BGE-reranker-v2-m3), and the explanation LLM (→ vLLM-served 7–8B model). Each has its real implementation already written and gated; turning it on is installing the dependency and dropping --mock."]));
body.push(H3("Suggested next steps"));
body.push(num(["Pull the real MVP slice (1–5M-work biomedical OpenAlex + pharma-CPC patents + all trials / SBIR) on Anvil and run 01–06 for real; ship the index to Drew."]));
body.push(num(["Wire 07's reranker and 08's explainer to real models; stand up the vLLM server on Drew."]));
body.push(num(["Build the curated 50–500-pair evaluation set and run 09 plus Recall@k / MRR@10 on a fixed GPU to get the first real stability and quality numbers — with confidence intervals — to show the investor."]));
body.push(num(["Map 08's agent functions onto LangGraph nodes once the single-process version is behaving as desired."]));

// ---------- assemble ----------
const doc = new Document({
  creator: "Discovery Hub",
  title: "Discovery Hub Pipeline — Description",
  styles: {
    default: { document: { run: { font: "Arial", size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: INK },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: HEAD_BG, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Arial", color: INK },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Discovery Hub Pipeline   ·   ", size: 18, color: LIGHT }),
        new TextRun({ children: [PageNumber.CURRENT], size: 18, color: LIGHT })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Discovery_Hub_Pipeline_Description.docx", buf);
  console.log("wrote Discovery_Hub_Pipeline_Description.docx", buf.length, "bytes");
});
