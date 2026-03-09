# LLM PDF Extractor — Implementation Plan

## Stack Overview

| Component | Tool | Reason |
|---|---|---|
| PMC Fetching | `requests` + `BeautifulSoup4` | Simple, no auth needed |
| Figure Extraction | `PyMuPDF` (fitz) | Best-in-class PDF image extraction |
| VLM Inference | Qwen2-VL 7B via HuggingFace | Strong chart/figure reasoning, ZeroGPU compatible |
| NER | scispaCy `en_core_sci_lg` + UMLS Entity Linker | Biomedical-specific, CPU-friendly |
| Frontend | Gradio | Native HuggingFace Spaces support |
| Hosting | HuggingFace Spaces (ZeroGPU) | Free GPU, public link, GitHub sync |
| Output | JSON (both tasks) | Displayed in UI + downloadable |

---

## Hosting Setup

### HuggingFace Spaces
- Create a Space at huggingface.co/spaces — free, public
- Enables ZeroGPU (shared A10G, spins up per request, no idle cost)
- Public URL: `https://huggingface.co/spaces/{username}/{space-name}`

### GitHub Integration
- Code lives on GitHub (your repo)
- HF Spaces syncs from GitHub via Spaces > Settings > "Link to GitHub repo"
- Push to main → auto-deploys to your Space
- Workflow: develop locally → push to GitHub → Space rebuilds automatically

---

## Project Structure

```
pdf-extractor/
├── app.py                  ← Gradio UI (HF Spaces entry point)
├── fetch_paper.py          ← PMC fetcher (by PMC ID)
├── task1_figures.py        ← figure extraction + Qwen2-VL digitization
├── task2_ner.py            ← scispaCy NER on paper text
├── requirements.txt        ← HF Spaces dependencies
└── README.md               ← HF Spaces config header + usage docs
```

No `output/` directory — all results returned in-memory to the Gradio UI.

---

## app.py — Gradio UI

Single interface, two output tabs.

**Inputs:**
- PMC ID text field (e.g. `PMC7614754`)
- Submit button

**Outputs (tabs):**
- **Figures** — JSON of digitized figure data (Task 1)
- **Entities** — JSON of detected biomedical entities (Task 2)
- Both tabs include a download button for the JSON

**Pipeline on submit:**
1. `fetch_paper.py` → fetch text + figure images from PMC
2. `task1_figures.py` → Qwen2-VL analysis of each figure (GPU)
3. `task2_ner.py` → scispaCy NER on full text (CPU)
4. Return both JSON outputs to tabs

---

## fetch_paper.py

- Accept PMC ID as argument
- Fetch HTML from `https://pmc.ncbi.nlm.nih.gov/articles/{PMC_ID}/`
- Extract:
  - Full article text (sections, paragraphs)
  - Figure image URLs → download to memory (not disk)
- Attempt PDF download via PMC's PDF link as fallback for figure extraction
- Returns: `(full_text: str, figures: list[PIL.Image])`

---

## Task 1 — Figure Digitization (task1_figures.py)

### Steps
1. Accept list of `PIL.Image` objects from fetcher
2. Pass each to **Qwen2-VL 7B** with structured prompt:
   ```
   Analyze this scientific figure and return ONLY a JSON object with:
   - figure_type (bar, line, scatter, table, etc.)
   - title
   - x_axis: {label, units, ticks}
   - y_axis: {label, units, ticks}
   - legend: [series names]
   - data_points: [...]
   - notes (any observations about the figure)
   ```
3. Parse and validate JSON output; include raw response on parse failure
4. Return list of figure result dicts

### Output Shape
```json
[
  {
    "figure_id": "fig1",
    "figure_type": "bar_chart",
    "title": "...",
    "x_axis": { "label": "...", "units": "...", "ticks": [] },
    "y_axis": { "label": "...", "units": "...", "ticks": [] },
    "legend": [],
    "data_points": [],
    "notes": "..."
  }
]
```

### GPU Usage
- Wrapped with `@spaces.GPU` decorator (HF ZeroGPU)
- Model loaded lazily on first GPU request
- Runs on shared A10G — expect 30–60s cold start, ~5–10s per figure after warmup

### Why Qwen2-VL?
- Best open-source VLM for document/chart understanding as of early 2025
- 7B fits on A10G (24GB VRAM)
- HuggingFace native — loads directly from Hub, no manual download

---

## Task 2 — Named Entity Recognition (task2_ner.py)

### Steps
1. Accept full paper text string
2. Load **scispaCy** pipeline: `en_core_sci_lg`
3. Add UMLS Entity Linker for concept normalization
4. Run NER, extract entities with:
   - `text` — raw entity string
   - `label` — entity type (DRUG, GENE, DISEASE, etc.)
   - `umls_cui` — UMLS Concept ID
   - `start_char`, `end_char` — position in text
5. Return list of entity dicts

### Output Shape
```json
[
  {
    "text": "imatinib",
    "label": "DRUG",
    "umls_cui": "C0935989",
    "start_char": 412,
    "end_char": 420
  }
]
```

### Why scispaCy?
- Purpose-built for biomedical text
- `en_core_sci_lg` trained on MedMentions, BC5CDR
- UMLS linker gives structured concept IDs
- CPU-friendly — no GPU needed, runs in parallel with Task 1

---

## README.md (HF Spaces config)

Must include YAML frontmatter at the top for HF Spaces to detect the app:

```yaml
---
title: PDF Figure & Entity Extractor
emoji:
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.x"
app_file: app.py
pinned: false
---
```

---

## requirements.txt

```
gradio
requests
beautifulsoup4
pymupdf
transformers
accelerate
torch
qwen-vl-utils
scispacy
https://s3-amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
spaces
```

---

## Deployment Workflow

1. Create HuggingFace account + new Space (Gradio, public, ZeroGPU hardware)
2. Create GitHub repo, push code
3. In Space settings: link to GitHub repo + branch
4. HF auto-builds on push — share the Space URL

---

## Notes

- All results in-memory — no disk writes in the hosted version
- JSON parse errors from VLM are caught and returned as `{"error": "...", "raw": "..."}`
- UMLS linker download (~1GB) happens at Space startup — cold start will be slow the first time
- PMC ID input must work with any valid ID — no hardcoded papers
