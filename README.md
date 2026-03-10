---
title: BioMed Paper Information Extractor
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
python_version: "3.12"
---

# BioMed Paper Information Extractor

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/prockot/biomed-paper-extractor)** | **[GitHub](https://github.com/PR0CK0/biomed-paper-extractor)**

![Demo](https://raw.githubusercontent.com/PR0CK0/biomed-paper-extractor/84b35f8/demo.gif)

An end-to-end pipeline for automated biomedical literature analysis. Accepts a PubMed Central (PMC) ID, PMC article URL, or any supported paper URL and returns two structured outputs: digitized figure data via a Vision Language Model (VLM) and biomedical named entities via a configurable NER pipeline.

Designed for researchers who need structured, machine-readable information extracted from open-access biomedical papers without manual annotation.

---

## Quick Start

**Python 3.11 or 3.12 is required.** Python 3.13+ breaks `blis`, a dependency of the NER stack.

### macOS / Linux

```bash
git clone https://github.com/PR0CK0/biomed-paper-extractor
cd biomed-paper-extractor
make install        # creates .venv with python3.11 and installs all dependencies
make run            # launches app at http://localhost:7860
```

**NVIDIA GPU (CUDA 12.1+)?** Use `make install-cuda` instead of `make install` to get the CUDA-enabled PyTorch build — HuggingFace VLM inference will be dramatically faster:

```bash
make install-cuda   # installs CUDA torch first, then all other dependencies
make run
```

### Windows

Install [Python 3.12](https://www.python.org/downloads/release/python-31210/) and [GNU Make for Windows](https://gnuwin32.sourceforge.net/packages/make.htm) (or via `winget install GnuWin32.Make`), then open a terminal in the repo directory:

```powershell
make install        # creates .venv with py -3.12 and installs all dependencies
make run            # launches app at http://localhost:7860
```

**NVIDIA GPU (CUDA 12.1+)?** Use `make install-cuda` instead:

```powershell
make install-cuda   # installs CUDA torch first, then all other dependencies
make run
```

> First install downloads ~2 GB of scispaCy models and ML dependencies — expect 5–10 minutes.

### API Keys (optional)

To use cloud VLM backends (OpenAI, Anthropic, Google), set the relevant environment variable before launching — or just paste the key into the UI at runtime:

```bash
# macOS / Linux
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GOOGLE_API_KEY = "..."
```

---

## Tasks

### Task 1 — Figure Digitization (VLM)

Each figure extracted from the paper is passed to a selected Vision Language Model with a structured prompt. The model identifies the figure type, reads axis labels, legend entries, and numeric values, and returns a structured JSON object per figure.

An optional **DePlot pre-pass** runs before the VLM on chart-type figures: it extracts the numeric data table from the chart image and injects it into the VLM prompt as additional context, significantly improving accuracy on bar charts, line graphs, and scatter plots.

**Output per figure:**
```json
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
```

### Task 2 — Biomedical Named Entity Recognition (NER)

The full paper text (abstract and body) is processed by a selected NER model. Recognized entities are returned with their surface form, entity type, and — where the model supports it — a canonical concept identifier (e.g. UMLS CUI).

**Output per entity:**
```json
{
  "text": "imatinib",
  "label": "DRUG",
  "umls_cui": "C0935989",
  "start_char": 412,
  "end_char": 420
}
```

---

## Models

### Vision Language Models (Task 1)

> **Note on model size and performance:** Smaller models (sub-4B parameters) are faster on simple figures but frequently stall, time out, or produce incomplete JSON on complex multi-panel figures, dense scatter plots, or figures with overlapping legends. For reliable extraction across a full paper, 7B+ models are strongly recommended. Sub-2B models are best suited for quick tests or simple single-panel images.

| Model | Size | License | Notes |
|---|---|---|---|
| Qwen2-VL 2B Instruct | 2.2B | Apache 2.0 | Fast, low VRAM — may stall on complex figures |
| Qwen2-VL 7B Instruct | 7.6B | Apache 2.0 | Recommended for chart accuracy |
| InternVL2 2B | 2B | MIT | Lightweight — may stall on complex figures |
| InternVL2 4B | 4B | MIT | Balanced quality/speed |
| InternVL2 8B | 8B | MIT | Strong document understanding |
| Phi-3.5-Vision Instruct | 4.2B | MIT | Microsoft multimodal |
| MiniCPM-V 2.6 | 8B | Apache 2.0 | Strong OCR and chart reading |
| Llama-3.2-11B-Vision | 11B | Llama 3.2 (gated) | Requires HF token and access approval |

**API Backends (bring your own key):**

| Provider | Models |
|---|---|
| OpenAI | GPT-4o, GPT-4o Mini |
| Anthropic | Claude 3.5 Sonnet, Claude 3.5 Haiku |
| Google | Gemini 2.0 Flash, Gemini 1.5 Flash |

**Local Inference (Ollama):**

The app queries your local Ollama server for any model that advertises `vision` capability. Below are known-good options:

> **Note:** The same size/complexity tradeoff applies to Ollama models — sub-3B models may produce incomplete or malformed output on complex figures.

| Model | Params | Size | Notes |
|---|---|---|---|
| qwen3-vl:8b | 8.8B | 6.1 GB | Qwen3-VL — latest generation, best quality |
| qwen3-vl:4b | 4.4B | 3.3 GB | Qwen3-VL — good balance of size and accuracy |
| qwen3-vl:2b | 2.1B | 1.9 GB | Qwen3-VL — may struggle with complex figures |
| qwen2.5vl:7b | 8.3B | 6.0 GB | Qwen2.5-VL — strong chart and document understanding |
| qwen2.5vl:3b | 3.8B | 3.2 GB | Qwen2.5-VL — lighter alternative |
| qwen3.5:4b | 4.7B | 3.4 GB | Qwen3.5 general multimodal |
| qwen3.5:2b | 2.3B | 2.7 GB | Qwen3.5 general multimodal |
| qwen3.5:0.8b | 873M | 1.0 GB | Qwen3.5 — sub-1B, very fast; expect stalls on complex images |
| gemma3:12b | 12.2B | 8.1 GB | Google Gemma 3 — highest quality option |
| gemma3:4b | 4.3B | 3.3 GB | Google Gemma 3 — compact |
| ministral-3:3b | 3.8B | 3.0 GB | Mistral 3B multimodal |
| granite3.2-vision:2b | 2.5B | 2.4 GB | IBM Granite 3.2 Vision |

Pull any of these with `ollama pull <model>`. Requires a locally running Ollama server (`ollama serve`); not available on the hosted Spaces instance.

### NER Models (Task 2)

| Model | Type | Entity Types | Notes |
|---|---|---|---|
| scispaCy + UMLS | Rule + Neural | General biomedical | Loads UMLS linker (~3GB RAM); returns UMLS CUIs |
| Triple scispaCy Stack | Rule + Neural | General biomedical | Three-model ensemble for broader coverage |
| d4data DistilBERT | Transformer | 107 types | General-purpose biomedical NER |
| GLiNER (zero-shot) | Zero-shot | Configurable | No task-specific training; define entity types at runtime |
| PubMedBERT Suite | Transformer | Diseases, genes | Two-model pipeline: disease NER + gene/protein NER |
| BioNLP13CG Stack | Transformer | 16 types — cancer, anatomy, cell hierarchy | Cancer genetics corpus; covers organs, tissues, cells, and pathological formations |
| HunFlair2 | Transformer | Genes, diseases, species, chemicals, cell lines | 31-corpus training; multi-domain biomedical |
| Species NER | Transformer | Species/taxa only | LINNAEUS-style species extraction |

---

## Features

### Multi-Provider VLM Support

Switch between local HuggingFace transformer models, Ollama-served local models, and three API providers (OpenAI, Anthropic, Google) from a single interface. API keys are entered per session and never stored.

### DePlot Chart Injection

Toggle a chart-to-table pre-pass that runs Google's DePlot model on each figure before the main VLM. DePlot extracts the underlying numeric data table from chart images and injects it into the VLM prompt. This materially improves extraction accuracy for figures with dense numeric content.

### Evaluation Suite

The **Evaluation** tab benchmarks any VLM and NER combination across a configurable list of PMC papers. Outputs per-paper metrics and a markdown report. Useful for selecting the right model combination for a specific paper type or domain.

### A/B Run Comparison

The **Run Summary** sub-tab (inside Extraction) loads two saved runs side-by-side for direct comparison. The canonical use case is comparing a VLM run with DePlot injection enabled versus disabled to quantify the accuracy delta.

### Output to Disk

Analysis results are written to `output/extraction/` alongside the app. Each run produces a dated subdirectory containing the figure JSON, entity JSON, figure images, metadata, and the full prompt sent to the VLM.

---

## Usage

### Single Paper Analysis

1. Enter a PMC ID (e.g. `PMC8141091`), a full PMC URL, or any supported paper URL in the input field.
2. Select a VLM from the **HuggingFace Models**, **API Backend**, or **Ollama** tab in the model selector.
3. Select a NER model from the NER dropdown.
4. Optionally enable **DePlot injection** for better chart extraction.
5. Click **Run Analysis**.
6. Results appear in the **Figures** and **Entities** tabs. Both are downloadable as JSON.

### Evaluation Across Multiple Papers

1. Open the **Evaluation** tab.
2. Enter a list of PMC IDs (one per line) or use the defaults.
3. Select the VLM and NER models to benchmark.
4. Click **Run Eval**. Progress streams to the log panel.
5. When complete, download the markdown report or view results inline.

### A/B Comparison

1. Run two analyses with different settings (e.g. DePlot on vs. off) and save each run.
2. Open the **Run Summary** sub-tab inside the Extraction tab.
3. Load the two run directories. Summary tables and diff metrics render side-by-side.

---

## Output Structure

```
output/
├── extraction/
│   └── PMC8141091_2024-03-15_14-30-22/
│       ├── task1_figures.json    ← Task 1 VLM output (all figures)
│       ├── task2_ner.json        ← Task 2 NER output (all entities)
│       ├── metadata.json         ← Paper metadata (title, authors, etc.)
│       ├── run_info.json         ← Run configuration and timing
│       ├── run_summary.json      ← Aggregated run metrics
│       ├── prompt.txt            ← Last VLM prompt (for debugging)
│       ├── paper.pdf             ← Source document (or article.html)
│       └── figures/
│           ├── fig1.png          ← Extracted figure images
│           ├── fig2.png
│           └── ...
└── evaluation/
    └── eval_2024-03-15_15-00-00/
        ├── report.md             ← Markdown summary report
        └── eval_result.json      ← Raw per-paper results
```

---

## Deployment Notes and Known Limitations

### Python Version

**Python 3.11 or 3.12 is required.** The HuggingFace Space runs on Python 3.12 (pinned in metadata above); local development uses Python 3.11 (pinned in the Makefile). Python 3.13+ breaks the `blis` dependency used by Flair/HunFlair2 due to removed private CPython APIs.

### Model Size and Inference Speed

Smaller VLMs (sub-4B parameters) are significantly slower on complex figures and may stall indefinitely waiting for a response on multi-panel or data-dense images. This applies to both HuggingFace models and Ollama-served models. If you experience timeouts or incomplete JSON output, switch to a 7B+ model. CPU-only inference with any model above 4B will be very slow (minutes per figure).

### scispaCy UMLS Linker — Memory and Cold Start

The scispaCy UMLS entity linker loads approximately 3GB of concept data into RAM on first use. Cold start on a Spaces free-tier instance takes 30–120 seconds. The linker is held in memory for the session lifetime; subsequent requests are fast.

If you are running on a memory-constrained instance, prefer the d4data DistilBERT or GLiNER NER options, which have no large in-memory knowledge base.

### scispaCy Model Installation

The scispaCy language models (`en_core_sci_lg` and related) are installed directly from S3-hosted wheel URLs at image build time. They are baked into the container and do not require a download at runtime.

### HuggingFace Hub Model Caching

Transformer-based VLMs and NER models are downloaded from the HuggingFace Hub on first use and cached in the standard HF cache directory. On Spaces, this cache persists across restarts if a persistent storage volume is attached; otherwise models re-download on each cold start.

For the 7B and 11B VLMs, first-run download time on Spaces can be 3–10 minutes depending on network conditions.

### Ollama Provider

The Ollama backend connects to a locally running Ollama server. This provider is only functional in self-hosted or local deployments where an Ollama process is accessible at `localhost:11434`. It will report "Ollama not running" on the hosted Spaces instance.

### Gated Models

Llama-3.2-11B-Vision requires a HuggingFace account, an approved access request on the model page, and a valid `HF_TOKEN` environment variable. Set this in your Space secrets if you intend to use it.

### ZeroGPU / Shared GPU

On HuggingFace Spaces with ZeroGPU, the GPU is allocated per request and released after each inference call. Model weights must be reloaded to VRAM on each GPU acquisition unless the Space uses a dedicated GPU tier. Expect 30–60 seconds of GPU warmup on the first figure analysis request in a session.

---

## License

Apache 2.0. Model licenses vary — see the Models section above. Llama 3.2 is subject to Meta's Llama 3.2 Community License.
