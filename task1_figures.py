"""
task1_figures.py — Multi-VLM figure digitization module.

Targets Apple Silicon (MPS) locally and HuggingFace Spaces (ZeroGPU / A10G)
in production. Model is lazy-loaded on first call and cached in module globals.

Supported backends:
- HuggingFace transformers: Qwen2-VL, InternVL2, Phi-3.5-Vision, MiniCPM-V, Llama-3.2-Vision
- Ollama local daemon
- API: OpenAI, Anthropic, Google Gemini
- DePlot pre-pass for chart-to-table context injection
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Optional

import requests
import torch
from PIL import Image

try:
    import spaces
except ImportError:

    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(fn):  # type: ignore[override]
            return fn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_ID: str = os.environ.get("QWEN_MODEL", _DEFAULT_MODEL)

_HF_VLM_REGISTRY: list[tuple[str, str]] = [
    ("SmolVLM 256M  (256M · ~1 GB · Apache 2.0 · CPU-safe)", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    ("SmolVLM 500M  (500M · ~2 GB · Apache 2.0 · CPU-safe)", "HuggingFaceTB/SmolVLM-500M-Instruct"),
    ("Qwen2-VL 2B  (2.2B · ~5 GB · Apache 2.0)", "Qwen/Qwen2-VL-2B-Instruct"),
    ("Qwen2-VL 7B  (7.6B · ~15 GB · Apache 2.0)", "Qwen/Qwen2-VL-7B-Instruct"),
    ("InternVL2 2B  (2B · ~5 GB · MIT)", "OpenGVLab/InternVL2-2B"),
    ("InternVL2 4B  (4B · ~9 GB · MIT)", "OpenGVLab/InternVL2-4B"),
    ("InternVL2 8B  (8B · ~17 GB · MIT)", "OpenGVLab/InternVL2-8B"),
    ("Phi-3.5-Vision  (4.2B · ~9 GB · MIT)", "microsoft/Phi-3.5-vision-instruct"),
    ("MiniCPM-V 2.6  (8B · ~16 GB · Apache)", "openbmb/MiniCPM-V-2_6"),
    ("Llama-3.2-11B-Vision  (11B · ~22 GB · gated)", "meta-llama/Llama-3.2-11B-Vision-Instruct"),
]

# API model choices per provider
_API_OPENAI_CHOICES: list[tuple[str, str]] = [
    ("GPT-4o  (best quality)", "openai/gpt-4o"),
    ("GPT-4o mini  (fast · cheap)", "openai/gpt-4o-mini"),
]
_API_ANTHROPIC_CHOICES: list[tuple[str, str]] = [
    ("Claude 3.5 Sonnet  (best quality)", "anthropic/claude-3-5-sonnet-20241022"),
    ("Claude 3.5 Haiku  (fast · cheap)", "anthropic/claude-3-5-haiku-20241022"),
]
_API_GOOGLE_CHOICES: list[tuple[str, str]] = [
    ("Gemini 2.0 Flash  (fast · free tier)", "google/gemini-2.0-flash"),
    ("Gemini 1.5 Flash  (stable)", "google/gemini-1.5-flash"),
]

PROMPT = """Analyze this scientific figure. It may be a single panel or a compound figure with multiple labeled panels (e.g., "(a)", "(b)", "(c)").

Return ONLY a valid JSON object with this exact structure:
{
  "is_compound": false,
  "panels": [
    {
      "panel_id": "main",
      "figure_type": "bar|line|scatter|table|image|diagram|other",
      "title": "",
      "x_axis": {"label": "", "units": "", "ticks": []},
      "y_axis": {"label": "", "units": "", "ticks": []},
      "legend": [],
      "data_points": [],
      "notes": ""
    }
  ]
}

Rules:
- If the figure has labeled sub-panels (a, b, c, etc.), set "is_compound" to true and add one entry per panel in "panels", using the label as "panel_id".
- If it is a single figure, set "is_compound" to false and use one entry with "panel_id": "main".
- "data_points" should capture numeric values, coordinates, or table rows visible in the panel.
- Return ONLY the JSON object. No explanation, no markdown fences, no extra text."""

_OLLAMA_BASE = "http://localhost:11434"
_OLLAMA_PREFIX = "ollama/"
OLLAMA_TIMEOUT: int = 180   # seconds per request
OLLAMA_MAX_RETRIES: int = 1  # extra attempts after first timeout


def set_ollama_timeout(seconds: int) -> None:
    global OLLAMA_TIMEOUT
    OLLAMA_TIMEOUT = max(10, int(seconds))


def set_ollama_retries(n: int) -> None:
    global OLLAMA_MAX_RETRIES
    OLLAMA_MAX_RETRIES = max(0, min(5, int(n)))


def check_ollama() -> bool:
    """Return True if an Ollama daemon is reachable at localhost:11434."""
    try:
        resp = requests.get(f"{_OLLAMA_BASE}/api/tags", timeout=2)
        return resp.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------

import threading as _threading
_load_lock = _threading.Lock()  # serialises concurrent load_vlm calls

_model: Optional[object] = None
_processor: Optional[object] = None
_ollama_model: Optional[str] = None  # set when active backend is Ollama

# API backend state
_api_provider: Optional[str] = None   # "openai", "anthropic", "google"
_api_model_name: Optional[str] = None  # e.g. "gpt-4o", "claude-3-5-sonnet-20241022"
_api_key: Optional[str] = None

# DePlot state
_deplot_model: Optional[object] = None
_deplot_processor: Optional[object] = None
DEPLOT_ENABLED: bool = False
_last_full_prompt: str = ""

# Active model family for HF dispatch
_model_family_active: str = "qwen2vl"


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _get_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype(device: str) -> torch.dtype:
    return torch.float16 if device in ("cuda", "mps") else torch.float32


# ---------------------------------------------------------------------------
# Model family routing
# ---------------------------------------------------------------------------


def _model_family(model_id: str) -> str:
    """Return a canonical family string for the given model ID."""
    if "SmolVLM" in model_id:
        return "smolvlm"
    if "InternVL2" in model_id or "InternVL2_5" in model_id:
        return "internvl2"
    if "Phi-3.5-vision" in model_id or "Phi-3-vision" in model_id:
        return "phi35"
    if "MiniCPM" in model_id:
        return "minicpm"
    if "Llama-3.2" in model_id and "Vision" in model_id:
        return "llama32"
    return "qwen2vl"  # default


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------


def _load_model() -> tuple[object, object]:
    """Load and cache the active HF VLM + processor/tokenizer on first call."""
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    device = _get_device()
    dtype = _get_dtype(device)
    family = _model_family_active

    print(f"[task1_figures] Loading {MODEL_ID} (family={family}) on device={device} dtype={dtype}")

    if family == "smolvlm":
        from transformers import AutoModelForVision2Seq, AutoProcessor

        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        if device == "cuda":
            _model = AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            _model = AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
            ).to(device)

    elif family == "internvl2":
        from transformers import AutoModel, AutoTokenizer

        if device == "cuda":
            _model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            _model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)  # type: ignore[union-attr]

        _processor = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    elif family == "phi35":
        from transformers import AutoModelForCausalLM, AutoProcessor

        if device == "cuda":
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                _attn_implementation="eager",
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
                _attn_implementation="eager",
            ).to(device)  # type: ignore[union-attr]

        _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    elif family == "minicpm":
        from transformers import AutoModel, AutoTokenizer

        if device == "cuda":
            _model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            _model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)  # type: ignore[union-attr]

        _processor = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    elif family == "llama32":
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        # Llama-3.2-11B is large; always use device_map="auto" for memory safety
        _model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )
        _processor = AutoProcessor.from_pretrained(MODEL_ID)

    else:
        # qwen2vl — original path unchanged
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if device == "cuda":
            _model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            _model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            ).to(device)  # type: ignore[union-attr]

        _processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[task1_figures] Model loaded.")
    return _model, _processor


def load_vlm(model_id: str, api_key: str | None = None) -> None:
    """Load (or reload) the VLM, switching models if necessary.

    model_id can be:
    - A HuggingFace model ID (e.g. "Qwen/Qwen2-VL-2B-Instruct")
    - An Ollama model prefixed with "ollama/" (e.g. "ollama/qwen2-vl:2b")
    - An OpenAI model prefixed with "openai/" (e.g. "openai/gpt-4o")
    - An Anthropic model prefixed with "anthropic/" (e.g. "anthropic/claude-3-5-sonnet-20241022")
    - A Google model prefixed with "google/" (e.g. "google/gemini-2.0-flash")

    Serialised by _load_lock so concurrent calls queue rather than race.
    """
    with _load_lock:
        global _model, _processor, MODEL_ID, _ollama_model
        global _api_provider, _api_model_name, _api_key
        global _model_family_active

        # --- Ollama path ---
        if model_id.startswith(_OLLAMA_PREFIX):
            ollama_name = model_id[len(_OLLAMA_PREFIX):]
            if not check_ollama():
                raise RuntimeError(
                    "Ollama daemon not reachable at localhost:11434. "
                    "Start it with: ollama serve"
                )
            # Unload any HF model that was loaded.
            if _model is not None:
                print(f"[task1_figures] Unloading HF model {MODEL_ID} for Ollama switch...")
                del _model, _processor
                _model = None
                _processor = None
                device = _get_device()
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()
            _api_provider = None
            _api_model_name = None
            _api_key = None
            _ollama_model = ollama_name
            MODEL_ID = model_id
            print(f"[task1_figures] Ollama backend active: {ollama_name}")
            return

        # --- API paths ---
        def _switch_to_api(provider: str, name: str) -> None:
            global _api_provider, _api_model_name, _api_key, _ollama_model, MODEL_ID
            global _model, _processor
            _api_provider = provider
            _api_model_name = name
            _api_key = api_key
            _ollama_model = None
            MODEL_ID = model_id
            if _model is not None:
                print(f"[task1_figures] Unloading HF model {MODEL_ID} for API switch ({provider})...")
                del _model, _processor
                _model = None
                _processor = None
                device = _get_device()
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()
            print(f"[task1_figures] API backend active: {provider}/{name}")

        if model_id.startswith("openai/"):
            _switch_to_api("openai", model_id[len("openai/"):])
            return

        if model_id.startswith("anthropic/"):
            _switch_to_api("anthropic", model_id[len("anthropic/"):])
            return

        # google/ prefix that is NOT google/deplot goes to API
        if model_id.startswith("google/") and model_id != "google/deplot":
            _switch_to_api("google", model_id[len("google/"):])
            return

        # --- HuggingFace path ---
        _api_provider = None
        _api_model_name = None
        _api_key = None
        _ollama_model = None

        if _model is not None and model_id == MODEL_ID:
            return

        if _model is not None:
            print(f"[task1_figures] Unloading {MODEL_ID}...")
            del _model
            del _processor
            _model = None
            _processor = None
            device = _get_device()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        MODEL_ID = model_id
        _model_family_active = _model_family(model_id)
        _load_model()


# ---------------------------------------------------------------------------
# DePlot pre-pass
# ---------------------------------------------------------------------------


def load_deplot() -> None:
    """Load google/deplot for chart-to-table pre-pass. Cached in module globals."""
    global _deplot_model, _deplot_processor
    if _deplot_model is not None:
        return
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    device = _get_device()
    dtype = _get_dtype(device)
    print("[task1_figures] Loading google/deplot...")
    _deplot_processor = Pix2StructProcessor.from_pretrained("google/deplot")
    _deplot_model = Pix2StructForConditionalGeneration.from_pretrained(
        "google/deplot", torch_dtype=dtype
    ).to(device)
    print("[task1_figures] DePlot loaded.")


def unload_deplot() -> None:
    """Unload DePlot model to free memory."""
    global _deplot_model, _deplot_processor, DEPLOT_ENABLED
    if _deplot_model is not None:
        del _deplot_model, _deplot_processor
        _deplot_model = None
        _deplot_processor = None
        device = _get_device()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    DEPLOT_ENABLED = False
    print("[task1_figures] DePlot unloaded.")


def toggle_deplot(enabled: bool) -> None:
    """Enable or disable DePlot pre-pass. Loads model on first enable."""
    global DEPLOT_ENABLED
    if enabled and _deplot_model is None:
        load_deplot()
    DEPLOT_ENABLED = enabled
    print(f"[task1_figures] DePlot pre-pass: {'enabled' if enabled else 'disabled'}")


def _run_deplot(pil_img: Image.Image) -> str:
    """Run DePlot on one image. Returns raw table string, or '' on failure."""
    if _deplot_model is None or _deplot_processor is None:
        return ""
    try:
        device = _get_device()
        inputs = _deplot_processor(
            images=pil_img,
            text="Generate underlying data table of the figure below:",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            preds = _deplot_model.generate(**inputs, max_new_tokens=512)  # type: ignore[union-attr]
        return _deplot_processor.decode(preds[0], skip_special_tokens=True)  # type: ignore[union-attr]
    except Exception as exc:
        print(f"[task1_figures] DePlot error: {exc}")
        return ""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json(raw: str) -> dict:
    """
    Attempt to parse a JSON dict from raw model output.

    Strategy (in order):
    1. Direct json.loads on stripped text.
    2. Extract first {...} block via regex and parse.
    3. Strip markdown fences (```json ... ```) and parse.
    4. Return error dict as last resort.
    """
    text = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 3. Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    return {"error": "Failed to parse JSON", "raw": raw}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(deplot_table: str = "") -> str:
    """Return PROMPT optionally prefixed with DePlot-extracted table context."""
    global _last_full_prompt
    if not deplot_table:
        _last_full_prompt = PROMPT
        return PROMPT
    result = (
        f"Here is a machine-extracted data table from this figure (may be incomplete):\n\n"
        f"{deplot_table}\n\n"
        f"Use this as reference context, but do not trust it blindly — verify against the image.\n\n"
        f"{PROMPT}"
    )
    _last_full_prompt = result
    return result


def get_last_prompt() -> str:
    """Return the last full prompt string that was sent to any VLM backend."""
    return _last_full_prompt


# ---------------------------------------------------------------------------
# Inference — single image — per family
# ---------------------------------------------------------------------------


def _analyze_single_smolvlm(
    pil_img: Image.Image,
    model: object,
    processor: object,
    device: str,
    deplot_table: str = "",
) -> dict:
    """Run SmolVLM inference on one PIL image and return parsed dict."""
    prompt_text = _build_prompt(deplot_table)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[pil_img], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Strip the prompt echo that some SmolVLM versions include
    if "Assistant:" in raw:
        raw = raw.split("Assistant:")[-1].strip()
    return _parse_json(raw)


def _analyze_single(
    pil_img: Image.Image,
    model: object,
    processor: object,
    device: str,
    deplot_table: str = "",
) -> dict:
    """Run Qwen2-VL inference on one PIL image and return parsed dict."""
    from qwen_vl_utils import process_vision_info

    prompt_text = _build_prompt(deplot_table)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(  # type: ignore[union-attr]
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(  # type: ignore[union-attr]
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048)  # type: ignore[union-attr]

    # Trim prompt tokens from output
    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(  # type: ignore[union-attr]
        output_ids,
        skip_special_tokens=True,
    )[0]

    return _parse_json(raw)


def _analyze_single_internvl2(
    pil_img: Image.Image,
    model: object,
    tokenizer: object,
    device: str,
    deplot_table: str = "",
) -> dict:
    """Run InternVL2 inference on one PIL image."""
    import numpy as np

    prompt_text = _build_prompt(deplot_table)

    # InternVL2 uses a specific pixel_values format
    IMG_SIZE = 448
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    pixel_values = torch.tensor(
        np.array(img).transpose(2, 0, 1), dtype=torch.float16
    ).unsqueeze(0).to(device) / 255.0

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16, device=device).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std

    question = f"<image>\n{prompt_text}"
    generation_config = {"max_new_tokens": 2048, "do_sample": False}

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, question, generation_config)  # type: ignore[union-attr]

    return _parse_json(response)


def _analyze_single_phi35(
    pil_img: Image.Image,
    model: object,
    processor: object,
    device: str,
    deplot_table: str = "",
) -> dict:
    """Run Phi-3.5-Vision inference on one PIL image."""
    prompt_text = _build_prompt(deplot_table)
    messages = [{"role": "user", "content": "<|image_1|>\n" + prompt_text}]
    prompt = processor.tokenizer.apply_chat_template(  # type: ignore[union-attr]
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, [pil_img], return_tensors="pt").to(device)  # type: ignore[union-attr]

    with torch.no_grad():
        output_ids = model.generate(  # type: ignore[union-attr]
            **inputs,
            max_new_tokens=2048,
            eos_token_id=processor.tokenizer.eos_token_id,  # type: ignore[union-attr]
        )
    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]  # type: ignore[union-attr]
    return _parse_json(raw)


def _analyze_single_minicpm(
    pil_img: Image.Image,
    model: object,
    tokenizer: object,
    device: str,
    deplot_table: str = "",
) -> dict:
    """Run MiniCPM-V-2_6 inference on one PIL image."""
    prompt_text = _build_prompt(deplot_table)
    msgs = [{"role": "user", "content": [pil_img, prompt_text]}]

    with torch.no_grad():
        response = model.chat(  # type: ignore[union-attr]
            image=None,  # images passed in msgs
            msgs=msgs,
            tokenizer=tokenizer,
        )
    return _parse_json(response)


def _analyze_single_llama32(
    pil_img: Image.Image,
    model: object,
    processor: object,
    device: str,
    deplot_table: str = "",
) -> dict:
    """Run Llama-3.2-Vision inference on one PIL image."""
    prompt_text = _build_prompt(deplot_table)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)  # type: ignore[union-attr]
    inputs = processor(pil_img, input_text, return_tensors="pt").to(device)  # type: ignore[union-attr]

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048)  # type: ignore[union-attr]
    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]  # type: ignore[union-attr]
    return _parse_json(raw)


def _analyze_single_dispatch(
    pil_img: Image.Image,
    model: object,
    processor_or_tokenizer: object,
    device: str,
) -> dict:
    """Dispatch to the correct inference function based on the active model family."""
    deplot_table = _run_deplot(pil_img) if DEPLOT_ENABLED else ""

    family = _model_family_active
    if family == "smolvlm":
        return _analyze_single_smolvlm(pil_img, model, processor_or_tokenizer, device, deplot_table)
    elif family == "internvl2":
        return _analyze_single_internvl2(pil_img, model, processor_or_tokenizer, device, deplot_table)
    elif family == "phi35":
        return _analyze_single_phi35(pil_img, model, processor_or_tokenizer, device, deplot_table)
    elif family == "minicpm":
        return _analyze_single_minicpm(pil_img, model, processor_or_tokenizer, device, deplot_table)
    elif family == "llama32":
        return _analyze_single_llama32(pil_img, model, processor_or_tokenizer, device, deplot_table)
    else:
        return _analyze_single(pil_img, model, processor_or_tokenizer, device, deplot_table)


# ---------------------------------------------------------------------------
# Inference — single image — Ollama
# ---------------------------------------------------------------------------


def _analyze_single_ollama(pil_img: Image.Image, model_name: str) -> dict:
    """Run inference via a local Ollama daemon and return parsed dict.

    Retries up to OLLAMA_MAX_RETRIES times on ReadTimeout before raising.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
        "stream": False,
    }

    last_exc: Exception | None = None
    attempts = OLLAMA_MAX_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(f"[task1_figures] Ollama retry {attempt - 1}/{OLLAMA_MAX_RETRIES} (timeout={OLLAMA_TIMEOUT}s)")
            resp = requests.post(
                f"{_OLLAMA_BASE}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            print(f"[task1_figures] Ollama raw response length: {len(raw)} chars")
            return _parse_json(raw)
        except requests.exceptions.ReadTimeout as exc:
            last_exc = exc
            print(f"[task1_figures] Ollama timed out (attempt {attempt}/{attempts}, limit={OLLAMA_TIMEOUT}s)")
        except Exception as exc:
            raise exc  # non-timeout errors bubble immediately

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Inference — single image — API backends
# ---------------------------------------------------------------------------


def _analyze_single_openai(
    pil_img: Image.Image,
    model_name: str,
    api_key: str,
    deplot_table: str = "",
) -> dict:
    """Run OpenAI vision inference on one PIL image."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install openai: pip install openai")

    prompt_text = _build_prompt(deplot_table)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content or ""
    return _parse_json(raw)


def _analyze_single_anthropic(
    pil_img: Image.Image,
    model_name: str,
    api_key: str,
    deplot_table: str = "",
) -> dict:
    """Run Anthropic vision inference on one PIL image."""
    try:
        import anthropic as anthropic_sdk
    except ImportError:
        raise RuntimeError("Install anthropic: pip install anthropic")

    prompt_text = _build_prompt(deplot_table)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = anthropic_sdk.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model_name,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    )
    raw = msg.content[0].text if msg.content else ""
    return _parse_json(raw)


def _analyze_single_google(
    pil_img: Image.Image,
    model_name: str,
    api_key: str,
    deplot_table: str = "",
) -> dict:
    """Run Google Gemini vision inference on one PIL image."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("Install google-genai: pip install google-genai")

    prompt_text = _build_prompt(deplot_table)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[pil_img, prompt_text],
    )
    raw = response.text or ""
    return _parse_json(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@spaces.GPU
def _analyze_figures_hf(images: list[Image.Image], fig_offset: int = 0, total_figs: int | None = None) -> list[dict]:
    """HuggingFace transformers path (runs on ZeroGPU in HF Spaces)."""
    model, processor = _load_model()
    device = _get_device()
    n = total_figs if total_figs is not None else len(images)
    results: list[dict] = []
    for idx, img in enumerate(images):
        abs_idx = fig_offset + idx
        figure_id = f"fig{abs_idx + 1}"
        print(f"[task1_figures] HF processing {figure_id} ({abs_idx + 1}/{n})")
        try:
            parsed = _analyze_single_dispatch(img, model, processor, device)
        except Exception as exc:
            print(f"[task1_figures] Error on {figure_id}: {exc}")
            parsed = {"error": str(exc)}
        results.append({"figure_id": figure_id, **parsed})
    return results


def _analyze_figures_ollama(images: list[Image.Image], model_name: str, fig_offset: int = 0, total_figs: int | None = None) -> list[dict]:
    """Ollama local daemon path."""
    n = total_figs if total_figs is not None else len(images)
    results: list[dict] = []
    for idx, img in enumerate(images):
        abs_idx = fig_offset + idx
        figure_id = f"fig{abs_idx + 1}"
        print(f"[task1_figures] Ollama processing {figure_id} ({abs_idx + 1}/{n}) via {model_name}")
        try:
            parsed = _analyze_single_ollama(img, model_name)
        except Exception as exc:
            print(f"[task1_figures] Error on {figure_id}: {exc}")
            parsed = {"error": str(exc)}
        results.append({"figure_id": figure_id, **parsed})
    return results


def _analyze_figures_api(images: list[Image.Image], fig_offset: int = 0, total_figs: int | None = None) -> list[dict]:
    """API backend path (OpenAI / Anthropic / Google)."""
    if _api_provider is None or _api_model_name is None or _api_key is None:
        raise RuntimeError(
            "[task1_figures] API backend not configured. "
            "Call load_vlm('openai/model-name', api_key='...')"
        )

    n = total_figs if total_figs is not None else len(images)
    results: list[dict] = []
    for idx, img in enumerate(images):
        abs_idx = fig_offset + idx
        figure_id = f"fig{abs_idx + 1}"
        print(
            f"[task1_figures] API ({_api_provider}/{_api_model_name}) "
            f"processing {figure_id} ({abs_idx + 1}/{n})"
        )
        deplot_table = _run_deplot(img) if DEPLOT_ENABLED else ""
        try:
            if _api_provider == "openai":
                parsed = _analyze_single_openai(img, _api_model_name, _api_key, deplot_table)
            elif _api_provider == "anthropic":
                parsed = _analyze_single_anthropic(img, _api_model_name, _api_key, deplot_table)
            elif _api_provider == "google":
                parsed = _analyze_single_google(img, _api_model_name, _api_key, deplot_table)
            else:
                parsed = {"error": f"Unknown API provider: {_api_provider}"}
        except Exception as exc:
            parsed = {"error": str(exc)}
        results.append({"figure_id": figure_id, **parsed})
    return results


def analyze_figures(images: list[Image.Image], fig_offset: int = 0, total_figs: int | None = None) -> list[dict]:
    """Digitize a list of scientific figure images.

    Routes to:
    - Ollama local daemon if an Ollama model is active
    - API backend (OpenAI / Anthropic / Google) if configured
    - HuggingFace transformers backend otherwise (ZeroGPU-compatible)

    fig_offset / total_figs are forwarded to the backend for accurate progress logging
    when the caller processes images one-at-a-time in a loop.
    """
    if _ollama_model is not None:
        return _analyze_figures_ollama(images, _ollama_model, fig_offset, total_figs)
    if _api_provider is not None:
        return _analyze_figures_api(images, fig_offset, total_figs)
    return _analyze_figures_hf(images, fig_offset, total_figs)
