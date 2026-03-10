from __future__ import annotations

import json
import sys
import warnings
from contextlib import contextmanager
from typing import Any

# ---------------------------------------------------------------------------
# Patch accelerate.init_empty_weights at import time (before any threads start).
#
# On Windows + PyTorch 2.x + transformers >=4.50, AutoModel.from_config() uses
# accelerate.init_empty_weights() which initialises the backbone on the meta
# device.  GLiNER's from_pretrained() then calls .to("cpu") which raises
# "Cannot copy out of meta tensor".  Replacing init_empty_weights with a no-op
# means the backbone initialises directly on CPU.  Harmless on Mac/Linux/HF.
# Must run at module level so the patch is in place before any concurrent
# model-loading threads start.
# ---------------------------------------------------------------------------
try:
    import accelerate

    @contextmanager
    def _noop_init(*args, **kwargs):
        yield

    accelerate.init_empty_weights = _noop_init
    if hasattr(accelerate, "hooks"):
        accelerate.hooks.init_empty_weights = _noop_init
except Exception:
    pass

try:
    import spaces
except ImportError:

    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(fn):  # type: ignore[override]
            return fn


def _ner_device() -> int:
    """Return 0 (CUDA GPU 0) if available, else -1 (CPU).

    HF pipeline accepts device=0 for GPU or device=-1 for CPU.
    MPS (Apple Silicon) is mapped to CPU here because HF pipeline's MPS
    support is inconsistent across models — CPU is safer.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1

# Suppress benign warnings from scispaCy's pickled UMLS linker (sklearn version
# mismatch) and spaCy's tokenizer deserializer (FutureWarning). Both are cosmetic.
warnings.filterwarnings("ignore", category=FutureWarning, module="spacy")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Public metadata
# ---------------------------------------------------------------------------

NER_MODEL_OPTIONS: dict[str, str] = {
    "scispacy_umls": "scispaCy + UMLS (en_core_sci_lg)",
    "scispacy_triple": "Triple scispaCy Stack (BC5CDR + CRAFT + JNLPBA)",
    "d4data": "d4data/biomedical-ner-all (DistilBERT, 107 types)",
    "gliner": "GLiNER zero-shot (gliner-biomed-base-v1.0)",
    "pubmedbert": "PubMedBERT NER Suite (disease + gene)",
    "scispacy_bionlp13cg": "BioNLP13CG Cancer Genetics Stack (16 types · cancer + anatomy)",
    "hunflair2": "HunFlair2  (31-corpus · Disease + Chemical + Gene + Species + CellLine)",
    "species_ner": "Species NER (PubMedBERT · Species-800 · organism recognition)",
}

NER_MODEL_INFO: dict[str, dict] = {
    "scispacy_umls": {
        "description": (
            "scispaCy's en_core_sci_lg model detects entity spans across all biomedical "
            "entity types using a NER model trained on MedMentions. Detected spans are then "
            "linked to UMLS (~3M concepts, 200+ vocabularies) using TF-IDF approximate nearest "
            "neighbor search. Entity type labels are derived from UMLS semantic types (TUIs)."
        ),
        "labels": [
            "Disease", "Drug", "Gene", "Protein", "Species", "CellLine", "CellType",
            "Anatomy", "BiologicalProcess", "MolecularFunction", "Device", "ENTITY (fallback)",
        ],
        "size": "en_core_sci_lg: ~800MB + UMLS index: ~3GB RAM",
        "trained_on": "MedMentions (UMLS-annotated PubMed abstracts), UMLS 2023 release",
    },
    "scispacy_triple": {
        "description": (
            "Runs three specialized scispaCy NER models in parallel and merges their entity "
            "spans: (1) BC5CDR for diseases and chemicals, (2) CRAFT for genes, species, cell "
            "types, chemical entities, and GO terms, (3) JNLPBA for molecular biology entities "
            "including DNA, RNA, proteins, and cell lines. Overlapping spans resolved by "
            "first-occurrence priority."
        ),
        "labels": [
            "Disease", "Chemical", "Gene", "Species", "CellType", "CellLine",
            "Protein", "DNA", "RNA", "BiologicalProcess", "SequenceFeature",
        ],
        "size": "3 × ~85MB models",
        "trained_on": (
            "BC5CDR (1,500 PubMed abstracts), CRAFT (97 full-text PMC papers), "
            "JNLPBA (2,000 MEDLINE abstracts)"
        ),
    },
    "d4data": {
        "description": (
            "DistilBERT-based model fine-tuned on MACCROBAT2018 (biomedical case reports). "
            "Predicts 107 entity types covering clinical findings, medications, procedures, "
            "anatomy, demographics, and biological entities. Faster than full BERT due to "
            "DistilBERT architecture (40% smaller, 60% faster)."
        ),
        "labels": [
            "Disease_disorder", "Sign_symptom", "Medication", "Dosage",
            "Biological_structure", "Diagnostic_or_laboratory_procedure",
            "Therapeutic_procedure", "Gene", "Protein", "Organism", "Lab_value",
            "Age", "Sex", "... (107 total types)",
        ],
        "size": "~265MB (DistilBERT-base)",
        "trained_on": "MACCROBAT2018 biomedical case report corpus",
    },
    "gliner": {
        "description": (
            "GLiNER (Generalist Lightweight NER) is a zero-shot NER model — you specify entity "
            "type labels in natural language at inference time, no retraining required. Uses a "
            "bi-encoder architecture on DeBERTa-v3 to match text spans against entity type "
            "descriptions. The biomedical variant (gliner-biomed) was distilled from large "
            "biomedical generative LMs and achieves SOTA zero-shot biomedical NER (NAACL 2024)."
        ),
        "labels": [
            "User-defined — defaults: Drug, Gene, Disease, Protein, Species, CellLine, "
            "CellType, DNA, RNA, Mutation, Anatomy, BiologicalProcess, MolecularFunction, Chemical"
        ],
        "size": "~200MB (DeBERTa-v3-base backbone)",
        "trained_on": (
            "Distilled from large biomedical LMs; general NER corpora + synthetic biomedical "
            "annotations"
        ),
    },
    "pubmedbert": {
        "description": (
            "Two PubMedBERT-based NER pipelines merged: (1) alvaroalon2/biobert_diseases_ner "
            "for disease entities, (2) pruas/BENT-PubMedBERT-NER-Gene for gene/protein entities. "
            "Both use aggregation_strategy='simple'. Spans deduplicated by (start, end) with "
            "disease pipeline taking precedence on conflicts."
        ),
        "labels": ["Disease", "Gene"],
        "size": "2 × ~440MB (PubMedBERT-base / BioBERT-base)",
        "trained_on": (
            "NCBI Disease corpus (disease model); CRAFT + other gene NER corpora (gene model); "
            "both models pretrained on PubMed abstracts + PMC full-text"
        ),
    },
    "scispacy_bionlp13cg": {
        "description": (
            "scispaCy en_ner_bionlp13cg_md model trained on the BioNLP 2013 Cancer Genetics "
            "shared task corpus (600 PubMed cancer abstracts). Provides the most granular anatomical "
            "entity hierarchy of any model here: separate labels for Organ, Tissue, Cell, Cellular "
            "Component, Organism Substance, and Developing Anatomical Structure. Also covers cancer "
            "gene/gene product, simple chemicals, and pathological formations."
        ),
        "labels": [
            "CancerGene", "GeneOrGeneProduct", "SimpleChemical", "Organism",
            "OrganismSubdivision", "AnatomicalSystem", "Organ", "MultiTissueStructure",
            "Tissue", "Cell", "CellularComponent", "DevelopingAnatomicalStructure",
            "OrganismSubstance", "ImmunoPeptide", "GeneFunction", "PathologicalFormation",
        ],
        "size": "~85MB (spaCy md model)",
        "trained_on": "BioNLP 2013 Cancer Genetics task corpus (600 PubMed cancer abstracts, 16 entity types)",
    },
    "hunflair2": {
        "description": (
            "HunFlair2 is a PubMedBERT-based biomedical NER model trained on 31 annotated corpora — "
            "the broadest cross-corpus training of any model in this suite. Achieves the highest "
            "reported F1 on cross-domain biomedical NER benchmarks (Disease 88.3, Chemical 92.1, "
            "Gene/Protein 82.7, Species 80.5, CellLine 77.4). Uses the Flair NLP framework with "
            "CRF decoding on top of BERT embeddings. Note: slower on CPU than HuggingFace pipelines "
            "due to CRF overhead (~15-25s on MPS, ~90-150s on CPU for 10k-word articles)."
        ),
        "labels": ["Disease", "Chemical", "Gene", "Species", "CellLine"],
        "size": "~440MB (PubMedBERT-base + Flair CRF)",
        "trained_on": "31 annotated biomedical corpora (HunFlair2, Sanger et al. 2024, Bioinformatics)",
    },
    "species_ner": {
        "description": (
            "PubMedBERT fine-tuned for organism/species recognition (BENT series, pruas). Detects "
            "mentions of model organisms, human, bacteria, viruses, and any named organism. "
            "Fills the largest coverage gap in the existing model suite — species recognition is "
            "near-zero across all other models."
        ),
        "labels": ["Species"],
        "size": "~440MB (PubMedBERT-base)",
        "trained_on": "Organism/species NER corpora (BENT series, PubMedBERT backbone)",
    },
}

GLINER_DEFAULT_ENTITY_TYPES: list[str] = [
    "Drug", "Gene", "Disease", "Protein", "Species", "CellLine", "CellType",
    "DNA", "RNA", "Mutation", "Anatomy", "BiologicalProcess", "MolecularFunction", "Chemical",
]

# ---------------------------------------------------------------------------
# TUI / label normalization tables
# ---------------------------------------------------------------------------

_TUI_TO_LABEL: dict[str, str] = {
    "T047": "Disease", "T191": "Disease", "T048": "Disease", "T046": "Disease",
    "T121": "Drug", "T109": "Drug", "T195": "Drug", "T197": "Drug",
    "T028": "Gene", "T086": "Gene",
    "T116": "Protein", "T126": "Protein", "T125": "Protein",
    "T001": "Species", "T002": "Species", "T004": "Species", "T005": "Species", "T008": "Species",
    "T025": "CellLine", "T045": "CellType",
    "T026": "CellComponent",
    "T023": "Anatomy", "T029": "Anatomy", "T030": "Anatomy",
    "T074": "Device",
    "T038": "BiologicalProcess", "T043": "BiologicalProcess",
    "T044": "MolecularFunction",
}

_BIONLP13CG_LABEL_MAP: dict[str, str] = {
    "Cancer": "CancerGene",
    "Gene_or_gene_product": "GeneOrGeneProduct",
    "Simple_chemical": "SimpleChemical",
    "Organism": "Organism",
    "Organism_subdivision": "OrganismSubdivision",
    "Anatomical_system": "AnatomicalSystem",
    "Organ": "Organ",
    "Multi-tissue_structure": "MultiTissueStructure",
    "Tissue": "Tissue",
    "Cell": "Cell",
    "Cellular_component": "CellularComponent",
    "Developing_anatomical_structure": "DevelopingAnatomicalStructure",
    "Organism_substance": "OrganismSubstance",
    "Immuno_peptide": "ImmunoPeptide",
    "Gene_function": "GeneFunction",
    "Pathological_formation": "PathologicalFormation",
    # Also handle underscored/uppercased variants
    "CANCER": "CancerGene",
    "GENE_OR_GENE_PRODUCT": "GeneOrGeneProduct",
    "SIMPLE_CHEMICAL": "SimpleChemical",
    "ORGANISM": "Organism",
    "CELL": "Cell",
    "TISSUE": "Tissue",
    "ORGAN": "Organ",
}

_TRIPLE_LABEL_MAP: dict[str, str] = {
    "DISEASE": "Disease", "CHEMICAL": "Chemical",
    "GGP": "Gene", "TAXON": "Species", "CHEBI": "Chemical",
    "CL": "CellType", "GO": "BiologicalProcess", "SO": "SequenceFeature",
    "DNA": "DNA", "RNA": "RNA",
    "CELL_LINE": "CellLine", "CELL_TYPE": "CellType", "PROTEIN": "Protein",
}

# ---------------------------------------------------------------------------
# Module-level model state
# ---------------------------------------------------------------------------

# scispacy_umls backend (also used by legacy "mesh"/"umls" load_ner callers)
_nlp: Any = None
_active_linker: str = "umls"  # tracks which linker is loaded for _nlp

# scispacy_triple backend
_nlp_bc5cdr: Any = None
_nlp_craft: Any = None
_nlp_jnlpba: Any = None

# d4data transformer pipeline
_d4data_pipe: Any = None

# gliner model
_gliner_model: Any = None

# pubmedbert pipelines
_pubmedbert_disease_pipe: Any = None
_pubmedbert_gene_pipe: Any = None

# scispacy_bionlp13cg backend
_nlp_bionlp13cg: Any = None

# hunflair2 backend
_hunflair2_tagger: Any = None

# species_ner backend
_species_ner_pipe: Any = None

# Which top-level model key is currently active
_active_model: str = "scispacy_umls"

# Legacy alias: maps old linker strings to model keys
_LINKER_ALIAS: dict[str, str] = {
    "mesh": "scispacy_umls",
    "umls": "scispacy_umls",
}

# ---------------------------------------------------------------------------
# Internal helpers — scispaCy UMLS backend
# (kept at module level so unit tests can patch _get_nlp / _process_chunk)
# ---------------------------------------------------------------------------


def _get_nlp() -> Any:
    global _nlp
    if _nlp is None:
        _load_scispacy_umls()
    return _nlp


def _process_chunk(nlp: Any, chunk: str, char_offset: int) -> list[dict]:
    """Run NER on a single chunk and return entities with adjusted char positions.

    For the scispacy_umls backend, looks up TUI-based labels and attaches
    kb_id (CUI). Also sets umls_cui for backward-compatibility with existing
    tests and callers.
    """
    doc = nlp(chunk)
    results: list[dict] = []

    # Try to get the scispacy_linker pipe object once for the whole chunk.
    # This will be None if the pipe is not present or the nlp object is a mock.
    _linker_pipe: Any = None
    try:
        pipe_names: Any = nlp.pipe_names
        # pipe_names must be a real sequence of strings for this to be meaningful.
        if isinstance(pipe_names, list) and "scispacy_linker" in pipe_names:
            _linker_pipe = nlp.get_pipe("scispacy_linker")
    except Exception:
        pass

    for ent in doc.ents:
        umls_cui: str | None = None
        label = ent.label_

        try:
            kb_ents = ent._.kb_ents
            if kb_ents:
                umls_cui = kb_ents[0][0]
                # Resolve TUI-based label via UMLS linker knowledge base.
                if _linker_pipe is not None and umls_cui is not None:
                    try:
                        cui_entity = _linker_pipe.kb.cui_to_entity.get(umls_cui)
                        if cui_entity and isinstance(getattr(cui_entity, "types", None), (list, tuple, set)):
                            for tui in cui_entity.types:
                                mapped = _TUI_TO_LABEL.get(tui)
                                if mapped:
                                    label = mapped
                                    break
                            else:
                                label = "ENTITY"
                        else:
                            label = "ENTITY"
                    except Exception:
                        label = "ENTITY"
        except Exception as exc:
            print(
                f"Warning: failed to retrieve UMLS CUI for '{ent.text}': {exc}",
                file=sys.stderr,
            )

        results.append(
            {
                "text": ent.text,
                "label": label,
                "kb_id": umls_cui,
                "umls_cui": umls_cui,  # backward-compat alias
                "start_char": ent.start_char + char_offset,
                "end_char": ent.end_char + char_offset,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------


def _load_scispacy_umls() -> None:
    """Load en_core_sci_lg + UMLS linker into the module-global _nlp."""
    global _nlp, _active_linker

    import scispacy  # noqa: F401
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401

    print("[task2_ner] Loading en_core_sci_lg...")
    try:
        nlp = spacy.load("en_core_sci_lg")
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to load en_core_sci_lg. "
            f"Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz\n"
            f"Original error: {exc}"
        ) from exc

    print("[task2_ner] Adding UMLS entity linker (threshold=0.7)...")
    try:
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "threshold": 0.7,
            },
        )
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to add scispacy_linker (umls). "
            f"Ensure scispacy and the UMLS linker index are installed.\n"
            f"Original error: {exc}"
        ) from exc

    _nlp = nlp
    _active_linker = "umls"
    print("[task2_ner] Ready: en_core_sci_lg + umls linker")


def _load_scispacy_triple() -> None:
    """Load the three scispaCy NER models for the triple-stack backend."""
    global _nlp_bc5cdr, _nlp_craft, _nlp_jnlpba

    import scispacy  # noqa: F401
    import spacy

    models = [
        ("en_ner_bc5cdr_md", "_nlp_bc5cdr"),
        ("en_ner_craft_md", "_nlp_craft"),
        ("en_ner_jnlpba_md", "_nlp_jnlpba"),
    ]
    for model_name, attr in models:
        print(f"[task2_ner] Loading {model_name}...")
        try:
            loaded = spacy.load(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"[task2_ner] Failed to load {model_name}. "
                f"Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model_name}-0.5.4.tar.gz\n"
                f"Original error: {exc}"
            ) from exc
        globals()[attr] = loaded

    print("[task2_ner] Ready: triple scispaCy stack (BC5CDR + CRAFT + JNLPBA)")


def _load_d4data() -> None:
    """Load the d4data/biomedical-ner-all transformer pipeline."""
    global _d4data_pipe

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "[task2_ner] transformers is required for the d4data backend. "
            "Install with: pip install transformers"
        ) from exc

    _dev = _ner_device()
    print(f"[task2_ner] Loading d4data/biomedical-ner-all (device={_dev})...")
    try:
        _d4data_pipe = pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
            device=_dev,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to load d4data/biomedical-ner-all.\n"
            f"Original error: {exc}"
        ) from exc

    print("[task2_ner] Ready: d4data/biomedical-ner-all")


def _load_gliner() -> None:
    """Load the GLiNER biomedical model."""
    global _gliner_model

    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError(
            "[task2_ner] gliner package is required. Install with: pip install gliner"
        ) from exc

    print("[task2_ner] Loading Ihor/gliner-biomed-base-v1.0...")
    try:
        _gliner_model = GLiNER.from_pretrained("Ihor/gliner-biomed-base-v1.0", map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to load Ihor/gliner-biomed-base-v1.0.\n"
            f"Original error: {exc}"
        ) from exc

    print("[task2_ner] Ready: GLiNER gliner-biomed-base-v1.0")


def _load_pubmedbert() -> None:
    """Load the PubMedBERT disease + gene NER pipelines."""
    global _pubmedbert_disease_pipe, _pubmedbert_gene_pipe

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "[task2_ner] transformers is required for the pubmedbert backend. "
            "Install with: pip install transformers"
        ) from exc

    _dev = _ner_device()
    print(f"[task2_ner] Loading alvaroalon2/biobert_diseases_ner (disease, device={_dev})...")
    try:
        _pubmedbert_disease_pipe = pipeline(
            "ner",
            model="alvaroalon2/biobert_diseases_ner",
            aggregation_strategy="simple",
            device=_dev,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to load alvaroalon2/biobert_diseases_ner.\n"
            f"Original error: {exc}"
        ) from exc

    print(f"[task2_ner] Loading pruas/BENT-PubMedBERT-NER-Gene (gene, device={_dev})...")
    try:
        _pubmedbert_gene_pipe = pipeline(
            "ner",
            model="pruas/BENT-PubMedBERT-NER-Gene",
            aggregation_strategy="simple",
            device=_dev,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to load pruas/BENT-PubMedBERT-NER-Gene.\n"
            f"Original error: {exc}"
        ) from exc

    print("[task2_ner] Ready: PubMedBERT NER Suite (disease + gene)")


def _load_scispacy_bionlp13cg() -> None:
    global _nlp_bionlp13cg
    import scispacy  # noqa: F401
    import spacy
    model_name = "en_ner_bionlp13cg_md"
    print(f"[task2_ner] Loading {model_name}...")
    try:
        _nlp_bionlp13cg = spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to load {model_name}. "
            f"Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz\n"
            f"Original error: {exc}"
        ) from exc
    print(f"[task2_ner] Ready: {model_name} (BioNLP13CG cancer genetics, 16 types)")


def _load_hunflair2() -> None:
    global _hunflair2_tagger
    try:
        from flair.nn import Classifier
    except ImportError as exc:
        raise RuntimeError(
            "[task2_ner] flair package is required for HunFlair2. "
            "Install with: pip install flair"
        ) from exc
    print("[task2_ner] Loading HunFlair2 via hunflair/hunflair2-ner (downloads ~440MB on first run)...")
    try:
        _hunflair2_tagger = Classifier.load("hunflair2")
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to load HunFlair2 tagger.\n"
            f"Original error: {exc}"
        ) from exc
    print("[task2_ner] Ready: HunFlair2")


def _load_species_ner() -> None:
    global _species_ner_pipe
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "[task2_ner] transformers is required. Install: pip install transformers"
        ) from exc

    _dev = _ner_device()
    # pruas/BENT-PubMedBERT-NER-Organism: same BENT series as the working Gene model,
    # fine-tuned for organism/species mention recognition.
    model_id = "pruas/BENT-PubMedBERT-NER-Organism"
    print(f"[task2_ner] Loading {model_id} (device={_dev})...")
    try:
        _species_ner_pipe = pipeline(
            "ner",
            model=model_id,
            aggregation_strategy="simple",
            device=_dev,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[task2_ner] Failed to load {model_id}.\n"
            f"Original error: {exc}"
        ) from exc
    print(f"[task2_ner] Ready: {model_id}")


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_ner(model_key: str) -> None:
    """Load (or reload) the NER backend identified by model_key.

    Accepts new model keys (scispacy_umls, scispacy_triple, d4data, gliner,
    pubmedbert) as well as legacy linker strings (mesh, umls) which are
    aliased to scispacy_umls.
    """
    global _active_model, _nlp, _nlp_bc5cdr, _nlp_craft, _nlp_jnlpba
    global _d4data_pipe, _gliner_model, _pubmedbert_disease_pipe, _pubmedbert_gene_pipe
    global _nlp_bionlp13cg, _hunflair2_tagger, _species_ner_pipe

    # Resolve legacy aliases.
    resolved_key = _LINKER_ALIAS.get(model_key, model_key)

    if resolved_key not in NER_MODEL_OPTIONS:
        raise ValueError(
            f"[task2_ner] Unknown model key: '{model_key}'. "
            f"Valid keys: {list(NER_MODEL_OPTIONS.keys())} "
            f"(legacy aliases 'mesh'/'umls' also accepted)"
        )

    if resolved_key == _active_model:
        # Already loaded — check that the relevant objects are populated.
        already_loaded = {
            "scispacy_umls": _nlp is not None,
            "scispacy_triple": all(x is not None for x in [_nlp_bc5cdr, _nlp_craft, _nlp_jnlpba]),
            "d4data": _d4data_pipe is not None,
            "gliner": _gliner_model is not None,
            "pubmedbert": all(x is not None for x in [_pubmedbert_disease_pipe, _pubmedbert_gene_pipe]),
            "scispacy_bionlp13cg": _nlp_bionlp13cg is not None,
            "hunflair2": _hunflair2_tagger is not None,
            "species_ner": _species_ner_pipe is not None,
        }
        if already_loaded.get(resolved_key, False):
            return

    print(f"[task2_ner] Switching to model: {resolved_key}...")

    # Free previous model objects to release memory before loading a new one.
    if _active_model != resolved_key:
        _nlp = None
        _nlp_bc5cdr = _nlp_craft = _nlp_jnlpba = None
        _d4data_pipe = None
        _gliner_model = None
        _pubmedbert_disease_pipe = _pubmedbert_gene_pipe = None
        _nlp_bionlp13cg = None
        _hunflair2_tagger = None
        _species_ner_pipe = None

    _active_model = resolved_key

    if resolved_key == "scispacy_umls":
        _load_scispacy_umls()
    elif resolved_key == "scispacy_triple":
        _load_scispacy_triple()
    elif resolved_key == "d4data":
        _load_d4data()
    elif resolved_key == "gliner":
        _load_gliner()
    elif resolved_key == "pubmedbert":
        _load_pubmedbert()
    elif resolved_key == "scispacy_bionlp13cg":
        _load_scispacy_bionlp13cg()
    elif resolved_key == "hunflair2":
        _load_hunflair2()
    elif resolved_key == "species_ner":
        _load_species_ner()


# ---------------------------------------------------------------------------
# Extraction: scispaCy UMLS (uses _get_nlp / _process_chunk for testability)
# ---------------------------------------------------------------------------


def _extract_scispacy_umls(text: str) -> list[dict]:
    """Extract entities using the scispaCy + UMLS backend."""
    nlp = _get_nlp()
    max_length: int = nlp.max_length

    print(f"[task2_ner] Running scispacy_umls NER on {len(text):,} chars...")

    if len(text) <= max_length:
        results = sorted(_process_chunk(nlp, text, 0), key=lambda e: e["start_char"])
        print(f"[task2_ner] Found {len(results)} entities.")
        return results

    # Chunk text with overlap to avoid missing entities at boundaries.
    chunk_size = max_length
    overlap = 200
    total_chunks = -(-len(text) // (chunk_size - overlap))  # ceiling div

    seen: set[tuple[str, int]] = set()
    all_entities: list[dict] = []

    start = 0
    chunk_num = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_num += 1
        print(f"[task2_ner] Processing chunk {chunk_num}/{total_chunks} (chars {start}–{end})...")
        chunk_entities = _process_chunk(nlp, text[start:end], char_offset=start)

        for ent in chunk_entities:
            key = (ent["text"], ent["start_char"])
            if key not in seen:
                seen.add(key)
                all_entities.append(ent)

        if end == len(text):
            break
        start += chunk_size - overlap

    results = sorted(all_entities, key=lambda e: e["start_char"])
    print(f"[task2_ner] Found {len(results)} entities across {chunk_num} chunks.")
    return results


# ---------------------------------------------------------------------------
# Extraction: scispaCy triple stack
# ---------------------------------------------------------------------------


def _extract_scispacy_triple(text: str) -> list[dict]:
    """Extract entities using the triple scispaCy backend."""
    print(f"[task2_ner] Running scispacy_triple NER on {len(text):,} chars...")

    if any(m is None for m in [_nlp_bc5cdr, _nlp_craft, _nlp_jnlpba]):
        _load_scispacy_triple()

    raw_spans: list[dict] = []

    for nlp in [_nlp_bc5cdr, _nlp_craft, _nlp_jnlpba]:
        doc = nlp(text)
        for ent in doc.ents:
            raw_label = ent.label_
            label = _TRIPLE_LABEL_MAP.get(raw_label, raw_label)
            raw_spans.append(
                {
                    "text": ent.text,
                    "label": label,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

    # Sort by start_char, then merge — skip any span overlapping an accepted span.
    raw_spans.sort(key=lambda s: s["start_char"])
    merged: list[dict] = []
    last_end = -1
    for span in raw_spans:
        if span["start_char"] >= last_end:
            merged.append(span)
            last_end = span["end_char"]

    print(f"[task2_ner] Found {len(merged)} entities (after overlap resolution).")
    return merged


# ---------------------------------------------------------------------------
# Extraction: d4data transformer
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[tuple[str, int]]:
    """Split text on double-newlines, returning (paragraph, char_offset) tuples."""
    paragraphs: list[tuple[str, int]] = []
    pos = 0
    for para in text.split("\n\n"):
        paragraphs.append((para, pos))
        pos += len(para) + 2  # +2 for the "\n\n" separator
    return paragraphs


@spaces.GPU
def _extract_d4data(text: str) -> list[dict]:
    """Extract entities using the d4data DistilBERT pipeline."""
    print(f"[task2_ner] Running d4data NER on {len(text):,} chars...")

    if _d4data_pipe is None:
        _load_d4data()

    word_count = len(text.split())
    results: list[dict] = []

    if word_count <= 400:
        raw = _d4data_pipe(text)
        for ent in raw:
            results.append(
                {
                    "text": ent["word"],
                    "label": ent["entity_group"],
                    "start_char": ent["start"],
                    "end_char": ent["end"],
                }
            )
    else:
        paragraphs = _split_paragraphs(text)
        for para, offset in paragraphs:
            if not para.strip():
                continue
            raw = _d4data_pipe(para)
            for ent in raw:
                results.append(
                    {
                        "text": ent["word"],
                        "label": ent["entity_group"],
                        "start_char": ent["start"] + offset,
                        "end_char": ent["end"] + offset,
                    }
                )

    results.sort(key=lambda e: e["start_char"])
    print(f"[task2_ner] Found {len(results)} entities.")
    return results


# ---------------------------------------------------------------------------
# Extraction: GLiNER zero-shot
# ---------------------------------------------------------------------------


@spaces.GPU
def _extract_gliner(text: str, entity_types: list[str]) -> list[dict]:
    """Extract entities using the GLiNER zero-shot backend."""
    print(f"[task2_ner] Running GLiNER NER on {len(text):,} chars (types={len(entity_types)})...")

    if _gliner_model is None:
        _load_gliner()

    word_count = len(text.split())
    results: list[dict] = []

    if word_count <= 400:
        raw = _gliner_model.predict_entities(text, entity_types, threshold=0.5)
        for ent in raw:
            results.append(
                {
                    "text": ent["text"],
                    "label": ent["label"],
                    "start_char": ent["start"],
                    "end_char": ent["end"],
                }
            )
    else:
        # Split on paragraph boundaries and preserve offsets.
        paragraphs = _split_paragraphs(text)
        for para, offset in paragraphs:
            if not para.strip():
                continue
            raw = _gliner_model.predict_entities(para, entity_types, threshold=0.5)
            for ent in raw:
                results.append(
                    {
                        "text": ent["text"],
                        "label": ent["label"],
                        "start_char": ent["start"] + offset,
                        "end_char": ent["end"] + offset,
                    }
                )

    results.sort(key=lambda e: e["start_char"])
    print(f"[task2_ner] Found {len(results)} entities.")
    return results


# ---------------------------------------------------------------------------
# Extraction: PubMedBERT suite
# ---------------------------------------------------------------------------


@spaces.GPU
def _extract_pubmedbert(text: str) -> list[dict]:
    """Extract entities using the merged PubMedBERT disease + gene pipelines."""
    print(f"[task2_ner] Running pubmedbert NER on {len(text):,} chars...")

    if _pubmedbert_disease_pipe is None or _pubmedbert_gene_pipe is None:
        _load_pubmedbert()

    word_count = len(text.split())
    seen_spans: set[tuple[int, int]] = set()
    results: list[dict] = []

    def _run_pipe(pipe: Any, chunk: str, offset: int) -> list[dict]:
        out: list[dict] = []
        for ent in pipe(chunk):
            out.append(
                {
                    "text": ent["word"],
                    "label": ent["entity_group"],
                    "start_char": ent["start"] + offset,
                    "end_char": ent["end"] + offset,
                }
            )
        return out

    if word_count <= 400:
        for pipe in [_pubmedbert_disease_pipe, _pubmedbert_gene_pipe]:
            for ent in _run_pipe(pipe, text, 0):
                key = (ent["start_char"], ent["end_char"])
                if key not in seen_spans:
                    seen_spans.add(key)
                    results.append(ent)
    else:
        paragraphs = _split_paragraphs(text)
        for pipe in [_pubmedbert_disease_pipe, _pubmedbert_gene_pipe]:
            for para, offset in paragraphs:
                if not para.strip():
                    continue
                for ent in _run_pipe(pipe, para, offset):
                    key = (ent["start_char"], ent["end_char"])
                    if key not in seen_spans:
                        seen_spans.add(key)
                        results.append(ent)

    results.sort(key=lambda e: e["start_char"])
    print(f"[task2_ner] Found {len(results)} entities.")
    return results


# ---------------------------------------------------------------------------
# Extraction: scispaCy BioNLP13CG
# ---------------------------------------------------------------------------


def _extract_scispacy_bionlp13cg(text: str) -> list[dict]:
    print(f"[task2_ner] Running scispacy_bionlp13cg NER on {len(text):,} chars...")
    if _nlp_bionlp13cg is None:
        _load_scispacy_bionlp13cg()
    doc = _nlp_bionlp13cg(text)
    results = []
    for ent in doc.ents:
        raw_label = ent.label_
        label = _BIONLP13CG_LABEL_MAP.get(raw_label, raw_label)
        results.append({
            "text": ent.text,
            "label": label,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        })
    results.sort(key=lambda e: e["start_char"])
    print(f"[task2_ner] Found {len(results)} entities.")
    return results


# ---------------------------------------------------------------------------
# Extraction: HunFlair2
# ---------------------------------------------------------------------------


def _extract_hunflair2(text: str) -> list[dict]:
    print(f"[task2_ner] Running HunFlair2 NER on {len(text):,} chars...")
    if _hunflair2_tagger is None:
        _load_hunflair2()
    try:
        from flair.data import Sentence
    except ImportError as exc:
        raise RuntimeError("[task2_ner] flair is required.") from exc

    # Split text into sentences/paragraphs for Flair processing
    # Use paragraph-level splitting to preserve offsets
    paragraphs = _split_paragraphs(text)
    results: list[dict] = []

    for para, offset in paragraphs:
        if not para.strip():
            continue
        # Flair processes one Sentence at a time
        try:
            sentence = Sentence(para)
            _hunflair2_tagger.predict(sentence)
            for entity in sentence.get_spans("ner"):
                results.append({
                    "text": entity.text,
                    "label": entity.get_label("ner").value,
                    "start_char": entity.start_position + offset,
                    "end_char": entity.end_position + offset,
                })
        except Exception as exc:
            print(f"[task2_ner] HunFlair2 error on paragraph at offset {offset}: {exc}")
            continue

    results.sort(key=lambda e: e["start_char"])
    print(f"[task2_ner] Found {len(results)} entities.")
    return results


# ---------------------------------------------------------------------------
# Extraction: Species NER
# ---------------------------------------------------------------------------


def _normalize_ner_label(raw_label: str, default: str) -> str:
    """Normalize potentially numeric or 'O' labels from BERT NER pipelines."""
    if not raw_label or raw_label in ("0", "O", "LABEL_0"):
        return default
    # Strip B-/I- prefixes if aggregation_strategy didn't handle them
    for prefix in ("B-", "I-", "b-", "i-"):
        if raw_label.startswith(prefix):
            return raw_label[2:]
    return raw_label


@spaces.GPU
def _extract_species_ner(text: str) -> list[dict]:
    print(f"[task2_ner] Running species NER on {len(text):,} chars...")
    if _species_ner_pipe is None:
        _load_species_ner()

    word_count = len(text.split())
    results: list[dict] = []

    def _run(chunk: str, offset: int) -> None:
        for ent in _species_ner_pipe(chunk):
            label = _normalize_ner_label(ent.get("entity_group", ""), "Species")
            results.append({
                "text": ent["word"],
                "label": label,
                "start_char": ent["start"] + offset,
                "end_char": ent["end"] + offset,
            })

    if word_count <= 400:
        _run(text, 0)
    else:
        for para, offset in _split_paragraphs(text):
            if para.strip():
                _run(para, offset)

    results.sort(key=lambda e: e["start_char"])
    print(f"[task2_ner] Found {len(results)} entities.")
    return results


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------


def extract_entities(
    text: str,
    gliner_entity_types: list[str] | None = None,
) -> list[dict]:
    """Extract biomedical named entities from text using the active NER backend.

    Args:
        text: Input biomedical text.
        gliner_entity_types: Entity type labels for the GLiNER backend. When
            None, GLINER_DEFAULT_ENTITY_TYPES is used. Ignored for all other
            backends.

    Returns:
        List of entity dicts. All backends return at minimum:
            {"text": str, "label": str, "start_char": int, "end_char": int}
        The scispacy_umls backend additionally includes:
            {"kb_id": str | None, "umls_cui": str | None}
    """
    if not text or len(text) < 10:
        return []

    model = _active_model

    if model == "scispacy_umls":
        return _extract_scispacy_umls(text)
    elif model == "scispacy_triple":
        return _extract_scispacy_triple(text)
    elif model == "d4data":
        return _extract_d4data(text)
    elif model == "gliner":
        types = gliner_entity_types if gliner_entity_types is not None else GLINER_DEFAULT_ENTITY_TYPES
        return _extract_gliner(text, types)
    elif model == "pubmedbert":
        return _extract_pubmedbert(text)
    elif model == "scispacy_bionlp13cg":
        return _extract_scispacy_bionlp13cg(text)
    elif model == "hunflair2":
        return _extract_hunflair2(text)
    elif model == "species_ner":
        return _extract_species_ner(text)
    else:
        raise RuntimeError(f"[task2_ner] No handler for active model: '{model}'")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python task2_ner.py <text_file_path> [model_key]", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "scispacy_umls"

    with open(file_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    load_ner(model_arg)
    entities = extract_entities(content)
    print(f"Entity count: {len(entities)}")
    print("First 5 entities:")
    print(json.dumps(entities[:5], indent=2, default=str))
