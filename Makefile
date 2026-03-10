# Detect OS — use Scripts/ on Windows, bin/ on Unix/Mac
ifeq ($(OS),Windows_NT)
  PYTHON    := py -3.12
  BIN       := .venv/Scripts
  TOUCH     := $(BIN)/python -c "import pathlib; pathlib.Path('.venv/touchfile').touch()"
else
  PYTHON    := python3.11
  BIN       := .venv/bin
  TOUCH     := touch $(VENV)/touchfile
endif

VENV := .venv
PIP  := $(BIN)/pip

.PHONY: install install-cuda reinstall reinstall-cuda upgrade run test test-integration clean help

## Create venv and install all dependencies (CPU-only torch)
install: $(VENV)/touchfile

$(VENV)/touchfile: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	$(TOUCH)
	@echo "Install complete. Run: make run"
	@echo "Have an NVIDIA GPU? Run: make install-cuda"

## Install CUDA-enabled torch (NVIDIA GPU, CUDA 12.1+) then install everything else
install-cuda:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(PIP) install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -r requirements.txt
	$(TOUCH)
	@echo "CUDA install complete. Run: make run"

## Wipe venv and reinstall from scratch (CPU)
reinstall: clean install

## Wipe venv and reinstall with CUDA torch
reinstall-cuda: clean install-cuda

## Upgrade gradio and gradio_client in existing venv (fixes Gallery schema bug)
upgrade:
	$(PIP) install --upgrade "gradio>=4.44.0" gradio_client
	@echo "Gradio upgraded. Run: make run"

## Launch the Gradio app (http://localhost:7860)
run: $(VENV)/touchfile
	$(BIN)/python app.py

## Run unit tests (no network, no models)
test: $(VENV)/touchfile
	$(BIN)/pytest tests/unit/ -v

## Run integration tests (requires network)
test-integration: $(VENV)/touchfile
	$(BIN)/pytest tests/integration/ -m integration -v

## Remove venv and Python cache files
clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."

## Show available targets
help:
	@grep -E '^##' Makefile | sed 's/## //'
