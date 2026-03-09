PYTHON   := python3.11
VENV     := .venv
BIN      := $(VENV)/bin
PIP      := $(BIN)/pip

.PHONY: install reinstall upgrade run test test-integration clean help

## Create venv and install all dependencies
install: $(VENV)/touchfile

$(VENV)/touchfile: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/touchfile
	@echo "\nInstall complete. Run: make run"

## Wipe venv and reinstall from scratch
reinstall: clean install

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
