# Makefile 

# --- Configuration ---
VENV_DIR = .venv

# --- Cibles Principales ---
all: install

.PHONY: install
install: $(VENV_PIP)

	@echo "--> Installation des dependances depuis requirements.txt..."
	pip install -r requirements.txt
	@echo "--> Installation terminee."

.PHONY: merge-dataset
merge-dataset: install
	@echo "--> Lancement du script de fusion des jeux de données"
	python merge_dataset/merge_datasets.py
	@echo "--> Done"