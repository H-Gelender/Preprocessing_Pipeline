# Makefile 

# --- Configuration ---
VENV_DIR = venv

.PHONY: venv
venv: 
	@echo "--> Creation de l'environnement virtuel..."
	python3 -m venv $(VENV_DIR)
	@echo "--> Environnement virtuel créé dans $(VENV_DIR)."

.PHONY: install
install: 

	@echo "--> Installation des dependances depuis requirements.txt..."
	pip install -r requirements.txt
	@echo "--> Installation terminee."