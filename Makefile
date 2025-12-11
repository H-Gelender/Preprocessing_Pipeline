
# Bio-Swarm Automation Makefile

PYTHON = python
SCRIPTS = src/scripts
DASHBOARD = src/dashboard/app.py

.PHONY: help reset train dash serve all

help:
	@echo "Bio-Swarm Commands:"
	@echo "  make reset    - Reset experiment logs and checkpoints (Interactive)"
	@echo "  make train    - Launch RL Training (PPO)"
	@echo "  make dash     - Generate Static Visualization Report (HTML)"
	@echo "  make serve    - Launch Live Dash Application (Port 8080)"
	@echo "  make all      - Run full pipeline: Reset -> Train -> Report"

reset:
	$(PYTHON) $(SCRIPTS)/reset_experiment.py

train:
	$(PYTHON) $(SCRIPTS)/train_rl.py

dash:
	@echo "Generating static report..."
	$(PYTHON) $(SCRIPTS)/generate_report.py
	@echo "Report ready at /app/output/bio_swarm_report.html"

serve:
	@echo "Starting Dashboard on Port 8080..."
	$(PYTHON) $(DASHBOARD)

all:
	@echo "Running full pipeline..."
	make reset
	make train
	make dash