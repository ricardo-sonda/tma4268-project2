PROJECT_NAME = src
PYTHON = uv run python
MODEL ?= all

.PHONY: data refresh-data sql plots evaluate tui logistic-ground-zero clean

data:
	$(PYTHON) -m $(PROJECT_NAME).dataset build-all

refresh-data:
	$(PYTHON) -m $(PROJECT_NAME).dataset rebuild-from-scratch

sql:
	$(PYTHON) -m $(PROJECT_NAME).dataset db

plots:
	$(PYTHON) -m $(PROJECT_NAME).plots all

evaluate:
	$(PYTHON) -m $(PROJECT_NAME).modeling.run $(MODEL)

tui:
	$(PYTHON) -m $(PROJECT_NAME).tui

logistic-ground-zero:
	$(PYTHON) -m $(PROJECT_NAME).modeling.run logistic_ground_zero

clean:
	find . -path './.venv' -prune -o -type d \( -name "__pycache__" -o -name "*.egg-info" \) -exec rm -rf {} +
