PYTHON ?= python

.PHONY: download prepare analyze baseline crf smoke-crf compare audit test summary presentation reproduce

download:
	$(PYTHON) scripts/download_data.py

prepare:
	$(PYTHON) scripts/prepare_data.py

analyze:
	$(PYTHON) scripts/analyze_data.py

baseline:
	$(PYTHON) scripts/train_baselines.py --model o-only
	$(PYTHON) scripts/train_baselines.py --model most-frequent-word

crf:
	$(PYTHON) scripts/train_baselines.py --model crf

smoke-crf:
	$(PYTHON) scripts/train_baselines.py --model crf --max-train-contracts 10 --bootstrap-iterations 200

compare:
	$(PYTHON) scripts/compare_models.py results/most-frequent-word-seed42_predictions.jsonl results/o-only-seed42_predictions.jsonl

audit:
	$(PYTHON) scripts/audit_predictions.py results/most-frequent-word-seed42_predictions.jsonl

test:
	$(PYTHON) -m unittest discover -s tests

summary:
	$(PYTHON) scripts/summarize_experiments.py

presentation:
	$(PYTHON) scripts/build_presentation.py

reproduce: download prepare analyze baseline smoke-crf compare audit summary presentation test
