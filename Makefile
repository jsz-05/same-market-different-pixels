PYTHON ?= .venv/bin/python

.PHONY: test download gallery study analysis external-validation stock-replication extension-analysis

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

download:
	$(PYTHON) src/download_prices.py --end 2026-08-01 \
		--output data/raw/prices/etf_daily_final.parquet \
		--manifest data/raw/prices/manifest_final.json

gallery:
	PYTHONPATH=src $(PYTHON) src/make_style_gallery.py --output data/processed/style_gallery.png

study:
	PYTHONPATH=src $(PYTHON) src/run_study.py \
		--data data/raw/prices/etf_daily_final.parquet \
		--output-dir results/final --epochs 10 --seeds 2026 2027 2028

analysis:
	PYTHONPATH=src $(PYTHON) src/analyze_results.py --bootstrap-draws 2000 \
		--figure-dir data/processed/figures

external-validation:
	PYTHONPATH=src $(PYTHON) src/external_validation.py

stock-replication:
	PYTHONPATH=src $(PYTHON) src/download_stock_replication.py
	PYTHONPATH=src $(PYTHON) src/run_study.py \
		--data data/raw/prices/sp100_daily_final.parquet \
		--output-dir results/stock_replication --protocol protocol/extension_protocol.json \
		--epochs 10 --seeds 2026 2027 2028 \
		--models rgb_canonical rgb_aug_consistency

extension-analysis:
	PYTHONPATH=src $(PYTHON) src/make_extension_results.py --latex-dir data/processed/tables
