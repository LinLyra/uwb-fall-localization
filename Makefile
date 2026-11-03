PYTHON := .venv/bin/python
PACKAGE := data/extracted/abc2027_uwb_train_v1

.PHONY: test cache baselines figures
test:
	PYTHONPATH=src:. $(PYTHON) -m pytest -q
cache:
	PYTHONPATH=src $(PYTHON) scripts/build_feature_cache.py --package $(PACKAGE) --output data/cache --window 100 --hop 25 --range-bands 16
baselines:
	PYTHONPATH=src $(PYTHON) scripts/train_tabular_baselines.py --cache data/cache --package $(PACKAGE) --output reports/tabular
	PYTHONPATH=src $(PYTHON) scripts/nested_decoder_evaluation.py --cache data/cache --package $(PACKAGE) --output reports/nested
figures:
	MPLCONFIGDIR=/tmp/uwb-mpl PYTHONPATH=src $(PYTHON) scripts/plot_results.py
	MPLCONFIGDIR=/tmp/uwb-mpl PYTHONPATH=src $(PYTHON) scripts/plot_spatial_audit.py
