PYTHON := .venv/bin/python
PACKAGE := data/extracted/abc2027_uwb_train_v1

.PHONY: test cache temporal-cache baselines temporal figures
test:
	PYTHONPATH=src:. $(PYTHON) -m pytest -q
cache:
	PYTHONPATH=src $(PYTHON) scripts/build_feature_cache.py --package $(PACKAGE) --output data/cache --window 100 --hop 25 --range-bands 16
temporal-cache:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_temporal_cache.py --cache data/cache --package $(PACKAGE) --output data/temporal
baselines:
	PYTHONPATH=src $(PYTHON) scripts/train_tabular_baselines.py --cache data/cache --package $(PACKAGE) --output reports/tabular
	PYTHONPATH=src $(PYTHON) scripts/nested_decoder_evaluation.py --cache data/cache --package $(PACKAGE) --output reports/nested
temporal:
	PYTHONPATH=src:scripts $(PYTHON) scripts/train_temporal_tcn.py --cache data/temporal --package $(PACKAGE) --output reports/temporal_tcn --epochs 12 --seed 2027
figures:
	MPLCONFIGDIR=/tmp/uwb-mpl PYTHONPATH=src $(PYTHON) scripts/plot_results.py
	MPLCONFIGDIR=/tmp/uwb-mpl PYTHONPATH=src $(PYTHON) scripts/plot_spatial_audit.py
	MPLCONFIGDIR=/tmp/uwb-mpl PYTHONPATH=src $(PYTHON) scripts/plot_temporal_tcn.py --tcn reports/temporal_tcn/summary.json --output reports/temporal_tcn/temporal_controls
