# Reproduction

Run commands from the repository root. Raw and cached data are not tracked.

```bash
make test
make figures
make cache
make baselines
make temporal-cache
make temporal
```

The last two targets require the extracted official package. Shortcut controls are separate because several reread roughly 12 GB of local data. Each script has `--help`; the frozen settings and seeds are recorded in `docs/preregistration_shortcut_audit_v1.md` and the committed JSON reports.

Core audit commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_prefall.py --cache data/cache --events data/extracted/abc2027_uwb_train_v1/streams_events.csv --output reports/shortcut_audit/prefall.json --replicates 10000
PYTHONPATH=src .venv/bin/python scripts/audit_join_boundaries.py --cache data/cache --composition data/extracted/abc2027_uwb_train_v1/streams_composition.csv --events data/extracted/abc2027_uwb_train_v1/streams_events.csv --output reports/shortcut_audit/join_boundary.json --replicates 10000
```

`make temporal-cache` produces the compact order-sensitive cache from official
HDF5 streams. `make temporal` trains the fixed 37k-parameter TCN for 12 epochs,
evaluates clean and time-shuffled inputs with a frozen clean decoder, and runs
the matched pre-fall control. On CPU this takes several minutes; raw caches and
the generated checkpoint are excluded from Git.
