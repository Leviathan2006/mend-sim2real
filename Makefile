.PHONY: baselines budget mend tables all smoke

all: baselines budget mend tables

baselines:
	python scripts/run_baselines.py

budget:
	python scripts/run_budget.py

mend:
	python scripts/run_mend.py

tables:
	python scripts/make_tables.py

smoke:
	python scripts/run_baselines.py --quick
	python scripts/run_budget.py --quick
	python scripts/run_mend.py --quick
