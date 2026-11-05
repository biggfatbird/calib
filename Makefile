.PHONY: test recalibrate more-splits corruptions thresholds cross-dataset associate figures compare

test:
	python -m pytest -q tests

recalibrate:
	python -m wdbc_calib recalibration

more-splits:
	python -m wdbc_calib more-splits

corruptions:
	python -m wdbc_calib corruptions

thresholds:
	python -m wdbc_calib thresholds

cross-dataset:
	python -m wdbc_calib cross-dataset

associate:
	python -m wdbc_calib associate

compare:
	python -m wdbc_calib compare-preserved

figures:
	python -m wdbc_calib figures
