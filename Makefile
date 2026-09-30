.PHONY: install install-browser test acquire run

install:
	python -m pip install -e '.[dev]'

install-browser:
	python -m pip install -e '.[browser]'
	python -m playwright install chromium

test:
	pytest

acquire:
	divar-scanner acquire --config config/fatemi.yaml --transport browser --max-listings 200

run:
	divar-scanner run --config config/fatemi.yaml --crawl-transport browser
