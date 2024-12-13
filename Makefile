SHELL := /bin/bash

.PHONY: all deploy develop

all: requirements.txt
	@if not exist .venv\Scripts ( \
		python -m venv .venv \
	)
	@if exist .venv\Scripts ( \
		.venv\Scripts\activate && pip install -r requirements.txt \
	) else ( \
		source .venv/bin/activate && pip install -r requirements.txt \
	)