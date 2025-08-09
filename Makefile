.PHONY:
	prep
	uninstall
	install
	install-dev
	run
	type
	test
	coverage

all: run

prep:
	python3.13 -m venv .venv

uninstall:
	.venv/bin/pip uninstall lango -y

install:
	.venv/bin/pip install .

install-dev:
	.venv/bin/pip install -e .[dev]

run: install
	. .venv/bin/activate && \
	lango run --input_file examples/minio/example.minio

type: install
	. .venv/bin/activate && \
	lango typecheck examples/minio/example.minio

test: install
	. .venv/bin/activate && \
	pytest -vvs

coverage: install-dev
	. .venv/bin/activate && \
	coverage run -m pytest && \
	coverage report -m && \
	coverage html
