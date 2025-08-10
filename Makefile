.PHONY:
	prep
	uninstall
	install
	install-dev
	run
	type
	test
	coverage
	clean

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
	lango run --input_file test/files/minio/let/multiple.minio

types: install
	. .venv/bin/activate && \
	lango types examples/minio/example.minio

typecheck: install
	. .venv/bin/activate && \
	lango typecheck ./test/files/minio/types/custom/constrcutor/mixed.minio

test: install
	. .venv/bin/activate && \
	pytest -vvs

coverage: install-dev
	. .venv/bin/activate && \
	coverage run -m pytest && \
	coverage report -m && \
	coverage html

clean: uninstall
	rm -rf build/ dist/ *.egg-info/ .venv/ .pytest_cache/ .coverage htmlcov/
