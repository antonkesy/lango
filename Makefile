.PHONY:
	prep
	uninstall
	install
	install-dev
	run
	compile
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
	lango run --input_file examples/minio/example.minio

compile: install
	. .venv/bin/activate && \
	lango compile examples/minio/example.minio -o ./build/example.py && \
	python3.13 ./build/example.py

types: install
	. .venv/bin/activate && \
	lango types examples/minio/example.minio

ast: install
	. .venv/bin/activate && \
	lango ast examples/minio/short.minio

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

mypy: install-dev
	. .venv/bin/activate && \
	mypy lango
	mypy test

clean: uninstall
	rm -rf build/ dist/ *.egg-info/ .venv/ .pytest_cache/ .coverage htmlcov/
