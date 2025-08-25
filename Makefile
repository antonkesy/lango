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

compile-python: install
	. .venv/bin/activate && \
	lango compile examples/minio/example.minio -o ./build/example.py && \
	python3.13 ./build/example.py

compile-systemf: install
	. .venv/bin/activate && \
	lango compile test/minio/files/math/arithmetic/add.minio -o ./build/example.sf --target systemf && \
	fullpoly ./build/example.sf


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
	mypy lango
	mypy test

clean: uninstall
	rm -rf build/ dist/ *.egg-info/ .venv/ .pytest_cache/ .coverage htmlcov/

docker-build:
	docker build --target lango -t lango .
	docker build --target test -t lango-test .

docker-run:
	docker run -it --rm lango run --input_file examples/minio/example.minio

docker-test:
	docker run -it --rm lango-test
