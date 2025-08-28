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

# MiniO
minio-parse: install
	. .venv/bin/activate && \
	lango parse minio examples/minio/example.minio

minio-run: install
	. .venv/bin/activate && \
	lango run minio examples/minio/example.minio

minio-compile-python: install
	. .venv/bin/activate && \
	lango compile minio examples/minio/example.minio -o build/example.py && \
	python3.13 build/example.py

minio-compile-go: install
	. .venv/bin/activate && \
	lango compile minio examples/minio/example.minio -o build/example.go --target go && \
	go run build/example.go

minio-types: install
	. .venv/bin/activate && \
	lango types minio examples/minio/example.minio

minio-typecheck: install
	. .venv/bin/activate && \
	lango typecheck minio examples/minio/example.minio

# SystemO
systemo-parse: install
	. .venv/bin/activate && \
	lango parse systemo examples/systemo/example.syso

systemo-run: install
	. .venv/bin/activate && \
	lango run systemo examples/systemo/example.syso

systemo-types: install
	. .venv/bin/activate && \
	lango types systemo examples/systemo/example.syso

systemo-typecheck: install
	. .venv/bin/activate && \
	lango typecheck systemo examples/systemo/example.syso

# Quality
test: install
	. .venv/bin/activate && \
	pytest -vvs

test-systemo: install
	. .venv/bin/activate && \
	pytest -vvs test/systemo/test_systemo.py

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

docker-clean:
	docker rmi lango lango-test || true

docker-run:
	docker run -it --rm lango run --input_file examples/minio/example.minio

docker-test:
	docker run -it --rm lango-test
