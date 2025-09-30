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

all:
	@echo "Available targets:"
	@echo "  prep             - Set up a Python virtual environment"
	@echo "  install         - Install the lango package"
	@echo "  install-dev     - Install the lango package with development dependencies"
	@echo "  uninstall       - Uninstall the lango package"
	@echo "  clean           - Remove build artifacts and virtual environment"
	@echo "  help            - Show help for lango command"
	@echo ""
	@echo "MiniO targets:"
	@echo "  minio-parse            - Parse a MiniO source file"
	@echo "  minio-run              - Run a MiniO source file"
	@echo "  minio-compile-python   - Compile MiniO to Python and run it"
	@echo "  minio-compile-go       - Compile MiniO to Go and run it"
	@echo "  minio-types            - Show types in a MiniO source file"
	@echo "  minio-typecheck        - Typecheck a MiniO source file"
	@echo ""
	@echo "SystemO targets:"
	@echo "  systemo-parse          - Parse a SystemO source file"
	@echo "  systemo-run            - Run a SystemO source file"
	@echo "  systemo-compile-mono   - Compile SystemO to Python using monomorphization and run it"
	@echo "  systemo-compile-dp     - Compile SystemO to Python using dictionary passing and run it"
	@echo "  systemo-types          - Show types in a SystemO source file"
	@echo "  systemo-functions      - Show functions in a SystemO source file"
	@echo "  systemo-typecheck      - Typecheck a SystemO source file"
	@echo ""
	@echo "Profiling targets:"
	@echo "  profil-minio-run       - Profile running a MiniO source file"
	@echo "  profil-systemo-run     - Profile running a SystemO source file"
	@echo ""
	@echo "Benchmarking targets:"
	@echo "  bench-fibonacci        - Benchmark Fibonacci implementations"
	@echo ""
	@echo "Quality targets:"
	@echo "  test            - Run all tests"
	@echo "  test-minio      - Run MiniO tests"
	@echo "  test-systemo    - Run SystemO tests"
	@echo "  coverage        - Generate test coverage report"
	@echo "  mypy            - Run mypy type checks"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build    - Build Docker images for lango and tests"
	@echo "  docker-clean    - Remove Docker images"
	@echo "  docker-run-minio - Run MiniO example in Docker"
	@echo "  docker-test     - Run tests in Docker"


prep:
	python3.13 -m venv venv
	mkdir -p build

uninstall:
	venv/bin/pip uninstall lango -y

install:
	venv/bin/pip install .

install-dev:
	venv/bin/pip install -e .[dev]

clean: uninstall
	rm -rf build/ dist/ *.egg-info/ venv/ .pytest_cache/ .coverage htmlcov/

help: install
	. venv/bin/activate && \
	lango --help

# MiniO
minio-parse: install
	. venv/bin/activate && \
	lango parse minio examples/minio/example.minio

minio-run: install
	. venv/bin/activate && \
	lango run minio examples/minio/example.minio

minio-compile-python: install
	. venv/bin/activate && \
	lango compile minio examples/minio/short.minio -o build/example.py && \
	python3.13 build/example.py

minio-compile-go: install
	. venv/bin/activate && \
	lango compile minio examples/minio/example.minio -o build/example.go --target go && \
	go run build/example.go

minio-types: install
	. venv/bin/activate && \
	lango types minio examples/minio/example.minio

minio-typecheck: install
	. venv/bin/activate && \
	lango typecheck minio examples/minio/example.minio

# SystemO
systemo-parse: install
	. venv/bin/activate && \
	lango parse systemo examples/systemo/example.syso

systemo-run: install
	. venv/bin/activate && \
	lango run systemo examples/systemo/example.syso

systemo-types: install
	. venv/bin/activate && \
	lango types systemo examples/systemo/example.syso

systemo-functions: install
	. venv/bin/activate && \
	lango functions examples/systemo/example.syso

systemo-typecheck: install
	. venv/bin/activate && \
	lango typecheck systemo examples/systemo/example.syso

systemo-compile-mono: install
	. venv/bin/activate && \
	lango compile systemo examples/systemo/example.syso -o build/example.py --strategy monomorphization && \
	python3.13 build/example.py

systemo-compile-dp: install
	. venv/bin/activate && \
	lango compile systemo examples/systemo/example.syso -o build/example.py --strategy dictionary_passing && \
	python3.13 build/example.py

# Profiling
profil-minio-run: install
	. venv/bin/activate && \
	pip install py-spy && \
	py-spy top -- lango run minio examples/minio/example.minio

profil-systemo-run: install
	. venv/bin/activate && \
	pip install py-spy && \
	py-spy top -- lango run systemo examples/syso/example.syso

# Benchmarking
bench-fibonacci: install
	. venv/bin/activate && \
	python3.13 ./benchmark/benchmark.py

# Quality
test: install-dev
	. venv/bin/activate && \
	pytest -vvs

test-minio: install-dev
	. venv/bin/activate && \
	pytest -vvs test/test_minio.py

test-systemo: install-dev
	. venv/bin/activate && \
	pytest -vvs test/test_systemo.py

coverage: install-dev
	. venv/bin/activate && \
	coverage run -m pytest && \
	coverage report -m && \
	coverage html

mypy: install-dev
	mypy lango || true
	mypy test

# Docker
docker-build:
	docker build --target lango -t lango .
	docker build --target test -t lango-test .

docker-clean:
	docker rmi lango lango-test || true

docker-run-minio:
	docker run -it --rm lango run minio examples/minio/example.minio

docker-test:
	docker run -it --rm lango-test
