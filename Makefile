.PHONY:
	prep
	uninstall
	install
	install-dev
	test
	run

all: run

prep:
	python3.13 -m venv .venv

test: install
	. .venv/bin/activate && \
	pytest -vvs

uninstall:
	.venv/bin/pip uninstall lango -y

install: uninstall
	.venv/bin/pip install .

install-dev:
	.venv/bin/pip install -e .[dev]

run-dev:
	. .venv/bin/activate && \
		lango

run:
	. .venv/bin/activate && \
	lango run --input_file examples/minio/example.minio

type: install
	. .venv/bin/activate && \
	lango typecheck test/files/minio/datatypes/custom/constrcutor/mixed.minio
	# lango typecheck examples/minio/example.minio
