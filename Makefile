.PHONY:
	prep
	uninstall
	install
	install-dev
	test
	run

prep:
	python3.13 -m venv .venv

test: install
	. .venv/bin/activate && \
	pytest

uninstall:
	.venv/bin/pip uninstall system_o -y

install: uninstall
	.venv/bin/pip install .

install-dev:
	.venv/bin/pip install -e .[dev]

run-dev:
	. .venv/bin/activate && \
		system_o

run:
	. .venv/bin/activate && \
	system_o --help
