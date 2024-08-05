build:
	poetry build
	pip install dist/*.tar.gz

build-docs:
	sphinx-build --builder html src-docs build-docs

create-dev:
	rm -rf env
	python3.11 -m venv env
	( \
		. env/bin/activate; \
		pip install -r requirements.txt; \
		poetry install; \
		deactivate; \
	)

activate:
	source ./env/bin/activate

create-docs:
	sphinx-apidoc src --output-dir src-docs --maxdepth 100 --separate

test:
	python3 -m pytest ./tests
