.PHONY: check test stubs
test:
	uv run pytest
check:
	uv run ruff format src tests benchmarks
	uv run ruff check --fix src tests benchmarks
	clang-format -i src/mjbatch/csrc/*
	uv run pyright
stubs:
	bash typings/generate_mujoco_stubs.sh
	uv run python -m nanobind.stubgen -m mjbatch._bindings -o src/mjbatch/_bindings.pyi
