# Releasing torq-bench

This repo publishes the `torq-bench` package to PyPI. Release from `main` only.

## Preflight

1. Make sure the version in `pyproject.toml` is the version you intend to publish.
2. Commit and push `main`.
3. Run the repo preflight:

```bash
python scripts/release_preflight.py
```

The preflight fails if:

- the current branch is not `main`
- the worktree has meaningful tracked changes or untracked files
- `HEAD` does not match `origin/main`
- the version in `pyproject.toml` already exists on PyPI
- `build` or `twine` is missing in the current Python environment

## Release Environment

Use a fresh virtualenv for release work:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install build twine pytest torch pennylane
python -m pip install -e .[pennylane]
```

`torq-quantum>=0.1.3` is a runtime dependency. Installing `pennylane` lets you run the parity tests before shipping.

## Validate Before Upload

Run the preflight again from the release environment:

```bash
python scripts/release_preflight.py
```

Run the test suite:

```bash
pytest -q -rs
```

Build the artifacts and verify the metadata:

```bash
python -m build
python -m twine check dist/*
```

## Smoke Install The Built Wheel

This checks the exact wheel you are about to upload:

```bash
python -m venv /tmp/torq-bench-smoke
source /tmp/torq-bench-smoke/bin/activate
python -m pip install -U pip
python -m pip install "torq-quantum>=0.1.3"
python -m pip install dist/torq_bench-<VERSION>-py3-none-any.whl
python - <<'PY'
import importlib.metadata as md
import torq_bench

print("metadata version:", md.version("torq-bench"))
print("import version:", torq_bench.__version__)
PY
deactivate
```

Replace `<VERSION>` with the version from `pyproject.toml`.

## Upload

Upload to TestPyPI first if you want an extra dry run:

```bash
python -m twine upload --repository testpypi dist/*
```

Then publish to PyPI:

```bash
python -m twine upload dist/*
```

## After Publish

Verify the public package page and install from PyPI:

```bash
python -m pip install --upgrade "torq-bench[pennylane]"
python - <<'PY'
import importlib.metadata as md
import torq_bench

print("metadata version:", md.version("torq-bench"))
print("import version:", torq_bench.__version__)
PY
```

If the release is correct, tag it:

```bash
git tag -a v<VERSION> -m "Release <VERSION>"
git push origin main --tags
```
