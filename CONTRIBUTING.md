# Contributing

Create an isolated Python 3.11 or later environment and install the development
dependencies:

```console
python -m pip install -e ".[dev]"
```

Before submitting a change, run:

```console
python -m ruff check .
python -m pytest --cov=monopoly
python -m build
```

Public API changes require tests and corresponding updates to the README and method
documentation. Statistical changes must define their estimand, assumptions, failure
modes, and validation criteria before implementation. Do not promote a method based
only on additional discoveries in one biological dataset.
