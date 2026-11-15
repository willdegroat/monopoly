# Release Monopoly

The GitHub repository is `willdegroat/monopoly`; the Python distribution is
`monopoly-polysome`, and the import is `monopoly`.

## Validate a release

```console
python -m pip install -e ".[dev]" twine
python -m ruff check .
python -m pytest --cov=monopoly
python -m build
python -m twine check dist/*
```

Install the wheel into a clean environment and run the public workflow. Inspect
the distribution contents for local data or credentials. Keep scientific
benchmark calibration separate from implementation tests; passing tests do not
establish FDR control for every experiment.

## Publish to PyPI

For the first release, register a pending Trusted Publisher in the PyPI account
that will own the project:

| Field | Value |
| --- | --- |
| PyPI project | `monopoly-polysome` |
| GitHub owner | `willdegroat` |
| GitHub repository | `monopoly` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

See the [PyPI instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
No long-lived PyPI token is required. Once CI passes, create a GitHub release
for the tested version tag. The release workflow rebuilds and validates the
distributions before publishing with the configured identity. Running it manually
builds artifacts without publishing. Confirm the uploaded version and install it
from PyPI before announcing the release.

Keep Git author and pusher identity associated with the maintainer's account.
Never commit credentials, local environments, or unreleased patient-level data.
