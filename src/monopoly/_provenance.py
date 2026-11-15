from __future__ import annotations

import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from monopoly._design import DesignMatrices
    from monopoly.dataset import MonopolyDataset


def build_provenance(
    dataset: MonopolyDataset,
    design: DesignMatrices,
    *,
    fit: dict[str, Any],
    contrast: dict[str, Any],
) -> dict[str, Any]:
    """Build a reproducible analysis record."""
    return {
        'software': {
            'monopolyPolysome': _package_version('monopoly-polysome'),
            'formulaic': _package_version('formulaic'),
            'numpy': _package_version('numpy'),
            'pandas': _package_version('pandas'),
            'pydeseq2': _package_version('pydeseq2'),
            'python': platform.python_version(),
        },
        'inputs': {
            'participants': len(dataset.participants),
            'genes': len(dataset.genes),
            'fingerprints': _dataset_fingerprints(dataset),
        },
        'design': {
            'reference': design.reference,
            'test': design.test,
            'covariates': list(design.covariates),
            'fullFormula': design.full_formula,
            'reducedFormula': design.reduced_formula,
            'fullColumns': design.full.columns.tolist(),
            'residualDegreesOfFreedom': design.residual_degrees_of_freedom,
        },
        'fit': fit.copy(),
        'contrast': contrast.copy(),
    }


def _dataset_fingerprints(dataset: MonopolyDataset) -> dict[str, str | None]:
    return {
        fraction: _frame_fingerprint(dataset._counts[fraction])
        for fraction in ('input', 'monosome', 'polysome')
    } | {
        'metadata': _frame_fingerprint(dataset._metadata),
        'libraryMetadata': _frame_fingerprint(dataset._library_metadata),
        'geneMetadata': (
            None
            if dataset._gene_metadata is None
            else _frame_fingerprint(dataset._gene_metadata)
        ),
    }


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode())
    digest.update('\0'.join(map(str, frame.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(frame.index).values.tobytes())
    digest.update(pd.util.hash_pandas_object(frame.columns).values.tobytes())
    try:
        values = pd.util.hash_pandas_object(frame, index = True).values
    except TypeError:
        values = pd.util.hash_pandas_object(
            frame.map(repr),
            index = True,
        ).values
    digest.update(values.tobytes())
    return f'sha256:{digest.hexdigest()}'


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:  # pragma: no cover
        return 'unknown'
