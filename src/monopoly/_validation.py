from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from monopoly.exceptions import ModelFitError


def validate_control_genes(
    control_genes: Collection[str] | None,
    *,
    genes: pd.Index,
) -> list[str] | None:
    """Validate control genes and restore canonical gene order."""
    if control_genes is None:
        return None
    if isinstance(control_genes, str):
        raise ModelFitError('control_genes must be a collection of gene identifiers.')

    try:
        controls = list(control_genes)
    except TypeError as error:
        raise ModelFitError(
            'control_genes must be a collection of gene identifiers.',
        ) from error
    if not controls:
        raise ModelFitError('control_genes cannot be empty.')
    if any(not isinstance(gene, str) or not gene.strip() for gene in controls):
        raise ModelFitError('control_genes must contain nonempty strings.')
    if len(controls) != len(set(controls)):
        raise ModelFitError('control_genes cannot contain duplicates.')
    unknown = pd.Index(controls).difference(genes)
    if not unknown.empty:
        raise ModelFitError(f'Unknown control_genes: {unknown.tolist()}.')

    control_set = set(controls)
    return [gene for gene in genes if gene in control_set]
