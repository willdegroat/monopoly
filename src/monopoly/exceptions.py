class MonopolyError(Exception):
    """Base exception for Monopoly."""


class DataValidationError(MonopolyError):
    """Raised when dataset inputs do not satisfy the data contract."""


class DesignError(MonopolyError):
    """Raised when an experimental design cannot be estimated."""


class ContrastError(MonopolyError):
    """Raised when a requested contrast is invalid."""


class ModelFitError(MonopolyError):
    """Raised when a model-level fit cannot be completed."""

