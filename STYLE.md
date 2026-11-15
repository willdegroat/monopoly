# Monopoly style

Monopoly should remain small, explicit, and unsurprising.

Scientific terminology and assumptions are defined in `docs/METHODS.md`.

## Design

- Keep the public workflow centered on `MonopolyDataset`, `MonopolyModel`, and
  `MonopolyResults`.
- Prefer plain functions to inheritance, registries, factories, or plugin systems.
- Add an abstraction only when it removes repeated behavior that already exists.
- Use established numerical libraries whenever they can express the required method.
- Use PyDESeq2 for negative-binomial estimation and established scientific
  libraries for numerical work.
- Validate at public boundaries. Internal functions may rely on validated inputs.
- Never guess identifiers, silently remove observations, or mutate user objects.

## Source

- Use single-quoted strings. Docstrings use triple double quotes.
- Limit lines to 88 characters.
- Include spaces around keyword and default assignments.
- Use concise one-line docstrings ending in periods for public APIs.
- Use descriptive phrases for classes and imperative sentences for functions and
  methods.
- Add comments only when they explain scientific or non-obvious algorithmic reasoning.
- Use `snake_case` internally and camelCase in exported scientific tables.
- Use Ruff for linting and line-length enforcement. Format manually because standard
  Python formatters remove the approved spaces around keyword assignments.

## Errors

- Raise a dedicated Monopoly exception with an actionable message.
- Stop for invalid experiments or global model failures.
- Preserve valid genome-wide fits when isolated genes fail, and report those failures
  through diagnostics.

## Tests

- Test public contracts, scientific invariants, and failure modes.
- Do not reproduce dependency test suites or preserve tests for rejected methods.
- Use fixed random seeds only for deterministic numerical contract tests.
- A successful test suite establishes implementation behavior, not universal
  statistical calibration.
