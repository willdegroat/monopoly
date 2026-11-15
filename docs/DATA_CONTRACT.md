# Data contract

## Count matrices

`input`, `monosome`, and `polysome` must be pandas data frames with participants in
rows and stable gene identifiers in columns. All three matrices must have identical
unique participant and gene identifiers; column and row order may differ.

Values must be finite, nonnegative whole numbers within the signed 64-bit integer
range. Monopoly rejects TPM, fractional counts, missing values, Boolean columns,
duplicate identifiers, and silently mismatched experiments.

For matrix-only Salmon input, use gene-level `lengthScaledTPM` count-scale values and
round once upstream. This supports count models without sample-specific transcript-
length offsets. Do not round TPM and present it as counts.

Apply one condition-blind gene filter to the aligned three-matrix collection. A
reasonable rule should retain genes with useful counts in enough participants to
support the intended comparison. Monopoly does not choose that biological threshold
for the user.

## Participant metadata

`metadata` must use participant identifiers as its index and contain a lowercase
`condition` column. Version 0.1 accepts exactly two nonempty condition levels with at
least two participants per level.

The `reference` supplied to `MonopolyModel` must match one condition. The other level
is the test condition.

## Optional metadata

`gene_metadata` uses gene identifiers as its index. A `geneName` column is copied to
the result table.

`library_metadata` contains one row for every participant and fraction and must
include `participantId` and `fraction`. Additional measured covariates may be modeled
only if they vary within participants and leave the design identifiable.

Monopoly never infers participants, conditions, fractions, or covariates from file
names.
