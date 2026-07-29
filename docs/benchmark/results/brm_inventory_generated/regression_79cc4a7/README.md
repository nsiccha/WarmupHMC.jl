# Rejected last-bit-drift reference

This artifact was recorded against BayesianRegressionModels tree
`79cc4a75906445d925db810c45c3384c262fb313`. It completed all 216
inventory-generated benchmark rows, but it is not valid performance evidence:
the scalar adaptive-centering accessor changed the legacy floating-point
operation tree by one ULP and therefore did not preserve the exact Enzyme
gradient semantics.

It remains checked in only as regression evidence. Compared with the corrected
canonical BRM merge `aed667cfdb304718978750251547819bf5120bda`, every
non-timing row field is identical in this three-model run, but the invalid
accessor made the adaptive wrapper appear 3.7×–7.4× faster. Do not quote its
runtime or ESS-per-second results as supported performance.

The source artifact had SHA-256
`bdaad8b27bd1b07e57ea2e8a4e88bd7c9c3f80a81e15cff6d6284da199a0ee05`.
The checked-in copy differs only by a normalized final newline and has SHA-256
`14e8119e95aa6844a4c8627a5e3e3c3dd69f66b7e93bfbb16f9168cc38ee8da2`.
