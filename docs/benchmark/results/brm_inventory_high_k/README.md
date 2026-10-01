# How wide a random-effect block the inventory can reach

This artifact answers one question, and it is a *negative* result that was
measured rather than assumed:

> Does any historical-gallery card carry a random-effect block with three or
> more coefficient terms that this stack can benchmark on real data?

It exists because the obvious way to answer it is wrong. Screening the
catalogue for wide blocks and reporting "none are eligible" would silently
conflate three different things — a card with no wide block, a card with a wide
block the surface cannot lower, and a card with a wide block whose *data* is
unreachable. Those have different owners and different fixes, so
`run_brm_high_k_preflight.jl` carries every candidate as far as it actually goes
and records the stage it stopped at.

**No K-based exclusion is applied before testing.** Block width is what is being
measured, so using it as an eligibility filter would make the answer circular.

## Two readings of block width, both scanned

`(x | g)` does not mean the same thing in the historical formula and in the
generated BRM body, so a single width would be ambiguous:

- the **historical** lme4/brms reading, where `(x | g)` carries an *implicit*
  random intercept and correlates it with the slope, so the block is two
  coefficients wide; and
- the **literal** reading BRM's verbatim surface lowers, where `(x | g)` is the
  one term written and the block is one coefficient wide.

A card qualifies as a candidate if **either** reading reaches three. Widths per
block under both readings are recorded per candidate as `historical_blocks` and
`generated_blocks`, so the comparison stays inspectable instead of collapsing to
one number.

## Measured result

- 359 catalogue cards scanned; **3** carry a block of three or more coefficients
  under at least one reading.
- Widest block among rows the published benchmark can actually run: **K = 2
  under both readings**. Recorded as `max_historical_k_among_publishable_rows`
  and `max_generated_k_among_publishable_rows`.
- Total probe time: 104.049 seconds.

None of the three candidates is publishable by the main runner:

| card | historical K | generated K | furthest stage on real data | why it is not in the matrix |
| --- | ---: | ---: | --- | --- |
| `kruschke:income_famsize` | 3 | 3 | `sampled` | support class `already-expressible-via-semantic-rewrite`; the main runner admits only verbatim rows |
| `flocker:single_season_repvarying` | 3 | 0 | `no-real-data-adapter` | dataset receipt is `synthetic`; translation is `route-specific` |
| `flocker:augmented_multispecies` | 3 | 0 | `no-real-data-adapter` | dataset receipt is `synthetic`; translation is `route-specific` |

The one candidate with a fetchable dataset was carried onto that data, and on
this BRM it goes all the way: `kruschke:income_famsize` downloads, adapts,
parses, lowers, instantiates, evaluates a finite log density and gradient, and
samples (164 dimensions, a K = 3 block on `State`). At the previous pins it
stopped at transpile because `nu` was still symbolic; the inventory has since
recovered the historical prior (`nu ~ Exponential(29)`, from the source's
`exponential(rate = 1/29)`) and marks the row `ready`. It stays out of the
published matrix for one remaining reason: it is a `semantic-rewrite` row, not
a verbatim one, and the main runner's gate admits only verbatim rows. That is
now the single remaining gate between the matrix and a K = 3 model.

The two `flocker` rows never had row-level historical data to run: their dataset
receipts are `synthetic`. Their `generated_max_k` of 0 is not a narrower block —
it is the absence of a generated body at all, because `stanblocks_plate`
requires a faithful plate implementation with no ordinary-formula substitute.

## What this does and does not establish

It establishes that **at the recorded pins, the widest random-effect block the
published matrix can carry is two coefficients**, and that this is a property of
the *inventory's* current translation and data reach — not of WarmupHMC, and not
of a runtime or eligibility rule imposed here.

It does not establish that wide blocks are unreachable in principle: one K = 3
row already samples on real data. Admitting it is a question about the main
runner's verbatim-only gate, not about reach; the `flocker` rows still need a
`stanblocks_plate` implementation plus real data.

## Provenance

- Logical compute host: `strato2`; Julia 1.10.11
- WarmupHMC: `133093d0f32c` (src clean; code-identical to the standard artifact's `6adc5ea6`)
- BayesianRegressionModels: `a5e118b0119e29c137d80d887b571db2d77af8e1`
- StanBlocks: `d520980f98cc90141967a47dd6520fb5bb6e3f31`
- `translations.tsv` sha256: `3abb3199a194f74754c8e02b725dfac42081bc7e302c796cdf097ad1504b633d`
- `model_matrix.tsv` sha256: `edeed76f24931af8f53e296538d4d4b98c6768c9c5665f3f4dec55919ae78aeb`
- Runner sha256: `cca69a6173ff7413322020ff2d599d48c954ffdd04d39dbcf031160eba034266`
- `rows.json` sha256: `7c2834f750da7c5f3bd650f71a69ef93de9a7946036352bce44155ba512c1738`
- Probe draws per stage: 50

The two inventory checksums are the same ones the standard-warmup artifact
records, which is what makes the two artifacts comparable: the documentation
page asserts that equality rather than trusting it.

## Reproduction

```sh
julia --startup-file=no --project=/path/to/pinned/environment \
  docs/benchmark/run_brm_high_k_preflight.jl
```

`rows.json` is the source of truth. Everything the documentation says about
block width is derived from it at build time; no screened list or verdict is
stored separately.
