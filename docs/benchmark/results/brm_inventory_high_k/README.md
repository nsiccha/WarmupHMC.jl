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
- Total probe time: 19.068 seconds.

None of the three candidates is publishable, and each stops somewhere different:

| card | historical K | generated K | furthest stage on real data | what stopped it |
| --- | ---: | ---: | --- | --- |
| `kruschke:income_famsize` | 3 | 3 | `after-sbbrmi-lowered` | `StanBlocksError [transpile]: model` — `Could not find nu in model, builtin, Main or Main!` |
| `flocker:single_season_repvarying` | 3 | 0 | `no-real-data-adapter` | dataset receipt is `synthetic`; translation is `route-specific` |
| `flocker:augmented_multispecies` | 3 | 0 | `no-real-data-adapter` | dataset receipt is `synthetic`; translation is `route-specific` |

The one candidate with a fetchable dataset was carried onto that data:
`kruschke:income_famsize` downloaded, adapted, parsed, and lowered through
BRM's surface, then failed to transpile because `nu` is still symbolic. That is
exactly the limitation the inventory's own translation note predicts — *"the
surface is executable, but `nu` remains symbolic until the historical
degrees-of-freedom value/prior is recovered"* — so the stop is a confirmation of
a known gap, not a new discovery. It is a `semantic-rewrite` row and therefore
outside the published benchmark's readiness gate regardless of its width.

The two `flocker` rows never had row-level historical data to run: their dataset
receipts are `synthetic`. Their `generated_max_k` of 0 is not a narrower block —
it is the absence of a generated body at all, because `stanblocks_plate`
requires a faithful plate implementation with no ordinary-formula substitute.

## What this does and does not establish

It establishes that **at the recorded pins, the widest random-effect block the
published matrix can carry is two coefficients**, and that this is a property of
the *inventory's* current translation and data reach — not of WarmupHMC, and not
of a runtime or eligibility rule imposed here.

It does not establish that wide blocks are unreachable in principle. Two of the
three stops are upstream work items with named owners: recovering the historical
degrees-of-freedom for `kruschke:income_famsize`, and a `stanblocks_plate`
implementation plus real data for the `flocker` rows.

## Provenance

- Logical compute host: `strato2`; Julia 1.10.11
- WarmupHMC: `dc9636f8e8cdf6437e69980b3336d2e5659fecbf` (src clean)
- BayesianRegressionModels: `d452e97a90974d5bef5472978f3d2379255fbfbc`
- StanBlocks: `7a02d30ffb28215e79470ce9689c0f65902b10df`
- `translations.tsv` sha256: `8443c0a15bfe22bee55bd241dbf9b26c4ffe3de6596ddb71d2213d632a597f25`
- `model_matrix.tsv` sha256: `1e76f72dd7ec518e29c7e6fd0875a51f959842059453228c1d0146be4a6bc9d3`
- Runner sha256: `9ac61aef1a1733654839ebcc306ab606f7332c7fdc5fc0e2933bce9b9ff777e5`
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
