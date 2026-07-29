# Does the historical inventory contain a benchmarkable K>2 random-effect block?
#
# WHY THIS IS A RUNNER AND NOT A FILTER
#
# The published matrix in `run_brm_inventory_benchmark.jl` tops out at K=2 —
# correlated intercept-and-slope. The tempting explanation is "the inventory has
# no K>2 rows", and the tempting implementation is a `K <= 2` guard in the spec
# list. Both would be wrong, and wrong in the way that is hardest to notice: a
# benchmark that never attempted a shape reads exactly like a benchmark that
# attempted it and found nothing.
#
# So this asks the question by RUNNING it. It enumerates every card in the
# catalogue whose historical formula carries a random-effect block with three or
# more coefficient terms, and for each one either takes it all the way to a real
# gradient on real data or records precisely why it could not. The artifact it
# writes is the evidence for the negative claim on the published page.
#
#   julia --startup-file=no --project=/path/to/pinned/environment \
#     docs/benchmark/run_brm_high_k_preflight.jl
#
# It shares the pinned environment, the inventory contract and the download
# helper with the main runner. It is deliberately its OWN file: it must be able
# to attempt rows the main runner's `ready`/`verbatim`/`finite-gradient` gate
# refuses, and relaxing that gate in the main runner to let one candidate through
# would weaken every published row's receipt.

using LinearAlgebra
BLAS.set_num_threads(1)

using BayesianRegressionModels, CSV, DataFrames, Dates, Distributions, Downloads
using Enzyme, JSON, LogDensityProblems, Random, SHA
using StanBlocks, Statistics, WarmupHMC
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan

const OUT = get(ENV, "BRMI_HIGH_K_OUT",
                joinpath(@__DIR__, "results", "brm_inventory_high_k", "rows.json"))
const DATA_CACHE = get(ENV, "BRMI_DATA_CACHE",
                       joinpath(tempdir(), "brm-inventory-data"))
const PROBE_DRAWS = parse(Int, get(ENV, "BRMI_HIGH_K_DRAWS", "50"))

pkgdir_of(m) = dirname(dirname(pathof(m)))
const INVENTORY_DIR = joinpath(pkgdir_of(BRM), "research", "historical_model_inventory")
const TRANSLATIONS = joinpath(INVENTORY_DIR, "translations.tsv")
const MODEL_MATRIX = joinpath(INVENTORY_DIR, "model_matrix.tsv")

tsv_unescape(value) = replace(value,
    "\\t" => "\t", "\\n" => "\n", "\\r" => "\r", "\\\\" => "\\")

function read_tsv(path)
    lines = readlines(path)
    columns = split(first(lines), '\t'; keepempty=true)
    [Dict(columns .=> tsv_unescape.(split(line, '\t'; keepempty=true)))
     for line in Iterators.drop(lines, 1)]
end

const TRANSLATION_ROWS = read_tsv(TRANSLATIONS)
const MATRIX_ROWS = read_tsv(MODEL_MATRIX)

const RE_BLOCK_PATTERN = r"\(([^()|]*(?:\([^()]*\)[^()|]*)*)\|\|?([^()]*)\)"

block_terms(inner) = filter(!isempty, strip.(split(inner, '+')))

"""
Block widths under the HISTORICAL `lme4`/`brms` reading, in source order.

An intercept is implied unless the block suppresses it with `0` or `-1`, so
`(x | g)` is width two here and `(1 + x + I(x^2) || g)` is width three. The inner
alternative of the pattern tolerates ONE level of nesting so `I(x^2)` and
`protect(x^2)` count as a single term rather than splitting the block, and both
`|` and `||` are matched — an uncorrelated block has the same width as its
correlated twin, and pre-filtering `||` away is exactly the kind of silent
exclusion this file exists to avoid.
"""
function historical_block_widths(formula)
    out = NamedTuple[]
    for m in eachmatch(RE_BLOCK_PATTERN, formula)
        terms = block_terms(m.captures[1])
        suppressed = any(t -> t in ("0", "-1"), terms)
        slopes = count(t -> !(t in ("0", "1", "-1")), terms)
        push!(out, (k = slopes + (suppressed ? 0 : 1), terms = strip(m.captures[1]),
                    group = strip(m.captures[2])))
    end
    out
end

"""
Block widths as BRM's verbatim surface actually lowers them, in source order.

Literal term count: BRM adds no implicit intercept, so `(x | g)` is width one
here while the historical reading above calls the same text width two. Both are
computed because they answer different questions — the historical width says what
the publication fitted, the generated width says what this stack would sample —
and a K>2 screen that used only one of them would miss candidates in one
direction and invent them in the other.
"""
generated_block_widths(body) =
    [(k = count(t -> t != "0", block_terms(m.captures[1])),
      terms = strip(m.captures[1]), group = strip(m.captures[2]))
     for m in eachmatch(RE_BLOCK_PATTERN, body)]

max_width(blocks) = maximum((b.k for b in blocks); init = 0)

function dense_int(v)
    levels = sort(unique(v))
    code = Dict(x => i for (i, x) in enumerate(levels))
    [code[x] for x in v]
end

function cached_download(filename, url; expected_sha256=nothing)
    mkpath(DATA_CACHE)
    path = joinpath(DATA_CACHE, filename)
    isfile(path) || Downloads.download(url, path)
    actual = bytes2hex(open(sha256, path))
    isnothing(expected_sha256) || actual == expected_sha256 || error(
        "dataset checksum mismatch for $url: expected $expected_sha256, got $actual",
    )
    path
end

"""
Kruschke's family-income-by-family-size data, z-scored as the book fits it.

`se_z` is the SampErr divided by the SAME standard deviation used to scale the
response, not by its own — it is a known standard error attached to the response,
so it has to land on the response's scale to mean anything in
`se(se_z, sigma = TRUE)`.
"""
function income_famsize_adapter(df)
    income = Float64.(df.MedianIncome)
    income_sd = std(income)
    family_size = Float64.(df.FamilySize)
    (; median_income_z = (income .- mean(income)) ./ income_sd,
       se_z = Float64.(df.SampErr) ./ income_sd,
       family_size_z = (family_size .- mean(family_size)) ./ std(family_size),
       State = dense_int(String.(df.State)))
end

# Only rows with a real, fetchable, row-level dataset get an adapter. A row
# without one cannot be preflighted on actual data at all, and saying so is a
# different finding from "it was tried and the surface refused it".
const ADAPTERS = Dict(
    "kruschke:income_famsize" => (
        dataset = "income_famsize",
        url = "https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-" *
              "brms-and-the-tidyverse/master/data.R/IncomeFamszState3yr.csv",
        sha256 = "88f508e44c65442cebc8b07cf24d9708968a29880c789abbdae66931b2705c83",
        read_options = (; comment = "#"),
        note = "Leading census-URL comment line skipped. median_income_z and " *
               "family_size_z are z-scores; se_z = SampErr / sd(MedianIncome) so the " *
               "known response SE shares the response's scale; State densely recoded",
        adapt = income_famsize_adapter,
    ),
)

"""
Take one candidate as far as it goes, and record exactly where it stopped.

The stages are cumulative and each one is reported by name, because the answer
to "is there a benchmarkable K>2 row" depends on WHICH stage refuses: a surface
that cannot express the formula is a BRM-surface gap, while a body that lowers
and then produces a non-finite gradient is a numerical one, and they call for
different follow-up work.
"""
function probe(row_key, body, groups)
    stages = String[]
    record(stage) = (push!(stages, stage); stage)
    try
        adapter = get(ADAPTERS, row_key, nothing)
        isnothing(adapter) && return (; reached = "no-real-data-adapter", stages,
                                      error = "", data_sha256 = "", n_obs = 0,
                                      dimension = 0, log_density = nothing,
                                      gradient_finite = nothing, sampled = false)
        path = cached_download(adapter.dataset * ".csv", adapter.url;
                               expected_sha256 = adapter.sha256)
        df = CSV.read(path, DataFrame; adapter.read_options...)
        data = adapter.adapt(df)
        record("data-adapted")

        brmi = Core.eval(@__MODULE__, BRM._brm(body; df = data))
        record("brm-parsed")
        sb = SBBRMI(brmi; mod = @__MODULE__, centered_groups = Symbol[])
        record("sbbrmi-lowered")
        problem = StanBlocks.stan_instantiate(sb.model)
        record("bridgestan-instantiated")

        dimension = LogDensityProblems.dimension(problem)
        position = zeros(dimension)
        value = LogDensityProblems.logdensity(problem, position)
        record("log-density-evaluated")
        _, gradient = LogDensityProblems.logdensity_and_gradient(problem, position)
        finite = isfinite(value) && all(isfinite, gradient)
        record("gradient-evaluated")

        sampled = false
        if finite
            adaptive_warmup_mcmc(Xoshiro(1), problem; n_draws = PROBE_DRAWS,
                                 nonlinear_adapt = true, progress = nothing)
            record("sampled")
            sampled = true
        end
        (; reached = last(stages), stages, error = "",
         data_sha256 = bytes2hex(open(sha256, path)),
         n_obs = length(first(data)), dimension,
         log_density = isfinite(value) ? value : nothing,
         gradient_finite = finite, sampled)
    catch err
        (; reached = isempty(stages) ? "download" : "after-" * last(stages), stages,
         error = first(sprint(showerror, err, catch_backtrace()), 2000),
         data_sha256 = "", n_obs = 0, dimension = 0, log_density = nothing,
         gradient_finite = nothing, sampled = false)
    end
end

"""
One line naming the PROXIMATE reason a candidate is not in the published matrix.

A MEASURED failure outranks a status field. When the probe actually got the card
onto real data and something threw, that exception is the finding — the inventory's
own status is a separate recorded column, and quoting it in place of the error
would turn "we ran it and it stopped here" back into "we read a label". The status
and receipt reasons are the fallback for cards that could not be attempted at all.

Empty only when the candidate ran end to end, which is the outcome that would
oblige the published matrix to include it.
"""
function blocker_of(matrix, translation, receipts, result)
    result.reached == "sampled" && return ""
    if !isempty(result.error)
        first_line = first(eachsplit(result.error, '\n'))
        caused = filter(l -> occursin("Caused by:", l), split(result.error, '\n'))
        return "stopped after `$(isempty(result.stages) ? "download" : last(result.stages))`: " *
               strip(first_line) *
               (isempty(caused) ? "" : " — " * strip(first(caused)))
    end
    status = matrix["inferred_translation_status"]
    support = matrix["inferred_surface_support_class"]
    note = strip(get(translation, "translation_note", ""))
    if result.reached == "no-real-data-adapter"
        return strip(receipts) == "synthetic" ?
            "dataset receipt is synthetic, so there is no row-level historical " *
            "data to run; translation is `$(status)`" *
            (isempty(note) ? "" : ": " * note) :
            "no row-level dataset receipt this runner can fetch; translation is " *
            "`$(status)`"
    end
    status == "ready" ||
        return "translation is `$(status)`" * (isempty(note) ? "" : ": " * note)
    support == "already-expressible-verbatim" ||
        return "surface support is `$(support)`" *
               (isempty(get(matrix, "inferred_surface_secondary_gap", "")) ? "" :
                " (" * matrix["inferred_surface_secondary_gap"] * ")")
    "reached `$(result.reached)` without completing a sample"
end

sanitize(x) = x isa AbstractFloat && !isfinite(x) ? nothing : x
sanitize(x::AbstractDict) = Dict(k => sanitize(v) for (k, v) in x)
sanitize(x::AbstractVector) = [sanitize(v) for v in x]

gitsha(dir) = try readchomp(`git -C $dir rev-parse HEAD`) catch; "unavailable" end
gitdirty(dir, paths...) = try
    !isempty(readchomp(`git -C $dir status --porcelain -- $(collect(paths))`))
catch
    true
end

function main()
    started = time()
    whmc_dir = pkgdir_of(WarmupHMC)
    brm_dir = pkgdir_of(BRM)

    candidates = Any[]
    max_ready_historical_k = 0
    max_ready_generated_k = 0
    n_scanned = 0
    for matrix in MATRIX_ROWS
        n_scanned += 1
        formula = matrix["formula_claim"]
        ready = matrix["inferred_translation_status"] == "ready" &&
                matrix["inferred_surface_support_class"] == "already-expressible-verbatim" &&
                matrix["inferred_capability_tier"] == "bridgestan-finite-density-gradient"

        row_key = matrix["row_key"]
        source, key = split(row_key, ':'; limit = 2)
        translation = only(r for r in TRANSLATION_ROWS
                           if r["source"] == source && r["key"] == key &&
                              r["variant"] == "inferred-family")
        body = translation["current_brm_body"]
        historical = historical_block_widths(formula)
        generated = generated_block_widths(body)
        historical_k = max_width(historical)
        generated_k = max_width(generated)
        if ready
            max_ready_historical_k = max(max_ready_historical_k, historical_k)
            max_ready_generated_k = max(max_ready_generated_k, generated_k)
        end
        # EITHER reading qualifies. Screening on only the generated width would
        # hide a card whose publication fitted a wide block that the verbatim
        # surface silently narrowed; screening on only the historical width would
        # claim a wide candidate this stack would never actually sample that wide.
        max(historical_k, generated_k) >= 3 || continue

        groups = filter(!isempty, split(translation["group_columns"], ','))
        receipts = matrix["dataset_receipt_urls"]

        result = probe(row_key, body, groups)
        push!(candidates, Dict(
            "spec" => row_key,
            "historical_formula" => formula,
            "historical_blocks" =>
                [Dict("k" => b.k, "terms" => b.terms, "group" => b.group)
                 for b in historical],
            "generated_blocks" =>
                [Dict("k" => b.k, "terms" => b.terms, "group" => b.group)
                 for b in generated],
            "historical_max_k" => historical_k,
            "generated_max_k" => generated_k,
            "max_k" => max(historical_k, generated_k),
            "current_brm_body" => body,
            "inventory_group_columns" => string.(groups),
            "translation_status" => matrix["inferred_translation_status"],
            "surface_support_class" => matrix["inferred_surface_support_class"],
            "capability_tier" => matrix["inferred_capability_tier"],
            "surface_secondary_gap" => get(matrix, "inferred_surface_secondary_gap", ""),
            "translation_note" => get(translation, "translation_note", ""),
            "semantic_route" => get(matrix, "semantic_route", ""),
            "source_fidelity_verdict" => get(matrix, "source_fidelity_verdict", ""),
            "dataset_claim" => matrix["dataset_claim"],
            "dataset_support" => get(matrix, "dataset_support", ""),
            "dataset_receipt_urls" => receipts,
            "dataset_is_synthetic" => strip(receipts) == "synthetic",
            "row_source_claim" => get(matrix, "row_source_claim", ""),
            "publishable_by_main_runner" => ready,
            "data_adapter" => haskey(ADAPTERS, row_key) ?
                ADAPTERS[row_key].note : "",
            "reached" => result.reached,
            "blocker" => blocker_of(matrix, translation, receipts, result),
            "stages_completed" => result.stages,
            "n_obs" => result.n_obs,
            "dimension" => result.dimension,
            "log_density" => result.log_density,
            "gradient_finite" => result.gradient_finite,
            "sampled" => result.sampled,
            "data_sha256" => result.data_sha256,
            "error" => result.error,
        ))
        @info "high-K candidate probed" row_key historical_k generated_k reached=result.reached
        flush(stderr)
    end

    config = Dict(
        "host" => get(ENV, "KB_HOST", gethostname()),
        "julia" => string(VERSION),
        "runner" => "docs/benchmark/run_brm_high_k_preflight.jl",
        "runner_sha256" => bytes2hex(open(sha256, @__FILE__)),
        "reproduction" => "julia --startup-file=no --project=/path/to/pinned/environment " *
            "docs/benchmark/run_brm_high_k_preflight.jl",
        "question" => "Does any historical-gallery card carry a random-effect block with " *
            "three or more coefficient terms that this stack can benchmark on real data?",
        "method" => "every catalogue card is scanned under BOTH block-width readings — the " *
            "historical lme4/brms one, where `(x | g)` implies an intercept, and the " *
            "literal one BRM's verbatim surface lowers — and a card qualifies if EITHER " *
            "is at least three. Each hit is then taken as far as it goes on ACTUAL data: " *
            "download, adapter, BRM._brm, SBBRMI, BridgeStan instantiation, log density, " *
            "gradient, a short sample. The stage it stopped at is recorded. No K-based " *
            "exclusion is applied before testing.",
        "warmuphmc_sha" => gitsha(whmc_dir),
        "warmuphmc_src_dirty" => gitdirty(whmc_dir, "src"),
        "brm_sha" => gitsha(brm_dir),
        "stanblocks_sha" => gitsha(pkgdir_of(StanBlocks)),
        "translations_sha256" => bytes2hex(open(sha256, TRANSLATIONS)),
        "model_matrix_sha256" => bytes2hex(open(sha256, MODEL_MATRIX)),
        "n_cards_scanned" => n_scanned,
        "n_high_k_cards" => length(candidates),
        "max_historical_k_among_publishable_rows" => max_ready_historical_k,
        "max_generated_k_among_publishable_rows" => max_ready_generated_k,
        "probe_draws" => PROBE_DRAWS,
        "generated_at" => string(now()),
        "total_elapsed_s" => round(time() - started; digits=3),
    )

    mkpath(dirname(OUT))
    temporary = OUT * ".tmp"
    open(temporary, "w") do io
        JSON.print(io, Dict("config" => sanitize(config),
                            "candidates" => sanitize(candidates)), 2)
    end
    mv(temporary, OUT; force=true)
    @info "wrote high-K preflight" OUT n=length(candidates) max_ready_historical_k max_ready_generated_k
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
