# Benchmark real data through BRM's landed historical-inventory translations.
#
# Unlike `run_brm_catalogue_benchmark.jl`, this runner contains no copied model
# formula. It selects ready rows from BRM's checked-in `translations.tsv`,
# evaluates each row's `current_brm_body` through BRM._brm, and lowers that BRMI
# through SBBRMI. Only the real-data adapters live here because the inventory's
# executable probe currently supplies synthetic data and has no generic
# real-data loader.
#
#   BRMI_SEEDS=12 BRMI_DRAWS=500 julia --startup-file=no \
#     --project=/path/to/pinned/environment \
#     docs/benchmark/run_brm_inventory_benchmark.jl
#
# Set `BRMI_MODE=standard` to compare WarmupHMC with DynamicHMC's default,
# Stan-style warmup on the same generated non-centered/centered targets. That
# mode counts gradients at one shared LogDensityProblems boundary for every
# sampler and runs the full six-arm linear/nonlinear comparison matrix instead
# of comparing sampler-specific internal counters.

using LinearAlgebra
BLAS.set_num_threads(1)

using BayesianRegressionModels, CSV, DataFrames, Dates, Distributions, Downloads
using Enzyme, JSON, LogDensityProblems, MCMCDiagnosticTools, Random, SHA
using StanBlocks, Statistics, WarmupHMC
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const AD_BACKEND = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
                                function_annotation=Enzyme.Const)

const N_SEEDS = parse(Int, get(ENV, "BRMI_SEEDS", "12"))
const N_DRAWS = parse(Int, get(ENV, "BRMI_DRAWS", "500"))
const PREFLIGHT_DRAWS = parse(Int, get(ENV, "BRMI_PREFLIGHT_DRAWS", "50"))
const MODE = get(ENV, "BRMI_MODE", "inventory")
MODE in ("inventory", "standard") ||
    error("BRMI_MODE must be inventory or standard, got $(repr(MODE))")
const OUT = get(ENV, "BRMI_OUT", joinpath(
    @__DIR__, "results",
    MODE == "standard" ? "brm_inventory_standard" : "brm_inventory_generated",
    "rows.json"))
const DATA_CACHE = get(ENV, "BRMI_DATA_CACHE", joinpath(tempdir(), "brm-inventory-data"))
const RADON_SRRS2_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/" *
    "5ae7aab3113eb8caa7b95faddb2ecefeaa0f7c9f/examples/data/srrs2.dat"
const RADON_CTY_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/" *
    "5ae7aab3113eb8caa7b95faddb2ecefeaa0f7c9f/examples/data/cty.dat"
const RADON_SRRS2_SHA256 =
    "241219ed05d4bf7dc171ca279c69db7f8d87ae666feae86161c973483a054877"
const RADON_CTY_SHA256 =
    "5aa648547b9b565d77b9f55defd2f292520441cc6df7a27206802006dade7b63"

pkgdir_of(m) = dirname(dirname(pathof(m)))
const INVENTORY_DIR = joinpath(pkgdir_of(BRM), "research", "historical_model_inventory")
const TRANSLATIONS = joinpath(INVENTORY_DIR, "translations.tsv")
const MODEL_MATRIX = joinpath(INVENTORY_DIR, "model_matrix.tsv")

"""
One benchmarked inventory row and the consumer-side data adapter it needs.

Only `adapter_note`, `adapt` and the receipt fields are authored here: the model
body, its grouping factors and every support/fidelity label come from BRM's
inventory. `coverage` records WHY the row is in the published set — the
structural feature it contributes that no other benchmarked row does — so the
set can be audited for coverage rather than only for cheapness.

`sha256` pins the RAW upstream file. It is not decoration: three of the tranche
datasets are served by redirecting hosts (OSF, figshare, an ndownloader alias),
where a silently re-published file would otherwise change the posterior without
changing anything recorded in the artifact.
"""
Base.@kwdef struct InventorySpec
    source::String
    key::String
    dataset::String
    url::String
    adapter_note::String
    adapt::Function
    sha256::String = ""
    coverage::String = ""
    # `CSV.read` keywords. Needed because the historical receipts are not all
    # comma-separated: the two `bruno.nicenboim.me` OSF downloads are tab- and
    # semicolon-separated respectively.
    read_options::NamedTuple = NamedTuple()
    # Second file some historical loaders join against, recorded verbatim in the
    # artifact so the join input carries a checksum of its own.
    auxiliary::Dict{String,String} = Dict{String,String}()
    # Departures from the cited source's own coding that the verbatim BRM
    # surface forces on the consumer. Empty when there are none.
    categorical_departures::String = ""
end

function dense_int(v)
    levels = sort(unique(v))
    code = Dict(x => i for (i, x) in enumerate(levels))
    [code[x] for x in v]
end

"""
Two-level factor as a 0/1 treatment contrast on the sorted levels.

`dense_int` is right for grouping factors, where `1:n_levels` is exactly what
BRM wants, and wrong for a two-level *predictor*: it emits 1/2, which relocates
the intercept away from the reference level the historical `brm` call reported.
The reference level is the first in sorted order, which is also R's default for
an unordered factor.
"""
function treatment_contrast(v)
    levels = sort(unique(v))
    length(levels) == 2 || error(
        "treatment_contrast expects a two-level factor, got $(length(levels)) levels",
    )
    [x == first(levels) ? 0.0 : 1.0 for x in v]
end

parse_int(x) = parse(Int, strip(string(x)))

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

function radon_adapter(srrs2)
    cty_path = cached_download("radon_cty.dat", RADON_CTY_URL;
                               expected_sha256=RADON_CTY_SHA256)
    cty = CSV.read(cty_path, DataFrame)

    mn = copy(srrs2[strip.(String.(srrs2.state)) .== "MN", :])
    mn.log_radon = log.(Float64.(mn.activity) .+ 0.1)
    mn.fips = parse_int.(mn.stfips) .* 1000 .+ parse_int.(mn.cntyfips)
    cty.fips = parse_int.(cty.stfips) .* 1000 .+ parse_int.(cty.ctfips)
    cty.log_u = log.(Float64.(cty.Uppm))
    merged = innerjoin(mn, select(cty, :fips, :log_u); on=:fips)
    unique!(merged, :idnum)

    (; log_radon=Float64.(merged.log_radon),
       floor=dense_int(Int.(merged.floor)),
       county=dense_int(strip.(String.(merged.county))))
end

const RADON_ADAPTER_NOTE =
    "Pinned historical two-file loader: Minnesota rows from srrs2.dat; " *
    "log_radon=log(activity+0.1); FIPS join to cty.dat; unique idnum; " *
    "floor and stripped county densely recoded. Auxiliary cty.dat sha256=" *
    RADON_CTY_SHA256
const RADON_AUXILIARY = Dict("url" => RADON_CTY_URL, "sha256" => RADON_CTY_SHA256)

# ---------------------------------------------------------------------------
# Historical-gallery tranche adapters.
#
# Each of these reproduces the filtering its cited source performs before
# fitting. They are deliberately written as plain, total functions over the raw
# file: no sampling, no random subsetting, no `first(n)` convenience that would
# depend on file order unless the source itself does (baseball, which does).
# ---------------------------------------------------------------------------

"""
Baseball batting rows exactly as the bambi hierarchical-binomial notebook
selects them.

The notebook nulls out `AB == 0` (a plate-appearance-free row carries no
binomial information), drops the nulled rows, keeps `yearID >= 2016`, and then
takes the first fifteen remaining rows. The 1:15 slice is the source's own
choice, not a runtime budget: the notebook is about partial pooling across a
handful of players. File order is Lahman's, which is stable for a pinned file,
and the pinned `sha256` is what makes that reproducible.
"""
function baseball_adapter(df)
    keep = [i for i in 1:size(df, 1)
            if !ismissing(df.AB[i]) && df.AB[i] != 0 &&
               !ismissing(df.H[i]) && !ismissing(df.yearID[i]) &&
               df.yearID[i] >= 2016]
    length(keep) >= 15 || error("baseball filter left only $(length(keep)) rows")
    selected = df[keep[1:15], :]
    (; H=Int.(selected.H), AB=Int.(selected.AB),
       playerID=dense_int(String.(selected.playerID)))
end

"""
The twelve number-nonagreement studies of the interference meta-analysis.

`Effect` and `SE` are already on a common millisecond scale in the source file,
so nothing is rescaled. Every study contributes exactly one observation: the
random intercept is identified only because the response SE is known and fixed,
which is precisely the structure this row is here to cover.
"""
function meta_sbi_adapter(df)
    selected = df[(String.(df.TargetType) .== "Match") .&
                  (String.(df.DepType) .== "nonagreement"), :]
    size(selected, 1) == 12 || error(
        "meta_sbi filter selected $(size(selected, 1)) rows, expected 12")
    (; effect=Float64.(selected.Effect), SE=Float64.(selected.SE),
       study_id=dense_int(String.(selected.Publication)))
end

"""
Nieuwland et al. N400 amplitudes, all nine laboratories.

`c_cloze` is the cited book's predictor: cloze probability as a proportion,
grand-mean centred. The receipt file stores cloze as a percentage, so it is
divided by 100 first — that reproduces the book's predictor SCALE, which is what
its reported coefficient is expressed in, and keeps the predictor O(1).

The book's own worked example restricts to the Edinburgh laboratory (2,827 rows
here, 37 subjects) to speed up computation. This benchmark deliberately keeps
all 25,848 rows and all 334 subjects: the whole point of carrying this row is the
crossed subject-by-item structure at a scale nothing else in the set reaches.
"""
function n400_adapter(df)
    cloze = Float64.(df.cloze) ./ 100
    (; n400=Float64.(df.n400), c_cloze=cloze .- mean(cloze),
       subj=dense_int(String.(df.subject)), item=dense_int(Int.(df.item)))
end

const SPECS = InventorySpec[
    # ---- The eight already-published rows, unchanged. ------------------------
    InventorySpec(
        source="lme4", key="dyestuff_re", dataset="dyestuff",
        url="https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff.csv",
        sha256="7a0c76e36c68aad3bddeff58811b89a07fb7a88cbf9ccf5d2b6fa6c30f6ca578",
        coverage="gaussian intercept-only hierarchy, smallest target",
        adapter_note="Yield copied as Float64; Batch deterministically recoded to dense integers",
        adapt=df -> (; Yield=Float64.(df.Yield), Batch=dense_int(df.Batch)),
    ),
    InventorySpec(
        source="lme4", key="sleepstudy_slope", dataset="sleepstudy",
        url="https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv",
        sha256="20922ddc87d538bf2344f73d49000eeb2d7e14934b6191a3e071fefc329942c4",
        coverage="gaussian varying-slope hierarchy; the historical formula's " *
            "implicit random intercept is not generated, so the block is width one",
        adapter_note="Reaction and Days copied as Float64 without response scaling; Subject " *
            "deterministically recoded to dense integers",
        adapt=df -> (; Reaction=Float64.(df.Reaction), Days=Float64.(df.Days),
                     Subject=dense_int(df.Subject)),
    ),
    InventorySpec(
        source="bambi", key="sleepstudy", dataset="sleepstudy",
        url="https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv",
        sha256="20922ddc87d538bf2344f73d49000eeb2d7e14934b6191a3e071fefc329942c4",
        coverage="duplicate-card control: a second source's transcription of the same model",
        adapter_note="Reaction and Days copied as Float64 without response scaling; Subject " *
            "deterministically recoded to dense integers",
        adapt=df -> (; Reaction=Float64.(df.Reaction), Days=Float64.(df.Days),
                     Subject=dense_int(df.Subject)),
    ),
    InventorySpec(
        source="mixed_models_jl", key="penicillin_crossed", dataset="penicillin",
        url="https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv",
        sha256="0caff3b1bf332fcb8cc0e410ce3f5e49c390b3bc10beda1bc4587fee1c3a466d",
        coverage="two crossed intercept-only grouping factors",
        adapter_note="diameter copied as Float64; plate and sample converted to String and " *
            "deterministically recoded to dense integers",
        adapt=df -> (; diameter=Float64.(df.diameter),
                     plate=dense_int(String.(df.plate)),
                     sample=dense_int(String.(df.sample))),
    ),
    InventorySpec(
        source="bambi", key="radon_partial", dataset="radon",
        url=RADON_SRRS2_URL, sha256=RADON_SRRS2_SHA256,
        coverage="gaussian partial pooling with a group-level covariate",
        adapter_note=RADON_ADAPTER_NOTE, adapt=radon_adapter,
        auxiliary=RADON_AUXILIARY,
    ),
    InventorySpec(
        source="bambi", key="radon_floor", dataset="radon",
        url=RADON_SRRS2_URL, sha256=RADON_SRRS2_SHA256,
        coverage="gaussian varying intercept with a binary fixed effect",
        adapter_note=RADON_ADAPTER_NOTE, adapt=radon_adapter,
        auxiliary=RADON_AUXILIARY,
    ),
    InventorySpec(
        source="bambi", key="radon_slopes", dataset="radon",
        url=RADON_SRRS2_URL, sha256=RADON_SRRS2_SHA256,
        coverage="gaussian varying slopes over many small groups",
        adapter_note=RADON_ADAPTER_NOTE, adapt=radon_adapter,
        auxiliary=RADON_AUXILIARY,
    ),
    InventorySpec(
        source="bambi", key="dietox", dataset="dietox",
        url="https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv",
        sha256="4d32f92a38aa031b20319dfd959f2503ca261ad6938060cdb809e7a679b2ba88",
        coverage="gaussian longitudinal growth curve, moderate dimension",
        adapter_note="Weight and Time copied as Float64; Pig deterministically recoded to dense integers",
        adapt=df -> (; Weight=Float64.(df.Weight), Time=Float64.(df.Time),
                     Pig=dense_int(df.Pig)),
    ),

    # ---- Historical-gallery tranche, ordered cheapest-first. -----------------
    InventorySpec(
        source="vasishth", key="meta_sbi", dataset="meta_sbi",
        url="https://osf.io/du3qp/?action=download",
        sha256="f4d3c311dd4dc3c418f2c76cceeacb3289e0a4637b6951333fa91737aa49eacf",
        coverage="known-response-SE random-effects meta-analysis: the residual scale is " *
            "DATA, not a sampled parameter, and each group holds one observation",
        read_options=(; delim=';'),
        adapter_note="Semicolon-separated OSF receipt. Rows with TargetType==\"Match\" and " *
            "DepType==\"nonagreement\" retained (12 of 77); Effect and SE copied as Float64 on " *
            "the source's millisecond scale; Publication densely recoded as study_id",
        adapt=meta_sbi_adapter,
    ),
    InventorySpec(
        source="kruschke", key="fruitfly_anhecova", dataset="fruitfly",
        url="https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-" *
            "and-the-tidyverse/master/data.R/FruitflyDataReduced.csv",
        sha256="8d9575b76856558b1f244348d63080da3a0bd1d57d3fd3ff05c12f51ef51699e",
        coverage="correlated intercept-and-slope block over only FIVE groups — the " *
            "regime where the group-level covariance is worst identified",
        adapter_note="Longevity copied as Float64 without response scaling; " *
            "thorax_c = Thorax - mean(Thorax); CompanionNumber densely recoded",
        adapt=df -> (; Longevity=Float64.(df.Longevity),
                     thorax_c=Float64.(df.Thorax) .- mean(Float64.(df.Thorax)),
                     CompanionNumber=dense_int(String.(df.CompanionNumber))),
    ),
    InventorySpec(
        source="burkner_papers", key="epilepsy_simple", dataset="epilepsy",
        url="https://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv",
        sha256="6d4aded9dda1c8cd051d370ce77816264c972e773417e62cfad3d061d3083a99",
        coverage="Poisson log-link GLMM — first non-gaussian likelihood in the set",
        adapter_note="count = y (integer seizure counts, 0-102); Trt = a 0/1 treatment " *
            "contrast over the sorted trt levels (placebo=0, progabide=1); patient = " *
            "densely recoded subject. 236 rows / 59 patients",
        adapt=df -> (; count=Int.(df.y), Trt=treatment_contrast(String.(df.trt)),
                     patient=dense_int(Int.(df.subject))),
    ),
    InventorySpec(
        source="kruschke", key="therapeutic_touch", dataset="therapeutic_touch",
        url="https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-" *
            "and-the-tidyverse/master/data.R/TherapeuticTouchData.csv",
        sha256="8c196a3e124a649fe2df5e5446db1331b5af0c6041dfed4db8557c1f2267c64a",
        coverage="Bernoulli-logit intercept-only GLMM: the binary-response analogue of " *
            "dyestuff, and the control that separates the link from the structure",
        adapter_note="y parsed as 0/1 integers; s densely recoded. 280 trials / 28 subjects",
        adapt=df -> (; y=Int.(df.y), s=dense_int(String.(df.s))),
    ),
    InventorySpec(
        source="bambi", key="hierarchical_binomial_partial", dataset="baseball",
        url="https://ndownloader.figshare.com/files/29749140",
        sha256="bbbc9459632c738a07bbe0877970a7bbd1f4c2448193979337fe5bc3a4ab0228",
        coverage="aggregated BinomialLogit with a per-row trials count, and a hierarchy " *
            "over 15 groups of 1 observation each",
        adapter_note="Rows with AB==0 or missing AB/H/yearID dropped, then yearID>=2016, " *
            "then the first 15 remaining rows in file order — the notebook's own " *
            "selection. H and AB copied as Int; playerID densely recoded",
        adapt=baseball_adapter,
    ),
    InventorySpec(
        source="mixed_models_jl", key="contraception_glmm", dataset="contraception",
        url="https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Contraception.csv",
        sha256="dd76de5f4f1fb57081b01ef0f81581cd928ad545d13feb8bf7d337d71e690034",
        coverage="Bernoulli-logit GLMM at scale: 1,934 observations over 60 districts, " *
            "with a quadratic fixed effect evaluated inside the generated formula",
        adapter_note="use Y/N mapped to 1/0; age copied as Float64 (already centred " *
            "upstream) with abs2(age) supplied by the generated formula itself; urban " *
            "Y/N mapped to a 0/1 contrast; livch and district densely recoded. " *
            "1,934 rows / 60 districts",
        categorical_departures="livch is a four-level factor (0, 1, 2, 3+) in the source. " *
            "BRM's verbatim surface carries it as ONE term, so it enters as the monotone " *
            "integer code 1-4 rather than three contrast columns. The source's grouping " *
            "column dist is the RDatasets mirror's district (BRM records this as " *
            "adapted-but-defensible).",
        adapt=df -> (; use=Int.(String.(df.use) .== "Y"),
                     age=Float64.(df.age),
                     livch=Float64.(dense_int(String.(df.livch))),
                     urban=treatment_contrast(String.(df.urban)),
                     district=dense_int(Int.(df.district))),
    ),
    InventorySpec(
        source="bambi", key="predict_new_groups", dataset="pulmonary",
        url="https://gist.githubusercontent.com/ucals/2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/" *
            "43034c39052dcf97d4b894d2ec1bc3f90f3623d9/osic_pulmonary_fibrosis.csv",
        sha256="45aa8d255d4c26476d6ba1bd4ba38823752420a716b75cf3ec24f2fab93f7270",
        coverage="SLOPE-ONLY random-effect block with no random intercept and no " *
            "population intercept — the one structural shape the rest of the set " *
            "never exercises, over 176 groups",
        adapter_note="Source columns lowercased to the body's names: FVC->fvc, " *
            "Weeks->weeks, SmokingStatus->smoking_status, Patient->patient. fvc and " *
            "weeks copied as Float64 without response scaling; smoking_status and " *
            "patient densely recoded. 1,549 rows / 176 patients",
        categorical_departures="smoking_status is a three-level factor (Currently smokes, " *
            "Ex-smoker, Never smoked). The historical `0 + ... + smoking_status` term relies " *
            "on the factor expanding to three dummy columns; BRM's verbatim surface carries " *
            "one term, so it enters as the monotone integer code 1-3. The random-effect " *
            "structure under test — (0 + weeks | patient) — is unaffected.",
        adapt=df -> (; fvc=Float64.(df.FVC), weeks=Float64.(df.Weeks),
                     smoking_status=Float64.(dense_int(String.(df.SmokingStatus))),
                     patient=dense_int(String.(df.Patient))),
    ),
    InventorySpec(
        source="vasishth", key="n400_crossed", dataset="n400",
        url="https://osf.io/q7dsk/?action=download",
        sha256="c5a023fa9cf6a8d30ab6877a7e220aa8dbf7a03d89339c80440c711fa348fb38",
        coverage="the stress case: TWO crossed varying-slope blocks over 334 subjects " *
            "and 80 items, 25,848 observations, 181k shared constrained coordinates " *
            "— an order of magnitude past anything else in the set on data size and " *
            "on the cost of the ESS reduction itself",
        read_options=(; delim='\t'),
        adapter_note="Tab-separated OSF receipt, all nine laboratories retained. " *
            "n400 copied as Float64; c_cloze = cloze/100 - mean(cloze/100) (the cited " *
            "book's centred proportion-scale predictor); subject and item densely " *
            "recoded to subj and item. 25,848 rows / 334 subjects / 80 items",
        adapt=n400_adapter,
    ),
]

selected_specs() = haskey(ENV, "BRMI_SPECS") ?
    [spec for spec in SPECS
     if "$(spec.source):$(spec.key)" in split(ENV["BRMI_SPECS"], ',')] : SPECS

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

function inventory_row(spec)
    row = only(r for r in TRANSLATION_ROWS
               if r["source"] == spec.source && r["key"] == spec.key &&
                  r["variant"] == "inferred-family")
    row["translation_status"] == "ready" ||
        error("$(spec.source):$(spec.key) is not a ready translation")
    row["surface_support_class"] == "already-expressible-verbatim" ||
        error("$(spec.source):$(spec.key) is not a verbatim translation")

    matrix = only(r for r in MATRIX_ROWS
                  if r["row_key"] == "$(spec.source):$(spec.key)")
    matrix["inferred_translation_status"] == "ready" ||
        error("authoritative matrix does not mark $(spec.source):$(spec.key) ready")
    matrix["inferred_surface_support_class"] == "already-expressible-verbatim" ||
        error("authoritative matrix does not mark $(spec.source):$(spec.key) verbatim")
    matrix["inferred_capability_tier"] == "bridgestan-finite-density-gradient" ||
        error("authoritative matrix lacks a finite-gradient receipt for $(spec.source):$(spec.key)")
    matrix["inferred_current_brm_body"] == row["current_brm_body"] ||
        error("matrix/translation body disagreement for $(spec.source):$(spec.key)")
    row, matrix
end

function dataset(spec)
    expected = isempty(spec.sha256) ? nothing : spec.sha256
    cached_download(spec.dataset * ".csv", spec.url; expected_sha256=expected)
end

read_dataset(spec) = CSV.read(dataset(spec), DataFrame; spec.read_options...)

"""
The grouping factors that actually appear in the generated body.

`group_columns` in `translations.tsv` is a lexical scrape of the historical
formula's grouping and addition terms, so a response-side addition term can land
in it. `vasishth:meta_sbi` is the live case: its historical formula is
`effect | resp_se(SE, sigma = FALSE) ~ 1 + (1 | study_id)`, and the scrape
returns `resp_se,study_id`. Handing `resp_se` to `centered_groups` would ask BRM
to centre a grouping factor the generated body does not have.

This returns the tokens on the right of a `|` or `||` inside a random-effect
term of `current_brm_body`. The caller keeps the inventory's ORDER and records
both sets in the artifact, so the reconciliation is visible rather than implied.
"""
function body_group_columns(body)
    pattern = r"\(([^()]*(?:\([^()]*\)[^()]*)*)\|\|?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
    unique(m.captures[2] for m in eachmatch(pattern, body))
end

const RE_BLOCK_PATTERN =
    r"\(([^()|]*(?:\([^()]*\)[^()|]*)*)\|(\|?)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"

block_terms(inner) = filter(!isempty, strip.(split(inner, '+')))

"""
Width K of each random-effect block in the GENERATED body, in source order.

K is the number of coefficients sharing one group-level covariance — the
structural quantity a reparametrization benchmark is actually varying — and it is
counted LITERALLY from the terms BRM's verbatim surface was given. `(1 | g)` is
K=1 and `(1 + x | g)` is K=2.

That literal count is BRM's real block width, and it is NOT lme4's reading of the
same text. `lme4` and `brms` treat `(x | g)` as an implicit intercept plus a slope
(K=2); BRM's verbatim surface takes the terms as written, so the generated body
fits one random coefficient per group (K=1). The published dimensions confirm it:
`loc ~ Days + (Days | Subject)` over 18 subjects lowers to 21 unconstrained
coordinates — one fixed slope, one `log(sigma)`, 18 subject coefficients, one
group scale — with no intercept vector and no correlation. `historical_block_widths`
below computes the lme4 reading of the same formula so the difference is recorded
per row instead of being silently absorbed.

`run_brm_high_k_preflight.jl` parses the same shape out of the HISTORICAL formula
of cards that never reach this runner. Deliberately a separate implementation:
its input is the catalogue's `formula_claim` (brms/lme4 surface syntax, `I(...)`,
`zerocorr(...)`), not a generated BRM body.
"""
function random_effect_blocks(body)
    [Dict("k" => count(t -> t != "0", block_terms(m.captures[1])),
          "terms" => strip(m.captures[1]),
          "group" => m.captures[3],
          "correlated" => isempty(m.captures[2]))
     for m in eachmatch(RE_BLOCK_PATTERN, body)]
end

"""
Width of each random-effect block under the HISTORICAL `lme4`/`brms` convention.

An intercept is implied unless the block suppresses it with `0` or `-1`, so
`(x | g)` is width two there while `random_effect_blocks` reports the generated
body's one. Returned in the same source order so the two lists zip.
"""
function historical_block_widths(formula)
    out = Any[]
    for m in eachmatch(RE_BLOCK_PATTERN, formula)
        terms = block_terms(m.captures[1])
        suppressed = any(t -> t in ("0", "-1"), terms)
        slopes = count(t -> !(t in ("0", "1", "-1")), terms)
        push!(out, Dict("k" => slopes + (suppressed ? 0 : 1),
                        "terms" => strip(m.captures[1]),
                        "group" => m.captures[3],
                        "correlated" => isempty(m.captures[2])))
    end
    out
end

function materialize(row, data; centered_groups=Symbol[])
    body = row["current_brm_body"]
    brmi = Core.eval(@__MODULE__, BRM._brm(body; df=data))
    sb = SBBRMI(brmi; mod=@__MODULE__, centered_groups)
    name = Symbol(replace(row["source"] * "_" * row["key"], r"\W" => "_"))
    descriptor = brm_descriptor(sb; name)
    (; brmi, sb, descriptor)
end

gitsha(dir) = try readchomp(`git -C $dir rev-parse HEAD`) catch; "unavailable" end
gitdirty(dir, paths...) = try
    !isempty(readchomp(`git -C $dir status --porcelain -- $(collect(paths))`))
catch
    true
end

function ess_vec(draws::AbstractMatrix)
    d, n = size(draws)
    (n < 10 || !all(isfinite, draws)) && return Float64[]
    collect(MCMCDiagnosticTools.ess(reshape(permutedims(draws), (n, 1, d))))
end

function finite_min(v)
    w = filter(isfinite, v)
    isempty(w) ? (NaN, length(v)) : (minimum(w), length(v) - length(w))
end

function shared_constrained_ess(model, names, shared, draws)
    keep = [i for (i, name) in enumerate(names) if name in shared]
    out = Matrix{Float64}(undef, length(keep), size(draws, 2))
    ok = falses(size(draws, 2))
    for j in axes(draws, 2)
        try
            values = BS.param_constrain(model, Vector{Float64}(draws[:, j]); include_tp=true)
            out[:, j] = values[keep]
            ok[j] = all(isfinite, @view out[:, j])
        catch
            ok[j] = false
        end
    end
    minimum_ess, n_constant = finite_min(ess_vec(out[:, ok]))
    minimum_ess, count(ok), n_constant
end

function sampling_diagnostic_error(draws, n_constrained, ess_unc, ess_con,
                                   n_constant, n_divergent)
    problems = String[]
    n_draws = size(draws, 2)
    n_constrained == n_draws || push!(
        problems,
        "only $n_constrained/$n_draws draws could be constrained",
    )
    isfinite(ess_unc) && ess_unc > 0 || push!(
        problems,
        "unconstrained draws have no positive finite minimum ESS",
    )
    isfinite(ess_con) && ess_con > 0 || push!(
        problems,
        "shared constrained draws have no positive finite minimum ESS " *
        "($n_constant constant/non-finite coordinates)",
    )
    isempty(problems) && return ""
    "statistical sampling failure: " * join(problems, "; ") *
        "; divergences=$n_divergent/$n_draws"
end

function run_arm(; spec_key, arm, problem, model, names, shared, adapt, seed)
    try
        wall = @elapsed result = adaptive_warmup_mcmc(
            Xoshiro(seed), problem; n_draws=N_DRAWS,
            nonlinear_adapt=adapt, progress=nothing,
        )
        draws = Matrix{Float64}(result.posterior_position)
        ess_unc, _ = finite_min(ess_vec(draws))
        ess_con, n_ok, n_constant = shared_constrained_ess(
            model, names, shared, draws)
        grad = Int(result.total_evaluation_counter)
        ndiv = Int(result.n_divergent_samples)
        diagnostic = sampling_diagnostic_error(
            draws, n_ok, ess_unc, ess_con, n_constant, ndiv)
        (; spec=spec_key, arm, nonlinear_adapt=adapt, seed,
         ok=isempty(diagnostic),
         wall_s=wall, grad_evals=grad, ess_min_unconstrained=ess_unc,
         ess_min_shared_constrained=ess_con,
         ess_min_per_grad=ess_con / max(grad, 1),
         n_draws_constrained=n_ok, n_constant,
         n_divergent=ndiv, error=diagnostic)
    catch err
        (; spec=spec_key, arm, nonlinear_adapt=adapt, seed, ok=false,
         wall_s=NaN, grad_evals=0, ess_min_unconstrained=NaN,
         ess_min_shared_constrained=NaN, ess_min_per_grad=NaN,
         n_draws_constrained=0, n_constant=0, n_divergent=0,
         error=first(sprint(showerror, err, catch_backtrace()), 1200))
    end
end

function sample_standard(sampler, rng, problem, n_draws; nonlinear_adapt=true)
    if sampler == "warmuphmc"
        result = adaptive_warmup_mcmc(
            rng, problem; n_draws, nonlinear_adapt, progress=nothing,
        )
        Matrix{Float64}(result.posterior_position),
            Int(result.n_divergent_samples)
    elseif sampler == "dynamichmc"
        result = WarmupHMC.DynamicHMC.mcmc_with_warmup(
            rng, problem, n_draws;
            reporter=WarmupHMC.DynamicHMC.NoProgressReport(),
        )
        ndiv = count(
            s -> WarmupHMC.DynamicHMC.is_divergent(s.termination),
            result.tree_statistics,
        )
        Matrix{Float64}(result.posterior_matrix), ndiv
    else
        error("unknown standard-comparison sampler $(repr(sampler))")
    end
end

"""
Run one standard-warmup comparison arm.

`base_problem` is always the generated BRM density. `build_problem` receives
the externally counted version of that density, so the adaptive-centering arm
keeps its `ReparametrizedProblem` outer type while every sampler is counted at
the same inner logdensity-and-gradient boundary.
"""
function run_standard_arm(; spec_key, arm, sampler, parameterization,
                            base_problem, build_problem, model, names, shared,
                            nonlinear_adapt, seed, n_draws=N_DRAWS)
    try
        timed = WarmupHMC.count_and_time(base_problem) do counted
            problem = build_problem(counted)
            sample_standard(sampler, Xoshiro(seed), problem, n_draws;
                            nonlinear_adapt)
        end
        draws, ndiv = timed.result
        ess_unc, _ = finite_min(ess_vec(draws))
        ess_con, n_ok, n_constant = shared_constrained_ess(
            model, names, shared, draws)
        grad = Int(timed.n_evaluations)
        diagnostic = sampling_diagnostic_error(
            draws, n_ok, ess_unc, ess_con, n_constant, ndiv)
        (; spec=spec_key, arm, sampler, parameterization, nonlinear_adapt,
         seed, ok=isempty(diagnostic),
         wall_s=timed.elapsed, grad_evals=grad,
         n_draws_actual=size(draws, 2),
         ess_min_unconstrained=ess_unc,
         ess_min_shared_constrained=ess_con,
         ess_min_per_grad=ess_con / max(grad, 1),
         n_draws_constrained=n_ok, n_constant,
         n_divergent=Int(ndiv), error=diagnostic)
    catch err
        (; spec=spec_key, arm, sampler, parameterization, nonlinear_adapt,
         seed, ok=false,
         wall_s=NaN, grad_evals=0, n_draws_actual=0,
         ess_min_unconstrained=NaN,
         ess_min_shared_constrained=NaN, ess_min_per_grad=NaN,
         n_draws_constrained=0, n_constant=0, n_divergent=0,
         error=first(sprint(showerror, err, catch_backtrace()), 1200))
    end
end

sanitize(x) = x isa AbstractFloat && !isfinite(x) ? nothing : x
sanitize(x::AbstractDict) = Dict(k => sanitize(v) for (k, v) in x)
sanitize(x::AbstractVector) = [sanitize(v) for v in x]

const ROWS = Any[]
const MODEL_META = Any[]

function flush_out(config)
    mkpath(dirname(OUT))
    temporary = OUT * ".tmp"
    open(temporary, "w") do io
        JSON.print(io, Dict(
            "config" => sanitize(config),
            "models" => sanitize(MODEL_META),
            "rows" => [sanitize(Dict(string(k) => v for (k, v) in pairs(row)))
                       for row in ROWS],
        ), 2)
    end
    mv(temporary, OUT; force=true)
end

function main()
    overall_started = time()
    run_started_at = now()
    brm_dir = pkgdir_of(BRM)
    whmc_dir = pkgdir_of(WarmupHMC)
    reproduction = MODE == "standard" ?
        "BRMI_MODE=standard BRMI_SEEDS=$(N_SEEDS) BRMI_DRAWS=$(N_DRAWS) " *
        "BRMI_PREFLIGHT_DRAWS=$(PREFLIGHT_DRAWS) julia --startup-file=no " *
        "--project=/path/to/pinned/environment " *
        "docs/benchmark/run_brm_inventory_benchmark.jl" :
        "BRMI_SEEDS=$(N_SEEDS) BRMI_DRAWS=$(N_DRAWS) " *
        "BRMI_PREFLIGHT_DRAWS=$(PREFLIGHT_DRAWS) julia --startup-file=no " *
        "--project=/path/to/pinned/environment " *
        "docs/benchmark/run_brm_inventory_benchmark.jl"
    run_order = MODE == "standard" ?
        "one untimed preflight per comparison arm; the six recorded arms " *
        "rotate cyclically by seed to avoid systematic temporal bias" :
        "one untimed preflight per arm/flag; recorded flag order alternates " *
        "by seed to avoid systematic temporal bias"
    config = Dict(
        "host" => get(ENV, "KB_HOST", gethostname()),
        "julia" => string(VERSION),
        "blas_threads" => BLAS.get_num_threads(),
        "mode" => MODE,
        "sampler" => MODE == "standard" ?
            "adaptive_warmup_mcmc vs DynamicHMC.mcmc_with_warmup" :
            "adaptive_warmup_mcmc",
        "runner" => "docs/benchmark/run_brm_inventory_benchmark.jl",
        # The consumer-side adapters live in the runner, so `warmuphmc_sha` alone
        # does not pin them: a sweep is normally launched from a worktree whose
        # HEAD is the sampler revision it measures, with the runner edit still
        # uncommitted. This checksums the runner's own bytes, which is what makes
        # "the same adapters produced these rows" checkable independently of which
        # commit they eventually landed in.
        "runner_sha256" => bytes2hex(open(sha256, @__FILE__)),
        "reproduction" => reproduction,
        "warmuphmc_sha" => gitsha(whmc_dir),
        "dynamichmc_version" => string(Base.pkgversion(WarmupHMC.DynamicHMC)),
        "brm_sha" => gitsha(brm_dir),
        "stanblocks_sha" => gitsha(pkgdir_of(StanBlocks)),
        "warmuphmc_src_dirty" => gitdirty(whmc_dir, "src"),
        "brm_inventory_dirty" => gitdirty(brm_dir, "research/historical_model_inventory"),
        "translations_sha256" => bytes2hex(open(sha256, TRANSLATIONS)),
        "model_matrix_sha256" => bytes2hex(open(sha256, MODEL_MATRIX)),
        "inventory_contract" => "model bodies are read from translations.tsv and " *
            "cross-checked against model_matrix.tsv; real-data adapters are consumer-side",
        "ad_backend" => "AutoEnzyme(Reverse, runtime_activity, Const), used by " *
                        "BRM.adaptive_centering_problem",
        "controls" => "the six-arm standard matrix separates WarmupHMC's " *
            "linear adaptation on generated non-centered/centered targets from " *
            "the fixed and fitted nonlinear-wrapper paths",
        "n_seeds" => N_SEEDS,
        "n_draws" => N_DRAWS,
        "timing_preflight_draws" => PREFLIGHT_DRAWS,
        "run_order" => run_order,
        "gradient_counter" => MODE == "standard" ?
            "WarmupHMC.count_and_time around the generated BRM density for " *
            "every sampler; the adaptive wrapper is built over the counted inner target" :
            "adaptive_warmup_mcmc total_evaluation_counter",
        "warmup_budget" => MODE == "standard" ?
            "sampler defaults: DynamicHMC uses its default 1000-step Stan-style " *
            "warmup; WarmupHMC chooses its own gradient-targeted windows" : nothing,
        "seeds" => collect(1:N_SEEDS),
        "generated_at" => string(run_started_at),
        "run_started_at" => string(run_started_at),
    )

    for spec in selected_specs()
        started = time()
        row, matrix = inventory_row(spec)
        path = dataset(spec)
        df = read_dataset(spec)
        data = spec.adapt(df)
        inventory_groups = filter(!isempty, split(row["group_columns"], ','))
        present = body_group_columns(row["current_brm_body"])
        issubset(present, inventory_groups) || error(
            "$(spec.source):$(spec.key) body groups $(present) are not all listed in " *
            "the inventory's group_columns $(inventory_groups)")
        groups = Symbol.(filter(in(present), inventory_groups))

        built_nc = materialize(row, data)
        built_c = materialize(row, data; centered_groups=groups)
        problem_nc = StanBlocks.stan_instantiate(built_nc.sb.model)
        problem_c = StanBlocks.stan_instantiate(built_c.sb.model)
        names_nc = BS.param_names(problem_nc.model; include_tp=true)
        names_c = BS.param_names(problem_c.model; include_tp=true)
        unc_names_nc = BS.param_unc_names(problem_nc.model)
        shared = intersect(Set(names_nc), Set(names_c))
        descriptor_code = brm_execute(built_nc.descriptor, :transpile)

        spec_key = "$(spec.source):$(spec.key)"
        push!(MODEL_META, Dict(
            "spec" => spec_key,
            "source" => spec.source,
            "key" => spec.key,
            "variant" => row["variant"],
            "translation_status" => row["translation_status"],
            "surface_support_class" => row["surface_support_class"],
            "semantic_route" => row["semantic_route"],
            "capability_tier" => matrix["inferred_capability_tier"],
            "probe_evidence_kind" => matrix["probe_evidence_kind"],
            "probe_id" => matrix["probe_id"],
            "source_fidelity_verdict" => get(matrix, "source_fidelity_verdict", ""),
            "source_fidelity_reason" => get(matrix, "source_fidelity_reason", ""),
            "source_fidelity_manual_reviewed" =>
                get(matrix, "source_fidelity_manual_reviewed", ""),
            "inferred_family" => get(matrix, "inferred_family", ""),
            "inferred_family_provenance" =>
                get(matrix, "inferred_family_provenance", ""),
            "family_support" => get(matrix, "family_support", ""),
            "dataset_support" => get(matrix, "dataset_support", ""),
            "dataset_receipt_urls" => get(matrix, "dataset_receipt_urls", ""),
            "row_source_claim" => get(matrix, "row_source_claim", ""),
            "historical_formula" => row["formula_claim"],
            "current_brm_body" => row["current_brm_body"],
            "current_brm_body_sha256" => bytes2hex(sha256(row["current_brm_body"])),
            "grouping_factors" => string.(groups),
            "inventory_group_columns" => string.(inventory_groups),
            "random_effect_blocks" => random_effect_blocks(row["current_brm_body"]),
            "historical_random_effect_blocks" =>
                historical_block_widths(row["formula_claim"]),
            "dataset" => spec.dataset,
            "data_url" => spec.url,
            "data_sha256" => bytes2hex(open(sha256, path)),
            "data_sha256_pinned" => spec.sha256,
            "auxiliary_data" => spec.auxiliary,
            "data_adapter" => spec.adapter_note,
            "categorical_departures" => spec.categorical_departures,
            "coverage" => spec.coverage,
            "n_obs" => length(first(data)),
            "n_groups" => Dict(string(g) => length(unique(getproperty(data, g)))
                               for g in groups),
            "descriptor_operations" => string.(getproperty.(built_nc.descriptor.operations, :name)),
            "descriptor_stan_sha256" => bytes2hex(sha256(descriptor_code)),
            "dim_noncentered" => LogDensityProblems.dimension(problem_nc),
            "dim_centered" => LogDensityProblems.dimension(problem_c),
            "n_names_noncentered" => length(names_nc),
            "n_names_centered" => length(names_c),
            "n_names_shared" => length(shared),
        ))

        problem_for(arm) = if arm == "noncentered"
            (problem_nc, problem_nc.model, names_nc)
        elseif arm == "centered"
            (problem_c, problem_c.model, names_c)
        else
            # Rebuild the wrapper for every run so one seed's adaptive
            # centering cannot become the next seed's starting point.
            (BRM.adaptive_centering_problem(
                built_nc.sb, problem_nc, AD_BACKEND), problem_nc.model, names_nc)
        end

        if MODE == "inventory"
            # Keep Julia/Enzyme compilation out of the sampling-time comparison.
            # Both Boolean paths are exercised because the runtime flag controls a
            # different warm-up branch even though its type is the same.
            for arm in ("noncentered", "centered", "adaptive_centering"),
                adapt in (false, true)
                problem, _, _ = problem_for(arm)
                adaptive_warmup_mcmc(
                    Xoshiro(0x6b625000 + 10 * findfirst(==(arm),
                        ("noncentered", "centered", "adaptive_centering")) + adapt),
                    problem; n_draws=PREFLIGHT_DRAWS,
                    nonlinear_adapt=adapt, progress=nothing,
                )
            end

            for arm in ("noncentered", "centered", "adaptive_centering"),
                seed in 1:N_SEEDS
                adapt_order = isodd(seed) ? (false, true) : (true, false)
                for adapt in adapt_order
                    problem, model, names = problem_for(arm)
                    push!(ROWS, run_arm(; spec_key, arm, problem, model, names, shared,
                                        adapt, seed))
                end
            end
        else
            identity_problem = counted -> counted
            adaptive_problem = counted -> BRM.adaptive_centering_problem(
                built_nc.sb, counted, AD_BACKEND; unc_names=unc_names_nc)
            standard_arms = [
                (; arm="warmuphmc_noncentered", sampler="warmuphmc",
                 parameterization="noncentered", base_problem=problem_nc,
                 build_problem=identity_problem, model=problem_nc.model,
                 names=names_nc, nonlinear_adapt=false),
                (; arm="warmuphmc_centered", sampler="warmuphmc",
                 parameterization="centered", base_problem=problem_c,
                 build_problem=identity_problem, model=problem_c.model,
                 names=names_c, nonlinear_adapt=false),
                (; arm="warmuphmc_fixed_centering", sampler="warmuphmc",
                 parameterization="adaptive_wrapper_fixed_at_generated_endpoint",
                 base_problem=problem_nc, build_problem=adaptive_problem,
                 model=problem_nc.model, names=names_nc,
                 nonlinear_adapt=false),
                (; arm="warmuphmc_adaptive_centering", sampler="warmuphmc",
                 parameterization="adaptive_centering", base_problem=problem_nc,
                 build_problem=adaptive_problem, model=problem_nc.model,
                 names=names_nc, nonlinear_adapt=true),
                (; arm="dynamichmc_noncentered", sampler="dynamichmc",
                 parameterization="noncentered", base_problem=problem_nc,
                 build_problem=identity_problem, model=problem_nc.model,
                 names=names_nc, nonlinear_adapt=false),
                (; arm="dynamichmc_centered", sampler="dynamichmc",
                 parameterization="centered", base_problem=problem_c,
                 build_problem=identity_problem, model=problem_c.model,
                 names=names_c, nonlinear_adapt=false),
            ]

            for (i, a) in enumerate(standard_arms)
                preflight = run_standard_arm(;
                    spec_key, a..., shared,
                    seed=0x6b645000 + i, n_draws=PREFLIGHT_DRAWS,
                )
                preflight.ok || error(
                    "standard comparison preflight failed for $(a.arm): $(preflight.error)")
            end

            for seed in 1:N_SEEDS
                offset = mod(seed - 1, length(standard_arms))
                order = vcat(standard_arms[offset+1:end], standard_arms[1:offset])
                for a in order
                    push!(ROWS, run_standard_arm(; spec_key, a..., shared, seed))
                end
            end
        end

        completed = [r for r in ROWS if r.spec == spec_key]
        elapsed_seconds = round(time() - started; digits=1)
        config["elapsed_so_far_s"] = round(time() - overall_started; digits=3)
        flush_out(config)
        @info "inventory spec done" spec=spec_key seconds=elapsed_seconds rows=length(completed) failed=count(!r.ok for r in completed)
        flush(stderr)
    end

    config["run_finished_at"] = string(now())
    config["total_elapsed_s"] = round(time() - overall_started; digits=3)
    flush_out(config)
    @info "wrote inventory-generated benchmark" OUT rows=length(ROWS)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
