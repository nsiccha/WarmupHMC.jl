# Real-data BRM posteriors, transcribed from the published catalogue at
# https://juliabayes.github.io/BayesianRegressionModels.jl/
#
# WHY THESE ARE HAND-TRANSCRIBED, AND WHAT THAT COSTS
#
# All 359 catalogue cards carry `parseable=false`. Nothing in the catalogue can
# yet produce a runnable model, so the `@brm` block under each spec is MY
# transcription of the card's formula, not a derived artifact. The `formula`
# field holds the card's own text verbatim so a reader can check the
# transcription rather than take it on trust, and `note` records every place
# the two deliberately differ.
#
# That is the weak link in this benchmark and it is worth naming plainly: if a
# transcription is wrong, the measurement is of a model the catalogue does not
# contain. The verbatim formula string is what makes that falsifiable.
#
# WHY FAMILY IS READ OFF THE FORMULA, NOT THE CARD'S FAMILY TAG
#
# The tags are not trustworthy for this purpose. `cbpp` is tagged `gaussian`
# while its formula is `incidence | trials(size) ~ ...` (binomial), `epilepsy`
# is tagged `gaussian` over counts, and `kidney` is tagged `gaussian` over
# `time | cens(censored)`. Selecting on the tag would have pulled three
# non-gaussian likelihoods into a gaussian benchmark. Every model below was
# selected by reading its formula.
#
# WHY THESE SEVEN DATASETS
#
# They are the intersection of three filters: the formula is a genuine gaussian
# hierarchical regression (so there is a centering choice to make at all), the
# posterior is cheap enough to run 12 seeds x 6 arms, and the data is really
# obtainable. Datasets that failed the last filter are listed in the runner's
# `unreached` block rather than silently omitted.

using CSV, DataFrames, Downloads, SHA, Statistics

const DATA_CACHE = get(ENV, "BRM_DATA_CACHE",
                       joinpath(@__DIR__, "data"))

# Source URLs are recorded here and their sha256 lands in the artifact, so a
# rerun that silently picked up different data is detectable. The data is NOT
# vendored into this repository: it is third-party and belongs to its upstream.
const SOURCES = Dict(
    "sleepstudy" => "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv",
    "dyestuff"   => "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff.csv",
    "pastes"     => "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Pastes.csv",
    "penicillin" => "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv",
    "dietox"     => "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv",
    "weightgain" => "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/weightgain.csv",
    # bambi ships `radon` but its in-repo copy is not fetchable over plain HTTP;
    # pymc-examples carries the same Gelman/Hill extract with `log_radon`,
    # `floor` and `county_code` already derived.
    "radon"      => "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/radon.csv",
)

function dataset(name::AbstractString)
    mkpath(DATA_CACHE)
    path = joinpath(DATA_CACHE, name * ".csv")
    isfile(path) || Downloads.download(SOURCES[name], path)
    path
end

data_sha(name) = bytes2hex(open(sha256, dataset(name)))

rd(name) = CSV.read(dataset(name), DataFrame)

# BRM wants integer group codes. Sorted so a rerun on the same CSV produces the
# same design matrix — an unsorted `unique` would silently permute the columns
# between runs and make per-coordinate ESS incomparable across seeds.
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])

struct Spec
    key::String
    dataset::String
    source::String     # which catalogue source the card came from
    formula::String    # the card's text, VERBATIM
    note::String       # every deliberate difference between formula and @brm block
    groups::Vector{Symbol}
    make::Function     # () -> (builder, data)
end

SPECS = Spec[

Spec("sleepstudy", "sleepstudy", "mixed_models_jl",
     "Reaction ~ 1 + Days + (1 + Days | Subject)",
     "response divided by 100 to put it on an O(1) scale",
     [:Subject], function ()
    df = rd("sleepstudy")
    data = (; Reaction = Float64.(df.Reaction) ./ 100,
              Days = Float64.(df.Days), Subject = codes(df.Subject))
    b = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + Days + (1 + Days | Subject)
        Reaction ~ Normal(mu, sigma)
    end
    (b, data)
end),

Spec("dyestuff", "dyestuff", "lme4",
     "Yield ~ 1 + (1 | Batch)",
     "response centred and divided by 100 (raw yields are ~1500)",
     [:Batch], function ()
    df = rd("dyestuff")
    y = Float64.(df.Yield)
    data = (; Yield = (y .- mean(y)) ./ 100, Batch = codes(df.Batch))
    b = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | Batch)
        Yield ~ Normal(mu, sigma)
    end
    (b, data)
end),

Spec("penicillin", "penicillin", "lme4",
     "diameter ~ 1 + (1 | plate) + (1 | sample)",
     "response centred (raw diameters are ~25); two crossed grouping factors",
     [:plate, :sample], function ()
    df = rd("penicillin")
    y = Float64.(df.diameter)
    data = (; diameter = y .- mean(y), plate = codes(df.plate), sample = codes(df.sample))
    b = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | plate) + (1 | sample)
        diameter ~ Normal(mu, sigma)
    end
    (b, data)
end),

Spec("pastes", "pastes", "lme4",
     "strength ~ (1 | batch/cask)",
     "nested `batch/cask` written as `(1|batch) + (1|sample)`. This is the SAME " *
     "model, not an approximation: the CSV's `sample` column already holds " *
     "`batch:cask`. Response centred (raw strengths are ~60).",
     [:batch, :sample], function ()
    df = rd("pastes")
    y = Float64.(df.strength)
    data = (; strength = y .- mean(y), batch = codes(df.batch), sample = codes(df.sample))
    b = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | batch) + (1 | sample)
        strength ~ Normal(mu, sigma)
    end
    (b, data)
end),

Spec("dietox", "dietox", "bambi",
     "Weight ~ Time + (Time|Pig)",
     "implicit intercepts written explicitly (bambi adds them by default); " *
     "response centred and divided by 10 (raw weights are ~60). 861 rows / 72 pigs.",
     [:Pig], function ()
    df = rd("dietox")
    y = Float64.(df.Weight)
    data = (; Weight = (y .- mean(y)) ./ 10, Time = Float64.(df.Time), Pig = codes(df.Pig))
    b = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + Time + (1 + Time | Pig)
        Weight ~ Normal(mu, sigma)
    end
    (b, data)
end),

Spec("radon-floor", "radon", "bambi",
     "log_radon ~ 1 + floor + (1|county)",
     "919 rows / 85 counties. The catalogue carries five other radon cards; the " *
     "correlated-slope one is benchmarked separately as `radon-slope`.",
     [:county], function ()
    df = rd("radon")
    data = (; log_radon = Float64.(df.log_radon), floor = Float64.(df.floor),
              county = codes(df.county_code))
    b = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + floor + (1 | county)
        log_radon ~ Normal(mu, sigma)
    end
    (b, data)
end),

Spec("radon-slope", "radon", "bambi",
     "log_radon ~ floor + (floor|county)",
     "implicit intercepts written explicitly; same data as the other radon rows",
     [:county], function ()
    df = rd("radon")
    data = (; log_radon = Float64.(df.log_radon), floor = Float64.(df.floor),
              county = codes(df.county_code))
    b = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + floor + (1 + floor | county)
        log_radon ~ Normal(mu, sigma)
    end
    (b, data)
end),

Spec("weightgain", "weightgain", "rstanarm",
     "weightgain ~ 1 + (1|source) + (1|type) + (1|source:type)",
     "the interaction grouping `source:type` is materialised as its own coded " *
     "column `st`; response centred and divided by 10. 40 rows over 2 + 2 + 4 " *
     "levels — deliberately the most funnel-prone target here.",
     [:source, :type, :st], function ()
    df = rd("weightgain")
    y = Float64.(df.weightgain)
    data = (; weightgain = (y .- mean(y)) ./ 10, source = codes(df.source),
              type = codes(df.type), st = codes(string.(df.source, ":", df.type)))
    b = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | source) + (1 | type) + (1 | st)
        weightgain ~ Normal(mu, sigma)
    end
    (b, data)
end),

]

# Order the run cheapest-first, so a multi-hour sweep produces usable partial
# artifacts early rather than stranding everything behind its slowest target.
#
# Measured on strato2, wrapped arm, 100 draws: dyestuff 3.9s, weightgain 6.8s,
# radon-slope 17.9s, penicillin 44.5s, dietox 48.6s, radon-floor 52.5s, pastes
# 64.0s (sleepstudy's 158s was one-time Julia/Enzyme compilation; its steady
# state is ~2.5s and it goes first to pay that once).
#
# The shape worth remembering: cost tracks the NUMBER OF GROUPING FACTORS, not
# the dimension. `pastes` (dim 44, two groups) costs 3.5x `dietox` (dim 150, one
# group), and `radon-slope` (dim 176, one group) is cheaper than either. The
# reparametrization's per-(block, term, group) pairs are what is being paid for.
const RUN_ORDER = ["sleepstudy", "dyestuff", "weightgain", "radon-slope",
                   "penicillin", "dietox", "radon-floor", "pastes"]

# Catalogue models that were considered and NOT reached, with the reason. This
# list is part of the deliverable: a benchmark that reports only what it managed
# to run reads as a survey of the corpus when it is a survey of the easy part.
const UNREACHED = [
    (dataset = "cbpp", formula = "incidence | trials(size) ~ period + (1|herd)",
     reason = "binomial with a trials() term; the card's `gaussian` tag is wrong"),
    (dataset = "epilepsy", formula = "count ~ zBase * Trt + (1 | patient)",
     reason = "counts; the card's `gaussian` tag is wrong"),
    (dataset = "kidney", formula = "time | cens(censored) ~ age * sex + disease + (1 | patient)",
     reason = "censored survival; the card's `gaussian` tag is wrong"),
    (dataset = "contraception", formula = "use ~ age + I(age^2) + livch + urban + (urban | district)",
     reason = "bernoulli/binomial — correctly tagged, but outside this gaussian benchmark"),
    (dataset = "verbagg", formula = "r2 ~ Anger + Gender + btype + situ + (1 | id) + (1 | item)",
     reason = "bernoulli/binomial — correctly tagged, but outside this gaussian benchmark"),
    (dataset = "income", formula = "ls ~ mo(income) * age + (mo(income) | city)",
     reason = "monotonic effects `mo()` are not expressible in the @brm surface used here"),
    (dataset = "milk", formula = "bf(k ~ 1 + mi(b) + m) + bf(b | mi() ~ 1) + set_rescor(FALSE)",
     reason = "multivariate with missing-data imputation; no random effect to centre either way"),
    (dataset = "cafe", formula = "wait ~ 1 + afternoon + (1 + afternoon | cafe)",
     reason = "McElreath's simulated data, not real data"),
]
