# TestItemRunner entry point — the stack HTMX.jl / StanBlocks.jl / AlgebraOfVega.jl
# already use. Run it against the test environment beside this file:
#
#     julia --project=web/src/test -e 'include("web/src/test/runtests.jl")'
#
# Selectors go after a `--`, and compose (they AND together):
#
#     -- --list                       # print `path::name [tags]`, run nothing
#     -- --skip-tag=enzyme            # everything except the Enzyme items
#     -- --tag=reparametrization
#     -- --name=halo                  # substring, case-insensitive
#     -- --file=golden_awm.jl         # substring of the repo-relative path
#     -- --htmxo-test='web/src/test/golden_awm.jl::the golden adaptive_warmup_mcmc harness'
#
# An unknown selector, an unknown `--tag`/`--htmxo-test` value, or a selection
# that matches zero items is an ERROR, not a silent green run — a mistyped
# `--tag=enzym` that quietly ran nothing would report success while having tested
# nothing at all.

using TestItemRunner
using LinearAlgebra: BLAS

# Reproducibility: the adaptive transformation update runs multithreaded BLAS,
# whose reduction order is not deterministic run-to-run. Every byte-identity
# comparison in this suite is verified under this pin.
#
# It is ALSO in the `Determinism` snippet in `setup.jl`, and that is not
# redundant: an item run on its own — from the VS Code test explorer, or via
# `--htmxo-test=` — never executes this file, so an item that needs the pin must
# carry it itself. This line covers the whole-suite run.
BLAS.set_num_threads(1)

# The scan root is pinned rather than taken from `@run_package_tests`, which
# expands to `run_tests(joinpath(dirname(@__FILE__), ".."))`. That idiom assumes
# the runner sits at `<pkg>/test/`; this suite lives at `web/src/test/`, so the
# macro would scan `web/src/` — sweeping in `WarmupHMCWeb.jl` and anything else
# the web app grows. Pointing at this directory keeps the suite's boundary equal
# to its directory.
#
# `REPO_ROOT` is derived from this file's location and NOT from
# `pkgdir(WarmupHMC)`, so `--file=` selectors keep working whichever checkout
# WarmupHMC itself resolves from.
const TEST_ROOT = @__DIR__
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

test_path(ti) = replace(relpath(ti.filename, REPO_ROOT), '\\' => '/')

function option_value(args, i, name)
    i < length(args) || error("Missing value for $name")
    return args[i + 1], i + 1
end

function parse_test_args(args)
    names = String[]
    tags = Symbol[]
    skip_tags = Symbol[]
    files = String[]
    exact = nothing
    list = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--list"
            list = true
        elseif startswith(arg, "--name=")
            push!(names, split(arg, "="; limit=2)[2])
        elseif arg == "--name"
            value, i = option_value(args, i, arg)
            push!(names, value)
        elseif startswith(arg, "--tag=")
            push!(tags, Symbol(lstrip(split(arg, "="; limit=2)[2], ':')))
        elseif arg == "--tag"
            value, i = option_value(args, i, arg)
            push!(tags, Symbol(lstrip(value, ':')))
        # `--tag` is an AND-SELECTION, so it cannot express "everything except
        # X". CI needs exactly that: the nine-row matrix must run the whole suite
        # EXCEPT the `:enzyme` items, whose backend is heavy to compile and is
        # covered once by a separate job. `--skip-tag` is the complement, and
        # like `--tag` it accepts a leading colon.
        elseif startswith(arg, "--skip-tag=")
            push!(skip_tags, Symbol(lstrip(split(arg, "="; limit=2)[2], ':')))
        elseif arg == "--skip-tag"
            value, i = option_value(args, i, arg)
            push!(skip_tags, Symbol(lstrip(value, ':')))
        elseif startswith(arg, "--file=")
            push!(files, replace(split(arg, "="; limit=2)[2], '\\' => '/'))
        elseif arg == "--file"
            value, i = option_value(args, i, arg)
            push!(files, replace(value, '\\' => '/'))
        elseif startswith(arg, "--htmxo-test=")
            isnothing(exact) || error("--htmxo-test may only be specified once")
            exact = split(arg, "="; limit=2)[2]
        elseif arg == "--htmxo-test"
            isnothing(exact) || error("--htmxo-test may only be specified once")
            exact, i = option_value(args, i, arg)
        else
            error("Unknown test selector: $arg")
        end
        i += 1
    end

    exact_parts = isnothing(exact) ? nothing : split(replace(exact, '\\' => '/'), "::"; limit=2)
    isnothing(exact_parts) || length(exact_parts) == 2 || error("--htmxo-test must be file::name")
    return (; names, tags, skip_tags, files, exact_parts, list)
end

const TEST_SELECTION = parse_test_args(ARGS)

# Populated as the filter sweeps every discovered item, so an unknown selection
# can be named precisely afterwards instead of reported as "matched nothing".
const AVAILABLE_TAGS = Set{Symbol}()
const AVAILABLE_KEYS = Set{String}()
const SELECTED_KEYS = Set{String}()

function selected(ti)
    path = test_path(ti)
    key = path * "::" * ti.name
    union!(AVAILABLE_TAGS, ti.tags)
    push!(AVAILABLE_KEYS, key)

    selection = TEST_SELECTION
    matches = all(name -> occursin(lowercase(name), lowercase(ti.name)), selection.names) &&
              all(tag -> tag in ti.tags, selection.tags) &&
              !any(tag -> tag in ti.tags, selection.skip_tags) &&
              all(file -> occursin(lowercase(file), lowercase(path)), selection.files) &&
              (isnothing(selection.exact_parts) ||
               (path == selection.exact_parts[1] && ti.name == selection.exact_parts[2]))

    matches && push!(SELECTED_KEYS, key)
    if selection.list && matches
        tag_text = isempty(ti.tags) ? "" : " [" * join(sort!(string.(collect(ti.tags))), ", ") * "]"
        println(path, "::", ti.name, tag_text)
        return false
    end
    return matches
end

TestItemRunner.run_tests(TEST_ROOT; filter=selected, verbose=true)

# --- selection sanity, AFTER the run so the filter has seen every item --------
#
# `--skip-tag` is deliberately NOT validated against `AVAILABLE_TAGS`: CI passes
# `--skip-tag=enzyme` unconditionally, and skipping a tag with no items is a
# no-op rather than a mistake.
function require_known(label, requested, available)
    unknown = setdiff(requested, available)
    isempty(unknown) || error(
        "Unknown $label selection(s): " * join(sort!(string.(collect(unknown))), ", ") *
        ". Available: " * join(sort!(string.(collect(available))), ", "))
end

require_known("tag", Set(TEST_SELECTION.tags), AVAILABLE_TAGS)
isnothing(TEST_SELECTION.exact_parts) ||
    require_known("test", Set([join(TEST_SELECTION.exact_parts, "::")]), AVAILABLE_KEYS)

if isempty(SELECTED_KEYS) &&
   !(isempty(TEST_SELECTION.names) && isempty(TEST_SELECTION.tags) &&
     isempty(TEST_SELECTION.skip_tags) && isempty(TEST_SELECTION.files) &&
     isnothing(TEST_SELECTION.exact_parts))
    error("The requested selection matched zero tests.")
end
