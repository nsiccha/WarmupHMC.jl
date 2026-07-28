# The README names every export, and nothing but this file forces that.
#
# WHY THIS TEST EXISTS
#
# The paragraph it guards used to read "Exports `adaptive_warmup_mcmc` plus 4
# reparametrization types". That was accurate when written and described the
# whole surface. Three more exports arrived afterwards; the sentence kept
# rendering, kept looking like a complete answer, and never produced a merge
# conflict — because a README cannot fail. It was found by reading it against
# the code, which is not a process.
#
# The same file's `See also` block was worse: it named ReactiveHMC.jl as "the
# underlying HMC implementation" when `DynamicHMC` had been the dependency for
# two and a half years already. That one was never true — not stale, wrong on
# the day it was written — and it survived every CI run this repo has ever had.
# A claim nothing checks is not documentation, it is a guess with good
# typography.
#
# WHY `names(...)` AND NOT A GREP OF THE `export` LINE
#
# `names(WarmupHMC)` is what the module actually exports at run time. A grep
# anchors on the current *spelling* of the export statement, so a name added
# through a second `export`, a macro, or a rename would slip past while the
# grep kept reporting success — the same failure one level down. Ask the
# module, not the source text.
@testset "README names every export" begin
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)

    # HTML comments are instructions to whoever edits the file next, not
    # documentation for whoever reads it. Naming an export only inside one must
    # NOT satisfy this test — otherwise the fix for a failure here is to hide
    # the name where no reader will find it.
    rendered = replace(readme, r"<!--.*?-->"s => "")

    exported = filter(!=(:WarmupHMC), names(WarmupHMC))

    # Guard the guard: if `names` ever returns nothing useful, the loop below
    # passes vacuously and this file becomes a test that cannot fail.
    @test !isempty(exported)

    for n in exported
        @test occursin("`$n`", rendered)
    end
end
