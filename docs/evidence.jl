# The evidence appendix is the Markdown projection of the same HTMXObjects
# semantic tree served by WarmupHMCWeb. The structural JSON classification and
# table construction live in `benchmark/evidence_semantics.jl`; neither medium
# owns a second renderer.

import Markdown

Base.include(
    Base.@__MODULE__,
    joinpath(@__DIR__, "benchmark", "evidence_semantics.jl"),
)

"""
    md_evidence_all(; max_rows=40) -> Markdown.MD

Project every checked-in benchmark result through HTMXObjects' public Markdown
projection. Large raw-run tables are represented as `SemanticUnavailable`
rather than truncated, preserving the same contract as the interactive app.
"""
function md_evidence_all(; max_rows::Int=40)
    semantic = WarmupHMCBenchmarkEvidence.benchmark_all_semantic(; max_rows)
    Markdown.parse(repr(MIME"text/markdown"(), semantic))
end
