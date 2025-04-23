struct Progress{P,I}
    parent::P
    info::I
end
Base.parent(x::Progress) = x.parent
info(x::Progress) = x.info
initialize_multi_progress!(::Nothing, args...; kwargs...) = nothing
initialize_progress!(::Nothing; n) = fill(nothing, n)
initialize_progress!(::Nothing, args...; kwargs...) = nothing
update_progress!(::Nothing, args...; kwargs...) = nothing
finalize_progress!(::Nothing, args...; kwargs...) = nothing
finalize_progress!(::Vector{Nothing}, args...; kwargs...) = nothing