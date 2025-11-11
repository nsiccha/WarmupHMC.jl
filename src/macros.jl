broadcastable(x) = false # avoid dotting spliced objects (e.g. view calls inserted by @view)
# don't add dots to dot operators
broadcastable(x::Symbol) = (!Base.isoperator(x) || first(string(x)) != '.' || x === :..) && x !== :(:)
broadcastable(x::Expr) = x.head !== :$
unbroadcast(x) = x
function unbroadcast(x::Expr)
    if x.head === :.=
        Expr(:(=), x.args...)
    elseif x.head === :block # occurs in for x=..., y=...
        Expr(:block, Base.mapany(unbroadcast, x.args)...)
    else
        x
    end
end
__broadcasted__(x) = x
function __broadcasted__(x::Expr)
    broadcasted = :(Base.broadcasted)
    broadcastargs = Base.mapany(__broadcasted__, x.args)
    return if x.head === :call && broadcastable(x.args[1])
        Expr(:call, broadcasted, broadcastargs...)
    elseif x.head === :comparison
        error()
        Expr(:comparison, (iseven(i) && broadcastable(arg) && arg isa Symbol && Base.isoperator(arg) ?
                               Symbol('.', arg) : arg for (i, arg) in pairs(broadcastargs))...)
    elseif x.head === :$
        x.args[1]
    elseif x.head === :let # don't add dots to `let x=...` assignments
        Expr(:let, unbroadcast(broadcastargs[1]), broadcastargs[2])
    elseif x.head === :for # don't add dots to for x=... assignments
        Expr(:for, unbroadcast(broadcastargs[1]), broadcastargs[2])
    elseif (x.head === :(=) || x.head === :function || x.head === :macro) &&
           Meta.isexpr(x.args[1], :call) # function or macro definition
        Expr(x.head, x.args[1], broadcastargs[2])
    elseif x.head === :(<:) || x.head === :(>:)
        Expr(:call, broadcasted, x.head, broadcastargs...)
    else
        head = String(x.head)::String
        if last(head) == '=' && first(head) != '.' || head == "&&" || head == "||"
            Expr(:call, broadcasted, x.head, broadcastargs...)
        else
            Expr(x.head, broadcastargs...)
        end
    end
end
macro broadcasted(x)
    esc(__broadcasted__(x))
end
macro bsum(x)
    :(sum($(esc(__broadcasted__(x)))))
end