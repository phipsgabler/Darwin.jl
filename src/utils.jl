export ParametrizedFunction

struct TimeInfo
    time::Float64
    bytes::Int64
    gctime::Float64
    memallocs::Base.GC_Diff
end

const notime = TimeInfo(0.0, 0, 0.0, Base.GC_Diff(0, 0, 0, 0, 0, 0, 0, 0, 0))


mutable struct ParametrizedFunction <: Function
    f
    parameters::Vector
end

(pf::ParametrizedFunction)(x) = pf.f(x, pf.parameters...)


# see https://github.com/JuliaLang/julia/blob/master/doc/src/devdocs/ast.md
# using MacroTools
# macro parametrized(fundef)
#     fsplit = splitdef(fundef)
#     tname = gensym(fsplit[:name])
#     obj = gensym(:fp)
#     arg = gensym(:x)
#     funfield = gensym(:fun)
#     kwargs = [gensym(kw) for kw in fsplit[:kwargs]]

#     _struct = Expr(:struct, true, # mutable = true
#                    Expr(:(<:), tname, :Function),
#                    Expr(:block, funfield, kwargs...))
#     _call = Expr(:(=), Expr(:call, Expr(:(::), obj, tname), arg),
#                  Expr(:call,
#                       Expr(:(.), obj, Meta.quot(funfield)),
#                       arg,
#                       (Expr(:(.), obj, Meta.quot(kw)) for kw in kwargs)...))
#     Expr(:block, _struct, _call)
# end
