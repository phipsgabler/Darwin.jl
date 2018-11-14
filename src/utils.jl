import Distributions
const D = Distributions
import Base: +

export ParametrizedFunction

struct TimeInfo
    time::Float64
    bytes::Int64
    gctime::Float64
    memallocs::Base.GC_Diff
end

const notime = TimeInfo(0.0, 0, 0.0, Base.GC_Diff(0, 0, 0, 0, 0, 0, 0, 0, 0))

macro timeinfo(expr)
    quote
        val , time, bytes, gctime, memallocs = @timed $(esc(expr))
        val, TimeInfo(time, bytes, gctime, memallocs)
    end
end

maximumby(f, collection) = collection[argmax(f(x) for x in collection)]

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


# Will work if `D` is a location family.
struct Shifted{T<:D.ContinuousUnivariateDistribution} <: D.ContinuousUnivariateDistribution
    dist::T
    δ::Float64
end

Shifted(dist::T, δ) where {T} = Shifted{T}(dist, convert(Float64, δ))

D.rand(d::Shifted) = D.rand(d.dist) + d.δ
D.pdf(d::Shifted, x::Real) = D.pdf(d.dist, x - d.δ)
D.logpdf(d::Shifted, x::Real) = D.logpdf(d.dist, x - d.δ)
D.cdf(d::Shifted, x::Real) = D.cdf(d.dist, x - d.δ)
D.quantile(d::Shifted, q::Real) = D.quantile(d.dist, q) + d.δ
D.minimum(d::Shifted) = D.minimum(d.dist) + d.δ
D.maximum(d::Shifted) = D.maximum(d.dist) + d.δ
D.insupport(d::Shifted, x::Real) = D.insupport(d.dist, x - d.δ)
D.mean(d::Shifted) = D.mean(d.dist) + d.δ
D.var(d::Shifted) = D.var(d.dist)
D.modes(d::Shifted) = D.modes(d.dist) .+ d.δ
D.mode(d::Shifted) = D.mode(d.dist) + d.δ
# StatsBase.skewness(d::Shifted) = StatsBase.skewness(d.D)
# StatsBase.kurtosis(d::Distribution, correction::Bool) = StatsBase.kurtosis(d.D, correction)
# StatsBase.entropy(d::Shifted, b::Real) = StatsBase.entropy(d.D, b)
# mgf(d::Shifted, ::Any) = ???
# cf(d::Shifted, ::Any) = ???
