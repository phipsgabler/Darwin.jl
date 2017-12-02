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


