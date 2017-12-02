module Darwin

abstract type AbstractEvolutionaryModel end
export AbstractEvolutionaryModel

abstract type AbstractEvolutionarySolution end
export AbstractEvolutionarySolution

function evolve end

include("utils.jl")
include("GA.jl")

export GAModel, genetype, populationtype
export evolve, init, evolvestep!
export ParametrizedFunction

end # module
