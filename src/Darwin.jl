module Darwin

abstract type AbstractEvolutionaryModel end

export AbstractEvolutionaryModel

function evolve end

include("GA.jl")

export GAModel, evolve, genetype, populationtype

end # module
