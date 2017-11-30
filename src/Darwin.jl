module Darwin

abstract type AbstractEvolutionaryModel end

export AbstractEvolutionaryModel

function evolve end

include("GA.jl")

export GeneticModel, evolve, genetype, populationtype

end # module
