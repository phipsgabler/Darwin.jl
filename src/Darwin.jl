module Darwin

abstract type AbstractEvolutionaryModel end

export AbstractEvolutionaryModel

function evolve end

include("GA.jl")

export GeneticModel, evolve

end # module
