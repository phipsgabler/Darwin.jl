module Darwin

abstract type AbstractEvolutionaryProblem end

export AbstractEvolutionaryProblem

function evolve end

include("GA.jl")

export AbstractEvolutionaryAlgorithm

export GeneticProblem
export mutate, crossover, fitness, evolve

end # module
