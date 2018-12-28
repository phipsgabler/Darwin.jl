module Darwin

export AbstractEvolutionaryModel,
    FitnessValue


abstract type AbstractEvolutionaryModel end

const FitnessValue = Float64


include("utils.jl")
include("individual.jl")
include("rates.jl")
include("fitness.jl")
include("selection.jl")
include("mutation.jl")
include("crossover.jl")

include("GA.jl")

end # module
