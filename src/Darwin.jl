module Darwin

export AbstractEvolutionaryModel,
    FitnessValue


abstract type AbstractPopulationModel{G} end

const FitnessValue = Float64


include("utils.jl")
include("individual.jl")
include("rates.jl")
include("fitness.jl")
include("population_model.jl")

include("selection.jl")
include("mutation.jl")
include("crossover.jl")

include("GA.jl")

end # module
