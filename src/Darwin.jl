module Darwin

abstract type AbstractEvolutionaryModel end


"""
    populationtype(model)

Type of population used in `model`.
"""
function populationtype end

populationtype(t::AbstractEvolutionaryModel) = populationtype(typeof(t))


"""
    populationtype(model)

Type of genome (ie., individuals in population) used in `model`.
"""
function genotype end

genotype(t::AbstractEvolutionaryModel) = genotype(typeof(t))


include("utils.jl")
include("fitness.jl")
include("selection.jl")
include("mutation.jl")
include("crossover.jl")

include("GA.jl")

end # module
