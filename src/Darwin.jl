module Darwin


export AbstractEvolutionaryModel,
    AbstractEvolutionarySolution


abstract type AbstractEvolutionaryModel end

abstract type AbstractEvolutionarySolution end


function evolve end

include("utils.jl")
include("GA.jl")

end # module
