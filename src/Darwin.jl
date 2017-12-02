module Darwin

abstract type AbstractEvolutionaryModel end
export AbstractEvolutionaryModel

abstract type AbstractEvolutionarySolution end
export AbstractEvolutionarySolution

function evolve end

include("utils.jl")
include("GA.jl")

export GAModel, evolve, genetype, populationtype

end # module
