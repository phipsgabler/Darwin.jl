export select,
    SelectionStrategy,
    setup!

abstract type SelectionStrategy{S} end

setup!(strategy::SelectionStrategy, model::AbstractEvolutionaryModel) = strategy

"""
    selection(model[, generation]) -> selection

Select parts of population of the model to be used in breeding.  Should compare fitnesses using 
`isless`, if that is relevant.
"""
selection(model::AbstractEvolutionaryModel, generation) = selection(model)
# selection(model::AbstractEvolutionaryModel, generation) = selection(model.population, model.strategy, generation)



# struct TruncationSelection{μ, F<:AbstractFitness} <: SelectionStrategy{AbstractVector}
#     fitness::F
# end

# function selection(population::AbstractVector, strat::TruncationSelection{μ},
#                    generation::Integer) where {μ}
#     partialsortperm(population, 1:μ, by = assess!, rev = true)
# end
