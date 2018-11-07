export select,
    SelectionStrategy,
    setup!

abstract type SelectionStrategy end

setup!(strategy::SelectionStrategy, model) = strategy

"""
    selection(population, strategy[, generation]) -> indices

Select parts of population to be used in breeding.  Should compare fitnesses using `isless`, 
if that is relevant.
"""
selection(population, ::SelectionStrategy) = eachindex(population)
selection(population, strategy::SelectionStrategy, generation) = selection(population, strategy)

