export select, SelectionStrategy

abstract type SelectionStrategy end

setup!(strategy::SelectionStrategy) = strategy

"""
    select(population, strategy[, generation]) -> indices

Select indices of population to be used in breeding.  Should compare fitnesses using `isless`, 
if that is relevant.
"""
select(population, ::SelectionStrategy) = eachindex(population)
select(population, strategy::SelectionStrategy, generation) = select(population, strategy)

