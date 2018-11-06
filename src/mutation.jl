abstract type MutationStrategy end

setup!(strategy::MutationStrategy) = strategy

"""
    mutate!(population, strategy[, generation]) -> population

Mutate `population` in place.
"""
mutate!(population, ::MutationStrategy) = population
mutate!(population, strategy, generation) = mutate!(population, strategy)
