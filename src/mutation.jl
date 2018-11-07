export mutate!,
    MutationStrategy,
    setup!

abstract type MutationStrategy end

setup!(strategy::MutationStrategy, model) = strategy

"""
    mutate!(population, strategy[, generation]) -> population

Mutate `population` in place.
"""
mutate!(population, ::MutationStrategy) = population
mutate!(population, strategy, generation) = mutate!(population, strategy)
