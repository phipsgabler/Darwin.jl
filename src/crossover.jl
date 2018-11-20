export crossover,
    CrossoverStrategy,
    setup!

abstract type CrossoverStrategy{S} end

setup!(strategy::CrossoverStrategy, model::AbstractEvolutionaryModel) = strategy

"""
    crossover(parents, strategy[, generation]) -> children

Perform crossover between `parents`.
"""
crossover(parents::S, ::CrossoverStrategy{S}) where {S} = parents
crossover(parents::S,  strategy::CrossoverStrategy{S}, generation) where {S} =
    crossover(parents, strategy)
