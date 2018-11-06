abstract type CrossoverStrategy end

setup!(strategy::CrossoverStrategy) = strategy

"""
    crossover!(children, parents, strategy[, generation]) -> children

Perform crossover on `parents`, and write the result to `children`.  Return the new `children`.
"""
crossover!(children,  parents, ::CrossoverStrategy) = children .= population
crossover!(children,  parents, strategy::CrossoverStrategy, generation) =
    crossover!(children, parents, strategy)

