export crossover!,
    CrossoverStrategy,
    setup!

abstract type CrossoverStrategy end

setup!(strategy::CrossoverStrategy, model) = strategy

"""
    crossover!(children, parents, selection, strategy[, generation]) -> children

Perform crossover on `parents[selection]`, and write the result to `children`.  Return the new
`children`.  The resulting children should be new or copied.
"""
crossover!(children,  parents, selection, ::CrossoverStrategy) = copyto!(children, population)
crossover!(children,  parents, selection, strategy::CrossoverStrategy, generation) =
    crossover!(children, parents, selection, strategy)

