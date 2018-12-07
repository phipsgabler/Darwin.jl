export crossover,
    CrossoverStrategy,
    setup!

export ArithmeticCrossover

abstract type CrossoverStrategy{S} end

setup!(strategy::CrossoverStrategy, model::AbstractEvolutionaryModel) = strategy

"""
    crossover(parents, strategy[, generation]) -> children

Perform crossover between `parents`.
"""
crossover(parents::S, ::CrossoverStrategy{S}) where {S} = parents
crossover(parents::S,  strategy::CrossoverStrategy{S}, generation) where {S} =
    crossover(parents, strategy)


struct ArithmeticCrossover{T} <: CrossoverStrategy{NTuple{<:AbstractArray, 2}}
    rate::Float64
end

function crossover(parents::NTuple{<:AbstractArray{T}, 2}, strat::ArithmeticCrossover{T}) where {T}
    if rand() < strat.rate
        mixing = rand()
        return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
                (1 - mixing) .* parents[2] .+ mixing .* parents[1])
    else
         return parents[1], parents[2]
    end
end
