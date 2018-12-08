using Distributions: Dirichlet

export crossover,
    crossover!,
    CrossoverStrategy,
    setup!

export ArithmeticCrossover

abstract type CrossoverStrategy{S} end

setup!(strategy::CrossoverStrategy, model::AbstractEvolutionaryModel) = strategy

"""
    crossover!(parents, strategy[, generation]) -> children

Perform crossover between `parents`.
"""
crossover!(parents::S, ::CrossoverStrategy{S}) where {S} = parents
crossover!(parents::S,  strategy::CrossoverStrategy{S}, generation) where {S} =
    crossover!(parents, strategy)

crossover(parents::S, strat::CrossoverStrategy{S}) where {S} = crossover!(copy.(parents), strat)
crossover(parents::S,  strategy::CrossoverStrategy{S}, generation) where {S} =
    crossover!(copy.(parents), strat, generation)


struct ArithmeticCrossover{T, N} <: CrossoverStrategy{NTuple{N, <:AbstractArray{<:Real}}}
    rate::Float64
end

function crossover!(parents::NTuple{2, <:AbstractArray{T}}, strat::ArithmeticCrossover{T, 2}) where {T}
    if rand() < strat.rate
        mixing = rand()
        return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
                (1 - mixing) .* parents[2] .+ mixing .* parents[1])
    else
         return parents
    end
end

## TODO: move D to strategy struct, use something different than "N random parent permutations"?
# function crossover(parents::NTuple{N, <:AbstractArray{T}}, strat::ArithmeticCrossover{T, N}) where {T}
#     D = Dirichlet(N, 1)
    
#     if rand() < strat.rate
#         mixing = rand(D)
#         return ntuple(sum(parents[randperm(N)] .* mixing), N)
        
#         return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
#                 (1 - mixing) .* parents[2] .+ mixing .* parents[1])
#     else
#          return parents
#     end
# end


# struct UniformCrossover{T} <: CrossoverStrategy{NTuple{2, <:AbstractArray{T}}}
#     p::Float64
#     _dist::Bernoulli
# end

# function crossover!((p₁, p₂)::NTuple{2, <:AbstractArray{T}}, strat::UniformCrossover{T}) where {T}
#     @assert length(p₁) == length(p₂)
#     l = length(p₁)
    
#     crossover_points = rand(Uniform(strat.p), l)
#     p₁[crossover_points], p₂[crossover_points] = p₂[crossover_points], p₁[crossover_points]

#     p₁, p₂
# end


