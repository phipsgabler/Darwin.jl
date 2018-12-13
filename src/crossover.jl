using Distributions: Dirichlet

export crossover,
    crossover!,
    CrossoverStrategy,
    CrossoverResult,
    setup!

export ArithmeticCrossover,
    LiftedCrossover,
    NoCrossover,
    UniformCrossover


abstract type CrossoverStrategy{T, K, P} end


setup!(strategy::CrossoverStrategy, model::AbstractEvolutionaryModel) = strategy


"""
    LiftedCrossover{T, C}(args)

Lifts a `CrossoverStrategy{I, K, P}` on `I`, to a new `CrossoverStrategy{T, K, P}` on `T`, 
by storing `C(args)`.  The actual application/lifting needs to be defined manually.
"""
struct LiftedCrossover{T, C, I, K, P} <: CrossoverStrategy{T, K, P}
    inner::C

    LiftedCrossover{T}(strategy::C) where {T, I, K, P, C<:CrossoverStrategy{I, K, P}} =
        new{T, C, I, K, P}(strategy)
    LiftedCrossover{T, C}(args...) where {T, I, K, P, C<:CrossoverStrategy{I, K, P}} =
        new{T, C, I, K, P}(C(args...))
end


"""
    crossover!(parents, strategy[, generation]) -> children

Perform crossover between `parents`.
"""
crossover!(parents::Family{T, K}, strategy::CrossoverStrategy{T, K, P}) where {T, K, P} =
    Individual.(crossover!(genome.(parents), strategy))
crossover!(parents::Family{T, K},  strategy::CrossoverStrategy{T, K, P},
           generation::Int) where {T, K, P} =
    Individual.(crossover!(genome.(parents), strategy, generation))
crossover!(parents::NTuple{K, T}, strategy::CrossoverStrategy{T, K, P},
           generation::Int) where {T, K, P} =
    crossover!(parents, strategy)



struct NoCrossover{T, N} <: CrossoverStrategy{T, N, N} end

crossover!(parents::Family{<:Any, N}, strategy::NoCrossover{N}) where {N} = parents


struct ArithmeticCrossover{T, K, P} <: CrossoverStrategy{AbstractVector{T}, K, P}
    rate::Float64
end

function crossover!(parents::NTuple{2, AbstractVector{T}},
                    strategy::ArithmeticCrossover{T, 2, 2}) where {T}
    if rand() < strategy.rate
        mixing = rand()
        return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
                (1 - mixing) .* parents[2] .+ mixing .* parents[1])
    else
         return parents
    end
end


struct UniformCrossover{T, K, P} <: CrossoverStrategy{AbstractVector{T}, K, P}
    p::Float64
    
    UniformCrossover{T, K, P}(p = 0.5) where {T, K, P} = new{T, K, P}(p)
    UniformCrossover{T, N}(p = 0.5) where {T, N} = new{T, N, N}(p)
end

function crossover!(parents::NTuple{2, AbstractVector{T}},
                    strategy::UniformCrossover{T, 2, 1}) where {T}
    crossover_points = rand(length(parents[1])) .≤ strategy.p
    (map(ifelse, crossover_points, parents...),)
end

function crossover!(parents::NTuple{2, AbstractVector{T}},
                    strategy::UniformCrossover{T, 2, 2}) where {T}
    crossover_points = rand(length(parents[1])) .≤ strategy.p
    map(ifelse, crossover_points, parents...), map(ifelse, .~crossover_points, parents...)
end

# function crossover!(parents::Family{T, 1}, strategy::ArithmeticCrossover{T, 1}) where {T}
#     if rand() < strategy.rate
#         mixing = rand()
#         return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
#                 (1 - mixing) .* parents[2] .+ mixing .* parents[1])
#     else
#          return parents
#     end
# end

## TODO: move D to strategy struct, use something different than "N random parent permutations"?
# function crossover(parents::NTuple{N, <:AbstractArray{T}}, strategy::ArithmeticCrossover{T, N}) where {T}
#     D = Dirichlet(N, 1)
    
#     if rand() < strategy.rate
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

# function crossover!((p₁, p₂)::NTuple{2, <:AbstractArray{T}}, strategy::UniformCrossover{T}) where {T}
#     @assert length(p₁) == length(p₂)
#     l = length(p₁)
    
#     crossover_points = rand(Uniform(strategy.p), l)
#     p₁[crossover_points], p₂[crossover_points] = p₂[crossover_points], p₁[crossover_points]

#     p₁, p₂
# end


