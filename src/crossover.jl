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
    crossover!(parents::Union{Family, NTuple}, strategy, generation) -> children

Perform crossover between `parents`.  You only need to define the method for `parents` being an
`NTuple`; this method will be automatically lifted to `Family`s (i.e., tuples of `Individual`s).
"""
crossover!(parents::Family{T, K},
           strategy::CrossoverStrategy{T, K, P},
           generation::Int) where {T, K, P} =
    Individual.(crossover!(genome.(parents), strategy, generation))



struct NoCrossover{T, N} <: CrossoverStrategy{T, N, N} end

crossover!(parents::NTuple{N}, strategy::NoCrossover{N}, generation::Integer) where {N} =
    parents


struct ArithmeticCrossover{T<:AbstractVector, K, P} <: CrossoverStrategy{T, K, P}
    rate::Rate
end

function crossover!(parents::NTuple{2, T},
                    strategy::ArithmeticCrossover{T, 2, 2},
                    generation::Integer) where {T}
    if rand() < strategy.rate(generation)
        mixing = rand()
        return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
                (1 - mixing) .* parents[2] .+ mixing .* parents[1])
    else
         return parents
    end
end


struct UniformCrossover{T<:AbstractVector, K, P} <: CrossoverStrategy{T, K, P}
    p::Rate
    
    UniformCrossover{T, K, P}(p::Rate = ConstantRate(0.5)) where {T, K, P} = new{T, K, P}(p)
end

function crossover!(parents::NTuple{2, T},
                    strategy::UniformCrossover{T, 2, 1},
                    generation::Integer) where {T}
    crossover_points = rand(length(parents[1])) .≤ strategy.p(generation)
    (map(ifelse, crossover_points, parents...),)
end

function crossover!(parents::NTuple{2, T},
                    strategy::UniformCrossover{T, 2, 2},
                    generation::Integer) where {T}
    crossover_points = rand(length(parents[1])) .≤ strategy.p(generation)
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


