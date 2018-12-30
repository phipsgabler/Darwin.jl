using Distributions: Dirichlet

export crossover,
    crossover!,
    CrossoverOperator,
    CrossoverResult,
    setup!

export ArithmeticCrossover,
    LiftedCrossover,
    NoCrossover,
    UniformCrossover


abstract type CrossoverOperator{T, K, P} end


setup!(operator::CrossoverOperator, model::AbstractEvolutionaryModel) = operator


"""
    LiftedCrossover{T, C}(args)

Lifts a `CrossoverOperator{I, K, P}` on `I`, to a new `CrossoverOperator{T, K, P}` on `T`, 
by storing `C(args)`.  The actual application/lifting needs to be defined manually.
"""
struct LiftedCrossover{T, C, I, K, P} <: CrossoverOperator{T, K, P}
    inner::C

    LiftedCrossover{T}(operator::C) where {T, I, K, P, C<:CrossoverOperator{I, K, P}} =
        new{T, C, I, K, P}(operator)
    LiftedCrossover{T, C}(args...) where {T, I, K, P, C<:CrossoverOperator{I, K, P}} =
        new{T, C, I, K, P}(C(args...))
end


"""
    crossover!(parents::Union{Family, NTuple}, operator, generation) -> children

Perform crossover between `parents`.  You only need to define the method for `parents` being an
`NTuple`; this method will be automatically lifted to `Family`s (i.e., tuples of `Individual`s).
"""
crossover!(parents::Family{T, K},
           operator::CrossoverOperator{T, K, P},
           generation::Int) where {T, K, P} =
    Individual.(crossover!(genome.(parents), operator, generation))



struct NoCrossover{T, N} <: CrossoverOperator{T, N, N} end

crossover!(parents::NTuple{N}, operator::NoCrossover{N}, generation::Integer) where {N} =
    parents


struct ArithmeticCrossover{T<:AbstractVector, K, P} <: CrossoverOperator{T, K, P}
    rate::Rate
end

function crossover!(parents::NTuple{2, T},
                    operator::ArithmeticCrossover{T, 2, 2},
                    generation::Integer) where {T}
    if rand() < operator.rate(generation)
        mixing = rand()
        return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
                (1 - mixing) .* parents[2] .+ mixing .* parents[1])
    else
         return parents
    end
end


struct UniformCrossover{T<:AbstractVector, K, P} <: CrossoverOperator{T, K, P}
    p::Rate
    
    UniformCrossover{T, K, P}(p::Rate = ConstantRate(0.5)) where {T, K, P} = new{T, K, P}(p)
end

function crossover!(parents::NTuple{2, T},
                    operator::UniformCrossover{T, 2, 1},
                    generation::Integer) where {T}
    crossover_points = rand(length(parents[1])) .≤ operator.p(generation)
    (map(ifelse, crossover_points, parents...),)
end

function crossover!(parents::NTuple{2, T},
                    operator::UniformCrossover{T, 2, 2},
                    generation::Integer) where {T}
    crossover_points = rand(length(parents[1])) .≤ operator.p(generation)
    map(ifelse, crossover_points, parents...), map(ifelse, .~crossover_points, parents...)
end



# function crossover!(parents::Family{T, 1}, operator::ArithmeticCrossover{T, 1}) where {T}
#     if rand() < operator.rate
#         mixing = rand()
#         return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
#                 (1 - mixing) .* parents[2] .+ mixing .* parents[1])
#     else
#          return parents
#     end
# end

## TODO: move D to operator struct, use something different than "N random parent permutations"?
# function crossover(parents::NTuple{N, <:AbstractArray{T}}, operator::ArithmeticCrossover{T, N}) where {T}
#     D = Dirichlet(N, 1)
    
#     if rand() < operator.rate
#         mixing = rand(D)
#         return ntuple(sum(parents[randperm(N)] .* mixing), N)
        
#         return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
#                 (1 - mixing) .* parents[2] .+ mixing .* parents[1])
#     else
#          return parents
#     end
# end


# struct UniformCrossover{T} <: CrossoverOperator{NTuple{2, <:AbstractArray{T}}}
#     p::Float64
#     _dist::Bernoulli
# end

# function crossover!((p₁, p₂)::NTuple{2, <:AbstractArray{T}}, operator::UniformCrossover{T}) where {T}
#     @assert length(p₁) == length(p₂)
#     l = length(p₁)
    
#     crossover_points = rand(Uniform(operator.p), l)
#     p₁[crossover_points], p₂[crossover_points] = p₂[crossover_points], p₁[crossover_points]

#     p₁, p₂
# end


