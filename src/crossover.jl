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


abstract type CrossoverOperator{G, K, P} end


setup!(operator::CrossoverOperator{G}, model::AbstractPopulationModel{G}) where {G} = operator


"""
    LiftedCrossover{G, C}(args)

Lifts a `CrossoverOperator{I, K, P}` on `I`, to a new `CrossoverOperator{G, K, P}` on `G`, 
by storing `C(args)`.  The actual application/lifting needs to be defined manually.
"""
struct LiftedCrossover{G, C, I, K, P} <: CrossoverOperator{G, K, P}
    inner::C

    LiftedCrossover{G}(operator::C) where {G, I, K, P, C<:CrossoverOperator{I, K, P}} =
        new{G, C, I, K, P}(operator)
    LiftedCrossover{G, C}(args...) where {G, I, K, P, C<:CrossoverOperator{I, K, P}} =
        new{G, C, I, K, P}(C(args...))
end


"""
    crossover!(parents::Union{Family, NTuple}, operator, generation) -> children

Perform crossover between `parents`.  You only need to define the method for `parents` being an
`NTuple{K}`, returning an `NTuple{P}`.  The method will be automatically lifted to `Family`s (i.e.,
tuples of `Individual`s) of the same sizes.
"""
crossover!(parents::Family{G, K},
           operator::CrossoverOperator{G, K, P},
           generation::Int) where {G, K, P} =
    Individual.(crossover!(genome.(parents), operator, generation))



struct NoCrossover{N} <: CrossoverOperator{Any, N, N} end

crossover!(parents::NTuple{N}, operator::NoCrossover{N}, generation::Integer) where {N} =
    parents


struct ArithmeticCrossover{G<:AbstractVector, K, P} <: CrossoverOperator{G, K, P}
    rate::Rate
end

function crossover!(parents::NTuple{2, G},
                    operator::ArithmeticCrossover{G, 2, 2},
                    generation::Integer) where {G}
    if rand() < operator.rate(generation)
        mixing = rand()
        return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
                (1 - mixing) .* parents[2] .+ mixing .* parents[1])
    else
         return parents
    end
end


struct UniformCrossover{G<:AbstractVector, K, P} <: CrossoverOperator{G, K, P}
    p::Rate
    
    UniformCrossover{G, K, P}(p::Rate = ConstantRate(0.5)) where {G, K, P} = new{G, K, P}(p)
end

function crossover!(parents::NTuple{2, G},
                    operator::UniformCrossover{G, 2, 1},
                    generation::Integer) where {G}
    crossover_points = rand(length(parents[1])) .≤ operator.p(generation)
    (map(ifelse, crossover_points, parents...),)
end

function crossover!(parents::NTuple{2, G},
                    operator::UniformCrossover{G, 2, 2},
                    generation::Integer) where {G}
    crossover_points = rand(length(parents[1])) .≤ operator.p(generation)
    map(ifelse, crossover_points, parents...), map(ifelse, .~crossover_points, parents...)
end



# function crossover!(parents::Family{G, 1}, operator::ArithmeticCrossover{G, 1}) where {G}
#     if rand() < operator.rate
#         mixing = rand()
#         return ((1 - mixing) .* parents[1] .+ mixing .* parents[2],
#                 (1 - mixing) .* parents[2] .+ mixing .* parents[1])
#     else
#          return parents
#     end
# end

## TODO: move D to operator struct, use something different than "N random parent permutations"?
# function crossover(parents::NTuple{N, <:AbstractArray{G}}, operator::ArithmeticCrossover{G, N}) where {G}
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


# struct UniformCrossover{G} <: CrossoverOperator{NTuple{2, <:AbstractArray{G}}}
#     p::Float64
#     _dist::Bernoulli
# end

# function crossover!((p₁, p₂)::NTuple{2, <:AbstractArray{G}}, operator::UniformCrossover{G}) where {G}
#     @assert length(p₁) == length(p₂)
#     l = length(p₁)
    
#     crossover_points = rand(Uniform(operator.p), l)
#     p₁[crossover_points], p₂[crossover_points] = p₂[crossover_points], p₁[crossover_points]

#     p₁, p₂
# end


