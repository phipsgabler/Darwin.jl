using Distributions
using Random: shuffle
import Base: iterate

export select,
    SelectionOperator,
    SelectionResult,
    setup!

export FitnessProportionateSelection,
    L1Selection,
    PairWithBestSelection,
    SoftmaxSelection,
    TournamentSelection


abstract type SelectionOperator{G, P, K} end

setup!(operator::SelectionOperator{G}, model::AbstractPopulationModel{G}) where {G} = operator


"""
    selection(population, operator[, generation]) -> selection

Select parts of population of a population to be used in breeding.  Should compare fitnesses using 
`isless`, if that is relevant.

The result is an iterable of `Family{G, K}`.
"""
selection(population::Population{G}, operator::SelectionOperator{G, P, K},
          generation::Integer) where {G, P, K} =
    selection(population, operator)




mutable struct PairWithBestSelection{G, P} <: SelectionOperator{G, P, 2}
    model::AbstractPopulationModel

    PairWithBestSelection{G, P}() where {G, P} = new{G, P}()
end

function setup!(operator::PairWithBestSelection, model::AbstractPopulationModel)
    operator.model = model
    operator
end

function selection(population::Population{G}, operator::PairWithBestSelection{G, P}) where {G, P}
    # simple naive groupings that pair the best entitiy with every other
    PairWithBestSelectionIterator{P}(population, operator.model.fittest)
end

struct PairWithBestSelectionIterator{P, M, G}
    population::Population{G}
    fittest::Individual{G}

    function PairWithBestSelectionIterator{P}(population::Population{G},
                                              fittest::Individual{G}) where {P, G}
        M = length(population) ÷ P
        new{P, M, G}(population, fittest)
    end
end

function iterate(itr::PairWithBestSelectionIterator{P, M}, state = 0) where {P, M}
    if state ≥ M
        return nothing
    else
        (itr.fittest, rand(itr.population)), state + 1
    end
end

# function iterate(itr::PairWithBestSelectionIterator{1})
#     next = iterate(itr.population)
#     if next == nothing
#         return nothing
#     else
#         individual, state = next
#         (itr.fittest, individual), state
#     end
# end

# function iterate(itr::PairWithBestSelectionIterator{1}, state)
#     next = iterate(itr.population, state)
#     if next == nothing
#         return nothing
#     else
#         individual, state = next
#         (itr.fittest, individual), state
#     end
# end



struct FitnessProportionateSelection{G, P, K, F} <: SelectionOperator{G, P, K}
    transform::F
    temperature::Rate
end

function selection(population::Population{G},
                   operator::FitnessProportionateSelection{G, P, K},
                   generation::Integer) where {G, P, K}
    function transform(f)
        operator.transform(f, operator.temperature(generation))
    end
    FitnessProportionateSelectionIterator{P, K}(population, transform)
end

struct FitnessProportionateSelectionIterator{M, P, K, G}
    population::Population{G}
    transform

    function FitnessProportionateSelectionIterator{P, K}(population::Population{G},
                                                         transform) where {P, K, G}
        M = length(population) ÷ P
        new{M, P, K, G}(population, transform)
    end
end

function iterate(itr::FitnessProportionateSelectionIterator{M, P, K}, state = 0) where {M, P, K}
    if state ≥ M
        return nothing
    else
        dist = Categorical(itr.transform(fitness.(itr.population)))
        indices = eachindex(itr.population)
        ntuple(i -> itr.population[rand(dist)], Val{K}()), state + 1
    end
end

function l1normalize(f, _)
    y = f .- minimum(f)
    s = sum(y)
    s == 0 ? fill(1/length(f), size(f)) : y ./ s
end

function softmax(f, θ = 1)
    y = float(f) ./ θ
    y .= exp.(y .- maximum(y))
    y ./ sum(y)
end

const SoftmaxSelection{G, P, K} = FitnessProportionateSelection{G, P, K, typeof(softmax)}
(::Type{SoftmaxSelection{G, P, K}})(rate::Rate = ConstantRate(1.0)) where {G, P, K} =
    SoftmaxSelection{G, P, K}(softmax, rate)

const L1Selection{G, P, K} = FitnessProportionateSelection{G, P, K, typeof(l1normalize)}
(::Type{L1Selection{G, P, K}})() where {G, P, K} =
    L1Selection{G, P, K}(l1normalize, ConstantRate(1.0))




struct TournamentSelection{G, S, P, K} <: SelectionOperator{G, P, K} end

selection(population::Population{G}, operator::TournamentSelection{G, S, P, K}) where {G, S, P, K} = 
    TournamentSelectionIterator{S, P, K}(population)

struct TournamentSelectionIterator{M, S, P, K, G}
    population::Population{G}

    function TournamentSelectionIterator{S, P, K}(population::Population{G}) where {S, P, K, G}
        M = length(population) ÷ P
        new{M, S, P, K, G}(population)
    end
end

function iterate(itr::TournamentSelectionIterator{M, S, P, K}, state = 0) where {M, S, P, K}
    if state ≥ M
        return nothing
    else
        ntuple(i -> maximumby(fitness, randview(itr.population, S)), Val{K}()), state + 1
    end
end




struct TruncationSelection{G, P} <: SelectionOperator{G, P, 1} end

function selection(population::Population{G}, operator::TruncationSelection{G, P}) where {G, P}
    μ = length(population) ÷ P
    partialsortperm(model.population, 1:μ, by = fitness, rev = true)
end




# struct SUSSelection{G, P, K} <: SelectionOperator{G, P, K} end

# function selection(population::Population{G}, operator::SUSSelection{S, P, K})
#     fitness_cdf = fitness.(population)
#     fitness_cdf .-= minimum(fitness_cdf)
#     sum(fitness_cdf) == 0 && fitness_cdf .= 1
#     cumsum!(fitness_cdf, fitness_cdf)
#     range = fitness_cdf[end] / length(fitness_cdf)

#     SUSSelectionIterator{P, K}(population, shuffle(eachindex(population)), fitness_cdf, range)
# end

# struct SUSSelectionIterator{M, P, K, G}
#     population::Population{G}
#     permutation::Vector{Int}
#     fitness_cdf::Vector{Float64}
#     range::Float64

#     function SUSSelectionIterator{P, K}(population::Population{G}, permutation::Vector{Int}
#                                         fitness_cdf::Vector{Float64}, position::Float64) where {P, K, G}
#         M = length(population) ÷ P
#         new{M, P, K, G}(population, permutation, fitness_cdf, position)
#     end
# end

# function iterate(itr::SUSSelectionIterator{M, P, K}, state = rand() * itr.range) where {P, K}
#     if state > f[end]
#         return nothing
#     else
#         index = findfirst(f -> f ≥ itr.position, view(itr.fitness_cdf, itr.permutation))
#         return view(itr.population, itr.permutation)[index], state + itr.range
#     end
# end
