using Distributions
import Base: iterate

export select,
    SelectionStrategy,
    SelectionResult,
    setup!

export FitnessProportionate,
    L1Selection,
    PairWithBest,
    SoftmaxSelection,
    TournamentSelection


abstract type SelectionStrategy{T, P, K} end

setup!(strategy::SelectionStrategy, model::AbstractEvolutionaryModel) = strategy


"""
    selection(population, strategy[, generation]) -> selection

Select parts of population of a population to be used in breeding.  Should compare fitnesses using 
`isless`, if that is relevant.
"""
selection(population::Population{T}, strategy::SelectionStrategy{T, P, K},
          generation::Integer) where {T, P, K} =
    selection(population, strategy)


# struct TruncationSelection{μ} <: SelectionStrategy{AbstractVector{T}} end

# function selection(model::GAModel{T, <:AbstractVector{T},
#                                   F, TruncationSelection{T, μ}, Fc, Fm}) where {T, F, μ, Fc, Fm}
#     partialsortperm(model.population, 1:μ, by = assess!, rev = true)
# end


mutable struct PairWithBest{T, P} <: SelectionStrategy{T, P, 2}
    model::AbstractEvolutionaryModel

    PairWithBest{T, P}() where {T, P} = new{T, P}()
end

function setup!(strategy::PairWithBest, model::AbstractEvolutionaryModel)
    strategy.model = model
    strategy
end

function selection(population::Population{T}, strategy::PairWithBest{T, P}) where {T, P}
    M = length(population) ÷ P
    # simple naive groupings that pair the best entitiy with every other
    fittest = findfittest(strategy.model)
    Iterators.zip(Iterators.repeated(fittest, M),
                  repeatfunc(rand, M, population))
end


struct FitnessProportionate{T, P, K, F} <: SelectionStrategy{T, P, K}
    transform::F
    temperature::Rate
end

@generated function selection(population::Population{T},
                              strategy::FitnessProportionate{T, P, K},
                              generation::Integer) where {T, P, K}
    rndix = fill(:(view(population, indices[rand(dist, M)])), K)
    quote
        M = length(population) ÷ P
        dist = Categorical(strategy.transform(fitness.(population),
                                              strategy.temperature(generation)))
        indices = eachindex(population)
        Iterators.zip($(rndix...))
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

const SoftmaxSelection{T, P, K} = FitnessProportionate{T, P, K, typeof(softmax)}
(::Type{SoftmaxSelection{T, P, K}})(rate::Rate = ConstantRate(1.0)) where {T, P, K} =
    SoftmaxSelection{T, P, K}(softmax, rate)

const L1Selection{T, P, K} = FitnessProportionate{T, P, K, typeof(l1normalize)}
(::Type{L1Selection{T, P, K}})(rate::Rate = ConstantRate(1.0)) where {T, P, K} =
    L1Selection{T, P, K}(l1normalize, rate)


struct TournamentSelection{T, S, P, K} <: SelectionStrategy{T, P, K} end

struct TournamentSelectionIterator{S, P, K, T}
    M::Int
    population::Population{T}

    TournamentSelectionIterator{S, P, K}(population::Population{T}) where {S, P, K, T} =
        new{S, P, K, T}(length(population) ÷ P, population)
end

function iterate(itr::TournamentSelectionIterator{S, P, K}, state = 0) where {S, P, K}
    if state ≥ itr.M
        return nothing
    else
        ntuple(i -> maximumby(fitness, randview(itr.population, S)), K), state + 1
    end
end

selection(population::Population{T},
          strategy::TournamentSelection{T, S, P, K}) where {T, S, P, K} = 
    TournamentSelectionIterator{S, P, K}(population)
