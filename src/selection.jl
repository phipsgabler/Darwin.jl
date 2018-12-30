using Distributions
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


abstract type SelectionOperator{T, P, K} end

setup!(operator::SelectionOperator, model::AbstractEvolutionaryModel) = operator


"""
    selection(population, operator[, generation]) -> selection

Select parts of population of a population to be used in breeding.  Should compare fitnesses using 
`isless`, if that is relevant.
"""
selection(population::Population{T}, operator::SelectionOperator{T, P, K},
          generation::Integer) where {T, P, K} =
    selection(population, operator)


# struct TruncationSelection{T, P} <: SelectionOperator{T, P, 1} end

# function selection(population::Population{T}, operator::TruncationSelection{T, P}) where {T, P}
#     μ = length(population) ÷ P
#     partialsort(model.population, 1:μ, by = fitness, rev = true)
# end


mutable struct PairWithBestSelection{T, P} <: SelectionOperator{T, P, 2}
    model::AbstractEvolutionaryModel

    PairWithBestSelection{T, P}() where {T, P} = new{T, P}()
end

function setup!(operator::PairWithBestSelection, model::AbstractEvolutionaryModel)
    operator.model = model
    operator
end

function selection(population::Population{T}, operator::PairWithBestSelection{T, P}) where {T, P}
    # simple naive groupings that pair the best entitiy with every other
    fittest = findfittest(operator.model)
    PairWithBestSelectionIterator{P}(population, fittest)
end

struct PairWithBestSelectionIterator{P, M, T}
    population::Population{T}
    fittest::Individual{T}

    function PairWithBestSelectionIterator{P}(population::Population{T},
                                              fittest::Individual{T}) where {P, T}
        M = length(population) ÷ P
        new{P, M, T}(population, fittest)
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



struct FitnessProportionateSelection{T, P, K, F} <: SelectionOperator{T, P, K}
    transform::F
    temperature::Rate
end

function selection(population::Population{T},
                   operator::FitnessProportionateSelection{T, P, K},
                   generation::Integer) where {T, P, K}
    function transform(f)
        operator.transform(f, operator.temperature(generation))
    end
    FitnessProportionateSelectionIterator{P, K}(population, transform)
end

struct FitnessProportionateSelectionIterator{M, P, K, T}
    population::Population{T}
    transform

    function FitnessProportionateSelectionIterator{P, K}(population::Population{T},
                                                         transform) where {P, K, T}
        M = length(population) ÷ P
        new{M, P, K, T}(population, transform)
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

const SoftmaxSelection{T, P, K} = FitnessProportionateSelection{T, P, K, typeof(softmax)}
(::Type{SoftmaxSelection{T, P, K}})(rate::Rate = ConstantRate(1.0)) where {T, P, K} =
    SoftmaxSelection{T, P, K}(softmax, rate)

const L1Selection{T, P, K} = FitnessProportionateSelection{T, P, K, typeof(l1normalize)}
(::Type{L1Selection{T, P, K}})(rate::Rate = ConstantRate(1.0)) where {T, P, K} =
    L1Selection{T, P, K}(l1normalize, rate)




struct TournamentSelection{T, S, P, K} <: SelectionOperator{T, P, K} end

selection(population::Population{T}, operator::TournamentSelection{T, S, P, K}) where {T, S, P, K} = 
    TournamentSelectionIterator{S, P, K}(population)

struct TournamentSelectionIterator{M, S, P, K, T}
    population::Population{T}

    function TournamentSelectionIterator{S, P, K}(population::Population{T}) where {S, P, K, T}
        M = length(population) ÷ P
        new{M, S, P, K, T}(population)
    end
end

function iterate(itr::TournamentSelectionIterator{M, S, P, K}, state = 0) where {M, S, P, K}
    if state ≥ M
        return nothing
    else
        ntuple(i -> maximumby(fitness, randview(itr.population, S)), Val{K}()), state + 1
    end
end

