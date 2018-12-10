export select,
    SelectionStrategy,
    SelectionResult,
    setup!

export PairWithBestSelection


abstract type SelectionStrategy{P, K} end

const SelectionResult{T, K} = NTuple{K, Individual{T}}


setup!(strategy::SelectionStrategy, model::AbstractEvolutionaryModel) = strategy


"""
    selection(population, strategy[, generation]) -> selection

Select parts of population of a population to be used in breeding.  Should compare fitnesses using 
`isless`, if that is relevant.
"""
selection(population, strategy, generation) = selection(population, strategy)
# selection(model::AbstractEvolutionaryModel, generation) = selection(model.population, model.strategy, generation)



# struct TruncationSelection{μ} <: SelectionStrategy{AbstractVector{T}} end

# function selection(model::GAModel{T, <:AbstractVector{T},
#                                   F, TruncationSelection{T, μ}, Fc, Fm}) where {T, F, μ, Fc, Fm}
#     partialsortperm(model.population, 1:μ, by = assess!, rev = true)
# end

mutable struct PairWithBestSelection{T, P} <: SelectionStrategy{P, 2}
    model::AbstractEvolutionaryModel

    PairWithBestSelection{T, P}() where {T, P} = new{T, P}()
end

function setup!(strategy::PairWithBestSelection, model::AbstractEvolutionaryModel)
    strategy.model = model
    strategy
end

function selection(population::Population{T}, strat::PairWithBestSelection{T, P}) where {T, P}
    M = length(population) ÷ P
    # simple naive groupings that pair the best entitiy with every other
    fittest = findfittest(strat.model)
    Iterators.zip(Iterators.repeated(findfittest(strat.model), M),
                  Iterators.take(Sampling(population), M))
end
