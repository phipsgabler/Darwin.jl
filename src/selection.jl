export select,
    SelectionStrategy,
    SelectionResult,
    setup!

export PairWithBestSelection


abstract type SelectionStrategy{T, P, K} end

setup!(strategy::SelectionStrategy, model::AbstractEvolutionaryModel) = strategy


"""
    selection(population, strategy[, generation]) -> selection

Select parts of population of a population to be used in breeding.  Should compare fitnesses using 
`isless`, if that is relevant.
"""
selection(population::Population{T}, strategy::SelectionStrategy{T, P, K},
          generation::Int) where {T, P, K} =
    selection(population, strategy)


# struct TruncationSelection{μ} <: SelectionStrategy{AbstractVector{T}} end

# function selection(model::GAModel{T, <:AbstractVector{T},
#                                   F, TruncationSelection{T, μ}, Fc, Fm}) where {T, F, μ, Fc, Fm}
#     partialsortperm(model.population, 1:μ, by = assess!, rev = true)
# end


mutable struct PairWithBestSelection{T, P} <: SelectionStrategy{T, P, 2}
    model::AbstractEvolutionaryModel

    PairWithBestSelection{T, P}() where {T, P} = new{T, P}()
end

function setup!(strategy::PairWithBestSelection, model::AbstractEvolutionaryModel)
    strategy.model = model
    strategy
end

function selection(population::Population{T}, strat::PairWithBestSelection{T, P}) where {T, P}
    M = length(population) ÷ P
    fittest = findfittest(strat.model)
    # simple naive groupings that pair the best entitiy with every other
    fittest = findfittest(strat.model)
    Iterators.zip(Iterators.repeated(fittest, M),
                  repeatfunc(rand, M, population))
end
