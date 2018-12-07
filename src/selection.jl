export select,
    SelectionStrategy,
    setup!

export PairWithBestSelection


abstract type SelectionStrategy{S} end

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

mutable struct PairWithBestSelection{T} <: SelectionStrategy{NTuple{2, Individual{T}}}
    model::AbstractEvolutionaryModel

    PairWithBestSelection{T}() where T = new{T}()
end

function setup!(strategy::PairWithBestSelection{T}, model::AbstractEvolutionaryModel) where {T} 
    strategy.model = model
    strategy
end

function selection(population::Population{T}, strat::PairWithBestSelection{T}) where {T}
    # simple naive groupings that pair the best entitiy with every other
    Iterators.zip(Iterators.repeated(findfittest(strat.model), length(population)), population)
end
