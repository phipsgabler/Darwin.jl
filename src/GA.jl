import LearningStrategies
const L = LearningStrategies

export GAEvolver, GAModel

mutable struct GAModel{T,
                       Fs<:SelectionStrategy,
                       Fc<:CrossoverStrategy,
                       Fm<:MutationStrategy} <: AbstractEvolutionaryModel
    population::Vector{T}
    selectionstrategy::Fs
    crossoverstrategy::Fc
    mutationstrategy::Fm
end

populationtype(::Type{GAModel{T}}) where {T} = Vector{T}
genotype(::Type{GAModel{T}}) where {T} = T


mutable struct GAEvolver{T} <: L.LearningStrategy
    generations::Int
    childrencache::Vector{T}

    GAEvolver{T}(generations) where {T} = new{T}(generations)
end


function L.setup!(evolver::GAEvolver{T}, model::GAModel{T}) where T
    evolver.childrencache = similar(model.population)
    setup!(model.selectionstrategy, model)
    setup!(model.mutationstrategy, model)
    setup!(model.crossoverstrategy, model)
end


function L.update!(model::GAModel{T}, evolver::GAEvolver{T}, i, _item) where T
    parents = model.population
    parent_selection = selection(parents, model.selectionstrategy, i)
    children = evolver.childrencache

    # val, ti = @timeinfo breed!(evolver.model, parents, selections, children)

    breed!(model, parents, parent_selection, children, i)

    # swap parents and children -- saves reallocations
    model.population, evolver.childrencache = evolver.childrencache, model.population

    model
end


function breed!(model::GAModel, parents, selection, children, generation)
    crossover!(children, parents, selection, model.crossoverstrategy, generation)
    mutate!(children, model.mutationstrategy, generation)
end


L.finished(evolver::GAEvolver, model::GAModel, i) = i ≥ evolver.generations

function L.finished(verbose_evolver::L.Verbose{<:GAEvolver}, model::GAModel, data, i)
    done = L.finished(verbose_evolver.strategy, model, data, i)
    done && @info "Evolved $i generations in some time, final population size $(length(model.population))"
    done
end
