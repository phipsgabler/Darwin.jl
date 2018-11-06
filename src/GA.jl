import LearningStrategies: LearningStrategy, Verbose,
    cleanup!, finished, hook, setup!, update!

export GAModel, genetype, populationtype
export evolve, init, evolvestep!

mutable struct GAModel{T,
                       Fs<:SelectionStrategy,
                       Fm<:MutationStrategy,
                       Fc<:CrossoverStrategy} <: AbstractEvolutionaryModel
    population::Vector{T}
    selectionstrategy::Fs
    mutationstrategy::Fm
    crossoverstrategy::Fc
end

populationtype(::Type{GAModel{T}}) where {T} = Vector{T}
genotype(::Type{GAModel{T}}) where {T} = T


mutable struct GAEvolver{T} <: LearningStrategy
    generations::Int
    childrencache::Vector{T}

    GAEvolver{T}(generations) where {T} = new{T}(generations)
end


function setup!(evolver::GAEvolver{T}, model::GAModel{T}) where T
    evolver.childrencache = similar(model.population)
    setup!(model.selectionstrategy)
    setup!(model.mutationstrategy)
    setup!(model.crossoverstrategy)
end


function update!(model::GAModel{T}, evolver::GAEvolver{T}, i, _item) where T
    parents = evolver.population
    selections = select(parents, model.selectionstrategy, i)
    children = evolver.childrencache

    # val, ti = @timeinfo breed!(evolver.model, parents, selections, children)

    breed!(model, parents, selections, children, i)

    # swap parents and children -- saves reallocations
    model.population, evolver.childrencache = evolver.childrencache, model.population

    model
end


function breed!(model::GAModel, parents, selections, children, generation)
    crossover!(children, view(parents, selections), model.crossoverstrategy, generation)
    mutate!(children, model.mutationstrategy, generation)
end


finished(evolver::GAEvolver, model::GAModel, i) = i ≥ evolver.generations

function finished(verbose_evolver::Verbose{<:GAEvolver}, model::GAModel, data, i)
    done = finished(verbose_evolver.strategy, model, data, i)
    done && @info ("Evolved ", i, " generations in ",
                   "some time, ", #evolver.cumtime, " seconds total, ",
                   "final population size ", length(model.population))
    done
end
