import LearningStrategies
const L = LearningStrategies

export GAEvolver, GAModel

mutable struct GAModel{T, P, K,
                       F<:AbstractFitness{>:T},
                       Fs<:SelectionStrategy{P, K},
                       Fc<:CrossoverStrategy{P, K},
                       Fm<:MutationStrategy{T}} <: AbstractEvolutionaryModel
    population::Population{T}
    fittest::Union{Individual{T}, Nothing}
    fitness::F
    selectionstrategy::Fs
    crossoverstrategy::Fc
    mutationstrategy::Fm
end

GAModel(p, f, s, c, m) = GAModel(p, nothing, f, s, c, m)

findfittest!(model::GAModel) = model.fittest = maximumby(i -> assess!(i, model.fitness),
                                                         model.population)
findfittest(model::GAModel) = model.fittest


mutable struct GAEvolver{T} <: L.LearningStrategy
    cache::Population{T}

    GAEvolver{T}() where {T} = new{T}(Population{T}())
end

preparecache!(evolver::GAEvolver, n) = sizehint!(empty!(evolver.cache), n)


function L.setup!(evolver::GAEvolver{T}, model::GAModel{T}) where T
    # setup!(model.fitness, model)
    setup!(model.selectionstrategy, model)
    setup!(model.mutationstrategy, model)
    setup!(model.crossoverstrategy, model)
    findfittest!(model)
end


function L.update!(model::GAModel{T, P, K}, evolver::GAEvolver{T}, i, _item) where {T, P, K}
    preparecache!(evolver, length(model.population))

    # TODO: log timing information
    for parents::Family{T, K} in selection(model.population, model.selectionstrategy, i)
        for child in crossover!(copy.(parents), model.crossoverstrategy, i)::Family{T, P}
            push!(evolver.cache, mutate!(child, model.mutationstrategy, i))
        end
    end

    # swap parents and children -- saves reallocations
    model.population, evolver.cache = evolver.cache, model.population

    findfittest!(model)

    model
end


function L.finished(verbose_evolver::L.Verbose{<:GAEvolver}, model::GAModel, data, i)
    done = L.finished(verbose_evolver.strategy, model, data, i)
    done && @info "Evolved $i generations, final population size $(length(model.population))"
    done
end
