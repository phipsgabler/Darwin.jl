import LearningStrategies
const L = LearningStrategies

export GAStrategy, GAModel

mutable struct GAModel{T, P, K,
                       F<:AbstractFitness{>:T},
                       Fs<:SelectionOperator{>:T, P, K},
                       Fc<:CrossoverOperator{>:T, K, P},
                       Fm<:MutationOperator{>:T}} <: AbstractEvolutionaryModel
    population::Population{T}
    fittest::Union{Individual{T}, Nothing}
    fitness::F
    selection::Fs
    crossover::Fc
    mutation::Fm
end

GAModel(p, f, s, c, m) = GAModel(p, nothing, f, s, c, m)

findfittest!(model::GAModel) = model.fittest = maximumby(i -> assess!(i, model.fitness),
                                                         model.population)
findfittest(model::GAModel) = model.fittest


mutable struct GAStrategy{T} <: L.LearningStrategy
    cache::Population{T}

    GAStrategy{T}() where {T} = new{T}(Population{T}())
end

preparecache!(evolver::GAStrategy, n) = sizehint!(empty!(evolver.cache), n)


function L.setup!(evolver::GAStrategy{T}, model::GAModel{T}) where T
    # setup!(model.fitness, model)
    setup!(model.selection, model)
    setup!(model.mutation, model)
    setup!(model.crossover, model)
    findfittest!(model)
end


function L.update!(model::GAModel{T, P, K}, evolver::GAStrategy{T}, i, _item) where {T, P, K}
    preparecache!(evolver, length(model.population))

    # TODO: log timing information
    for parents::Family{T, K} in selection(model.population, model.selection, i)
        for child in crossover!(copy.(parents), model.crossover, i)::Family{T, P}
            push!(evolver.cache, mutate!(child, model.mutation, i))
        end
    end

    # swap parents and children -- saves reallocations
    model.population, evolver.cache = evolver.cache, model.population

    findfittest!(model)

    model
end


function L.finished(verbose_evolver::L.Verbose{<:GAStrategy}, model::GAModel, data, i)
    done = L.finished(verbose_evolver.strategy, model, data, i)
    done && @info "Evolved $i generations, final population size $(length(model.population))"
    done
end
