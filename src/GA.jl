import LearningStrategies
const L = LearningStrategies

export GAStrategy, GAModel

mutable struct GAModel{T, F<:AbstractFitness{>:T}} <: AbstractEvolutionaryModel
    population::Population{T}
    fitness::F
    fittest::Individual{T}
    
    GAModel(population::Population{T}, fitness) where {T} =
        new{T, typeof(fitness)}(population, fitness)
end


findfittest!(model::GAModel) = model.fittest = maximumby(i -> assess!(i, model.fitness),
                                                         model.population)
findfittest(model::GAModel) = model.fittest


mutable struct GAStrategy{T, P, K,
                          Fs<:SelectionOperator{>:T, P, K},
                          Fc<:CrossoverOperator{>:T, K, P},
                          Fm<:MutationOperator{>:T}} <: L.LearningStrategy
    selection::Fs
    crossover::Fc
    mutation::Fm
    cache::Population{T}

    function GAStrategy(selection::SelectionOperator{U, P, K},
                        crossover::CrossoverOperator{V, K, P},
                        mutation::MutationOperator{W}) where {U, V, W, P, K}
        T = typejoin(U, V, W)
        S, C, M = typeof(selection), typeof(crossover), typeof(mutation)
        new{T, P, K, S, C, M}(selection, crossover, mutation, Population{T}())
    end

    function GAStrategy{T}(selection::SelectionOperator{>:T, P, K},
                           crossover::CrossoverOperator{>:T, K, P},
                           mutation::MutationOperator{>:T}) where {T, P, K}
        S, C, M = typeof(selection), typeof(crossover), typeof(mutation)
        new{T, P, K, S, C, M}(selection, crossover, mutation, Population{T}())
    end
end

preparecache!(strategy::GAStrategy, n) = sizehint!(empty!(strategy.cache), n)


function L.setup!(strategy::GAStrategy{T}, model::GAModel{T}) where T
    # setup!(model.fitness, model)
    setup!(strategy.selection, model)
    setup!(strategy.mutation, model)
    setup!(strategy.crossover, model)
    findfittest!(model)
end


function L.update!(model::GAModel{T}, strategy::GAStrategy{T, P, K}, i, _item) where {T, P, K}
    preparecache!(strategy, length(model.population))

    # TODO: log timing information
    for parents::Family{T, K} in selection(model.population, strategy.selection, i)
        for child in crossover!(copy.(parents), strategy.crossover, i)::Family{T, P}
            push!(strategy.cache, mutate!(child, strategy.mutation, i))
        end
    end

    # swap parents and children -- saves reallocations
    model.population, strategy.cache = strategy.cache, model.population

    findfittest!(model)

    model
end


function L.finished(verbose_strategy::L.Verbose{<:GAStrategy}, model::GAModel, data, i)
    done = L.finished(verbose_strategy.strategy, model, data, i)
    done && @info "Evolved $i generations, final population size $(length(model.population))"
    done
end
