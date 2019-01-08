import LearningStrategies
const L = LearningStrategies

export GAStrategy


mutable struct GAStrategy{G, P, K,
                          Fs<:SelectionOperator{>:G, P, K},
                          Fc<:CrossoverOperator{>:G, K, P},
                          Fm<:MutationOperator{>:G}} <: L.LearningStrategy
    selection::Fs
    crossover::Fc
    mutation::Fm
    cache::Population{G}

    function GAStrategy(selection::SelectionOperator{U, P, K},
                        crossover::CrossoverOperator{V, K, P},
                        mutation::MutationOperator{W}) where {U, V, W, P, K}
        G = typejoin(U, V, W)
        S, C, M = typeof(selection), typeof(crossover), typeof(mutation)
        new{G, P, K, S, C, M}(selection, crossover, mutation, Population{G}())
    end

    function GAStrategy{G}(selection::SelectionOperator{>:G, P, K},
                           crossover::CrossoverOperator{>:G, K, P},
                           mutation::MutationOperator{>:G}) where {G, P, K}
        S, C, M = typeof(selection), typeof(crossover), typeof(mutation)
        new{G, P, K, S, C, M}(selection, crossover, mutation, Population{G}())
    end
end


function L.setup!(strategy::GAStrategy{G}, model::AbstractPopulationModel{G}) where G
    # setup!(model.fitness, model)
    setup!(strategy.selection, model)
    setup!(strategy.mutation, model)
    setup!(strategy.crossover, model)
    assessfitness!(model)
end


function L.update!(model::AbstractPopulationModel{G}, strategy::GAStrategy{G, P, K},
                   generation, _item) where {G, P, K}
    preparecache!(strategy.cache, length(model.population))

    # TODO: log timing information
    for parents::Family{G, K} in selection(model.population, strategy.selection, generation)
        for child in crossover!(copy.(parents), strategy.crossover, generation)::Family{G, P}
            push!(strategy.cache, mutate!(child, strategy.mutation, generation))
        end
    end

    # swap parents and children -- saves reallocations
    model.population, strategy.cache = strategy.cache, model.population
    assessfitness!(model)

    model
end


function L.finished(verbose_strategy::L.Verbose{<:GAStrategy}, model::AbstractPopulationModel,
                    data, generation)
    done = L.finished(verbose_strategy.strategy, model, data, generation)
    done && @info "Evolved $generation generations, final population size $(length(model.population))"
    done
end
