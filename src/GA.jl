struct GeneticModel{P<:AbstractVector, Fs, Fc, Fm} <: AbstractEvolutionaryModel
    initial_population::P
    selection::Fs
    crossover::Fc
    mutate!::Fm
    generations::Int
    matingfactor::Int
end

GeneticModel(ip, sel, co, mut!, g, mf = 2) = GeneticModel(ip, sel, co, mut!, g, mf)

populationtype{P, Fs, Fc, Fm}(::GeneticModel{P, Fs, Fc, Fm}) = P
genetype{P, Fs, Fc, Fm}(::GeneticModel{P, Fs, Fc, Fm}) = eltype(P)

function evolve(model::GeneticModel)
    N = length(model.initial_population)
    M = model.matingfactor
    @assert N % M == 0
    
    populations = Vector{populationtype(model)}(model.generations)
    populations[1] = model.initial_population
    
    for g = 2:model.generations
        children = similar(model.initial_population)
        parents = populations[g - 1]
        
        for n = 1:M:N-M+1
            selected = model.selection(parents)
            offspring = model.crossover(parents[selected])

            model.mutate!.(offspring)
            children[n:n+M-1] .= offspring
        end

        populations[g] = children
    end

    populations[end]
end
