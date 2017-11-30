struct GeneticModel{P<:AbstractVector, Fs, Fc, Fm} <: AbstractEvolutionaryModel
    initial_population::P
    selection::Fs
    crossover::Fc
    mutate!::Fm
    generations::Int
end

populationtype{P, Fs, Fc, Fm}(::GeneticModel{P, Fs, Fc, Fm}) = P
genetype{P, Fs, Fc, Fm}(::GeneticModel{P, Fs, Fc, Fm}) = eltype(P)

function evolve(model::GeneticModel)
    N = length(model.initial_population)
    @assert iseven(N)
    
    populations = Vector{populationtype(model)}(model.generations)
    populations[1] = model.initial_population
    
    for g = 2:model.generations
        children = similar(model.initial_population)
        parents = populations[g - 1]
        
        for n = 1:2:N-1
            p1, p2 = model.selection(parents)
            c1, c2 = model.crossover(parents[p1], parents[p2])

            model.mutate!(c1)
            model.mutate!(c2)

            children[n], children[n + 1] = c1, c2
        end

        populations[g] = children
    end

    populations[end]
end
