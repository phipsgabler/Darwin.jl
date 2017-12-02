using Base.Iterators: take

struct GAModel{P<:AbstractVector, Fs, Fc, Fm} <: AbstractEvolutionaryModel
    initial_population::P
    selection::Fs
    crossover::Fc
    mutate!::Fm
    matingfactor::Int
end

GAModel(ip::P, sel::Fs, co::Fc, mut::Fm, mf = 2) where {P, Fs, Fc, Fm} =
    GAModel{P, Fs, Fc, Fm}(ip, sel, co, mut, mf)

populationtype{P, Fs, Fc, Fm}(::GAModel{P, Fs, Fc, Fm}) = P
genetype{P, Fs, Fc, Fm}(::GAModel{P, Fs, Fc, Fm}) = eltype(P)


struct GASolution{P<:AbstractVector} <: AbstractEvolutionarySolution
    population::P
    timeinfo::TimeInfo
end

mutable struct GAEvolver{P<:AbstractVector, Fs, Fc, Fm}
    model::GAModel{P, Fs, Fc, Fm}
    generation::Int
    solution::GASolution{P}
    populationsize::Int
    matingfactor::Int
    cumtime::Float64
end

GAEvolver(m::GAModel{P, Fs, Fc, Fm}, g, s::GASolution{P}, ps, mf, ct = 0.0) where {P, Fs, Fc, Fm} =
    GAEvolver{P, Fs, Fc, Fm}(m, g, s, ps, mf, ct)

function evolve(model::GAModel, generations::Int;
                verbose = false,
                callback = evaluator -> nothing)
    @assert generations >= 1
    
    evolver = init(model)
    verbose && println("Starting with population of size ", evolver.populationsize,
                       ", mating factor ", evolver.matingfactor, "...")

    for step in take(evolver, generations - 1)
        callback(step)
    end

    verbose && println("Evolved ", evolver.generation, " generations in ",
                       evolver.cumtime, " seconds total.")
    
    return evolver.solution
end



function breed!(model, N, M, parents, children)
    for i = 1:M:N-M+1
        children_section = i:i+M-1
        selected = model.selection(parents)
        children[children_section] .= model.crossover(parents[selected])
        model.mutate!.(children[children_section])
    end
end

function init(model::GAModel)
    N = length(model.initial_population)
    M = model.matingfactor
    @assert N % M == 0
    
    GAEvolver(model, 1, GASolution(model.initial_population, notime), N, M)
end

function evolvestep!(evolver::GAEvolver)
    N = evolver.populationsize
    M = evolver.matingfactor
    parents = evolver.solution.population
    children = similar(parents)

    _, t, bytes, gctime, memallocs = @timed breed!(evolver.model, N, M, parents, children)
    
    evolver.solution = GASolution(children, TimeInfo(t, bytes, gctime, memallocs))
    evolver.generation += 1
    evolver.cumtime += t

    evolver
end


# iterator interface for evolver; for implementation, see
# https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl/blob/master/src/iterator_interface.jl

import Base: start, next, done, eltype, iteratorsize, iteratoreltype

Base.start(evolver::GAEvolver) = 0
Base.next(evolver::GAEvolver, state) = begin
    state += 1
    evolvestep!(evolver)
    evolver, state
end
Base.done(::GAEvolver, state) = false
Base.iteratorsize{T<:GAEvolver}(::Type{T}) = Base.IsInfinite()
Base.iteratoreltype{T<:GAEvolver}(::Type{T}) = Base.HasEltype()
Base.eltype{T<:GAEvolver}(::T) = T
