using Base.Iterators: take

struct GAModel{P<:AbstractVector, Fs, Fc, Fm} <: AbstractEvolutionaryModel
    initial_population::P
    selections::Fs
    crossover::Fc
    mutate!::Fm
end

GAModel(ip::P, sel::Fs, co::Fc, mut::Fm) where {P, Fs, Fc, Fm} =
    GAModel{P, Fs, Fc, Fm}(ip, sel, co, mut)

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
    cumtime::Float64
end

GAEvolver(m::GAModel{P, Fs, Fc, Fm}, g, s::GASolution{P}) where {P, Fs, Fc, Fm} =
    GAEvolver{P, Fs, Fc, Fm}(m, g, s, length(s.population), 0.0)

function evolve(model::GAModel, generations::Int;
                verbose = false,
                callback = evaluator -> nothing)
    @assert generations >= 1

    evolver = init(model)
    verbose && println("Starting with population of size ", evolver.populationsize)

    for step in take(evolver, generations - 1)
        callback(step)
    end

    verbose && println("Evolved ", evolver.generation, " generations in ",
                evolver.cumtime, " seconds total, ",
                "final population size ", length(evolver.solution.population))

    return evolver.solution
end

function init(model::GAModel)
    GAEvolver(model, 1, GASolution(model.initial_population, notime))
end


function evolvestep!(evolver::GAEvolver)
    parents = evolver.solution.population
    children = similar(parents, 0)
    sizehint!(children, length(parents))

    _, t, bytes, gctime, memallocs = @timed breed!(evolver.model, parents, children)

    evolver.solution = GASolution(children, TimeInfo(t, bytes, gctime, memallocs))
    evolver.generation += 1
    evolver.cumtime += t

    evolver
end


function breed!(model, parents, children)
    for selected in model.selections(parents)
        offspring = model.crossover(parents[collect(Int, selected)])
        model.mutate!.(offspring)
        append!(children, collect(eltype(children), offspring))
    end
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
