import Base: copy
import Random: AbstractRNG, rand, SamplerType

export Family,
    fitness,
    genome,
    Individual,
    ongenome,
    Population


mutable struct Individual{G}
    genome::G
    fitnessvalue::Union{FitnessValue, Nothing}
end

Individual(genome::G) where {G} = Individual{G}(genome, nothing)
Individual(genome::G, fitnessvalue) where G =
    Individual{G}(genome, convert(FitnessValue, fitnessvalue))

genome(i::Individual) = i.genome
fitness(i::Individual) = i.fitnessvalue

copy(individual::Individual) = Individual(copy(individual.genome), individual.fitnessvalue)

rand(rng::AbstractRNG, ::SamplerType{Individual{T}}) where {T} = Individual{T}(rand(T), nothing)


"""
    ongenome(f, i[, is])

Apply `f` on the `genome` fields of all individuals: `f(i.genome, is[1].genome, ..., is[N].genome)`.
"""
function ongenome(f, i::Individual)
    f(i.genome)
end

function ongenome(f, i::Individual, is::Vararg{Individual, N}) where {N}
    if @generated
        :(f(i.genome, $((:(is[$n].genome) for n in 1:N)...)))
    else
        f(getfield.((i, is...), :genome)...)
    end
end


const Population{T} = Vector{Individual{T}}


const Family{T, N} = NTuple{N, Individual{T}}
(::Type{Family{T, N}})(args::Vararg{T, N}) where {T,N} = args


