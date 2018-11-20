import Base: copy
import Random: AbstractRNG, rand, SamplerType

export Individual

mutable struct Individual{G}
    genome::G
    fitnessvalue::Union{FitnessValue, Nothing}
end

Individual(genome::G) where {G} = Individual{G}(genome, nothing)
Individual(genome::G, fitnessvalue) where G =
    Individual{G}(genome, convert(FitnessValue, fitnessvalue))

copy(individual::Individual) = Individual(copy(individual.genome), individual.fitnessvalue)

rand(rng::AbstractRNG, ::SamplerType{Individual{T}}) where {T} = Individual{T}(rand(T), nothing)
