using Distributions

export mutate!,
    MutationStrategy,
    setup!

abstract type MutationStrategy{G} end

setup!(strategy::MutationStrategy, model) = strategy

"""
    mutate!(genome, strategy[, generation]) -> genome
    mutate!(individual, strategy, generation) -> individual

Mutate `genome` or `individual.genome` in place.  You only need to define the needed genome form.
"""
mutate!(genome::G, strategy::MutationStrategy{G}, generation::Integer) where {G} =
    mutate!(genome, strategy)

function mutate!(individual::Individual{G}, strategy::MutationStrategy{G},
                 generation::Integer) where {G}
    mutate!(individual.genome, strategy, generation)
    individual
end


struct BitFlip <: MutationStrategy{AbstractVector{Bool}}
    p::Float64
end

function mutate!(genome::AbstractVector{Bool}, strat::BitFlip)
    for i in eachindex(genome)
        if rand() < strat.p
            genome[i] = !genome[i]
        end
    end
end


struct BoundedConvolution{T<:Real} <: MutationStrategy{AbstractVector{T}}
    p::Float64
    tweak::Distribution{T}
    min::T
    max::T

    function BoundedConvolution(p, tweak::Distribution{T}, min, max) where T
        @assert mean(tweak) == zero(T) "`tweak` should have zero mean!"
        TT = promote_type(typeof(min), typeof(max), T)
        new{TT}(convert(Float64, p), r, convert(TT, min), convert(TT, max))
    end
end

function mutate!(genome::AbstractVector{T}, strat::BoundedConvolution{T}) where {T<:Real}
    for i in eachindex(genome)
        if rand() ≤ strat.p
            genome[i] = rand(Truncated(Shifted(strat.tweak, genome[i]), strat.min, strat.max))
        end
    end
end

BoundedUniformConvolution(p, r, min, max) = BoundedConvolution(p, Uniform(-r, r), min, max)
BoundedGaussianConvolution(p, σ, min, max) = BoundedConvolution(p, Normal(0, σ), min, max)
