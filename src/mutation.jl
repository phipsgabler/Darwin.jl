using Distributions

export mutate!,
    MutationStrategy,
    setup!
    
export BitFlip,
    BoundedConvolution,
    BoundedGaussianConvolution,
    BoundedUniformConvolution,
    UniformMutation

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


mutable struct UniformMutation{T<:Real} <: MutationStrategy{AbstractVector{T}}
    rate::Float64
    bounds::Uniform{T}

    function UniformMutation(rate, l, u)
        U = promote_type(l, u)
        new{U}(rate, Uniform{U}(l, u))
    end
    
    UniformMutation(rate, bounds::Uniform{T}) where T = new{T}(rate, bounds)
end

function mutate!(genome::AbstractVector{T}, strat::UniformMutation{T}) where {T}
    for i in eachindex(genome)
        (rand() < strat.rate) && (genome[i] = rand(strat.bounds))
    end

    # if generation % 1000 == 0
        # strat.rate *= 0.9
    # end
    
    genome
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
