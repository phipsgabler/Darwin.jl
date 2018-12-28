using Distributions

export mutate!,
    MutationStrategy,
    setup!
    
export BitFlip,
    BoundedConvolution,
    BoundedGaussianConvolution,
    BoundedUniformConvolution,
    LiftedMutation,
    PointwiseMutation

abstract type MutationStrategy{G} end

setup!(strategy::MutationStrategy, model) = strategy


struct LiftedMutation{T, M, I} <: MutationStrategy{T}
    inner::M

    LiftedMutation{T}(strat::M) where {T, I, M<:MutationStrategy{I}} = new{T, M, I}(strat)
    LiftedMutation{T, M}(args...) where {T, I, M<:MutationStrategy{I}} = new{T, M, I}(M(args...))
end



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


struct NoMutation{T} <: MutationStrategy{T} end

mutate!(genome::Any, strat::NoMutation) = genome


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


struct PointwiseMutation{T<:AbstractVector} <: MutationStrategy{T}
    rate::Float64
    tweak::Distribution{Univariate}

    PointwiseMutation{T}(rate, tweak::Distribution{Univariate, Discrete}) where {T<:AbstractVector{<:Integer}} =
        new{T}(rate, tweak)
    PointwiseMutation{T}(rate, tweak::Distribution{Univariate, Continuous}) where {T<:AbstractVector{<:AbstractFloat}} =
        new{T}(rate, tweak)
end

function mutate!(genome::T, strat::PointwiseMutation{T}) where {T}
    for i in eachindex(genome)
        (rand() < strat.rate) && (genome[i] = rand(strat.tweak))
    end

    genome
end


struct BoundedConvolution{T<:AbstractVector} <: MutationStrategy{T}
    rate::Float64
    tweak::Distribution{Univariate, Continuous}
    min::Real
    max::Real

    function BoundedConvolution{T}(rate, tweak::Distribution{Univariate, Continuous}, min, max) where T
        @assert (mean(tweak) == 0) "`tweak` should have zero mean!"
        new{T}(rate, tweak, min, max)
    end
end

function mutate!(genome::T, strat::BoundedConvolution{T}) where {T}
    for i in eachindex(genome)
        if rand() ≤ strat.rate
            genome[i] = rand(Truncated(Shifted(strat.tweak, genome[i]), strat.min, strat.max))
        end
    end
end


struct BoundedUniformConvolution{T<:AbstractVector} <: MutationStrategy{T}
    bc::BoundedConvolution{T}
    BoundedUniformConvolution{T}(rate, r, min, max) where T =
        new{T}(BoundedConvolution{T}(rate, Uniform(-r, r), min, max))
end

struct BoundedGaussianConvolution{T<:AbstractVector} <: MutationStrategy{T}
    bc::BoundedConvolution{T}
    BoundedGaussianConvolution{T}(rate, σ, min, max) where T =
        new{T}(BoundedConvolution{T}(rate, Normal(0, σ), min, max))
end

function mutate!(genome::T,
                 strat::Union{BoundedUniformConvolution{T}, BoundedGaussianConvolution{T}}) where {T}
    mutate!(genome, strat.bc)
end
