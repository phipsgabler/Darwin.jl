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


struct PointwiseMutation{T} <: MutationStrategy{AbstractVector{T}}
    rate::Float64
    tweak::Distribution{Univariate}

    PointwiseMutation{Int}(rate, tweak::Distribution{Univariate, Discrete}) = new{Int}(rate, tweak)
    PointwiseMutation(rate, tweak::Distribution{Univariate, Discrete}) = new{Int}(rate, tweak)
    
    PointwiseMutation(rate, tweak::Distribution{Univariate, Continuous}) = new{Float64}(rate, tweak)
end

function mutate!(genome::AbstractVector{T}, strat::PointwiseMutation{T}) where {T}
    for i in eachindex(genome)
        (rand() < strat.rate) && (genome[i] = rand(strat.tweak))
    end

    genome
end


struct BoundedConvolution{T<:Real} <: MutationStrategy{AbstractVector{T}}
    rate::Float64
    tweak::Distribution{Univariate}
    min::T
    max::T

    function BoundedConvolution(rate, tweak::Distribution{Univariate}, min, max)
        T = eltype(tweak)
        @assert (mean(tweak) == zero(T)) "`tweak` should have zero mean!"
        TT = promote_type(typeof(min), typeof(max), T)
        new{TT}(convert(Float64, rate), r, convert(TT, min), convert(TT, max))
    end
end

function mutate!(genome::AbstractVector{T}, strat::BoundedConvolution{T}) where {T<:Real}
    for i in eachindex(genome)
        if rand() ≤ strat.rate
            genome[i] = rand(Truncated(Shifted(strat.tweak, genome[i]), strat.min, strat.max))
        end
    end
end

BoundedUniformConvolution(rate, r, min, max) = BoundedConvolution(rate, Uniform(-r, r), min, max)
BoundedGaussianConvolution(rate, σ, min, max) = BoundedConvolution(rate, Normal(0, σ), min, max)
