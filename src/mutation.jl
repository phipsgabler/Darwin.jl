using Distributions

export mutate!,
    MutationOperator,
    setup!
    
export BitFlipMutation,
    AdditiveMutation,
    LiftedMutation,
    PointwiseMutation

abstract type MutationOperator{G} end

setup!(operator::MutationOperator{G}, model::AbstractPopulationModel{G}) where {G} = operator


"""
    mutate!(genome, operator, generation) -> genome
    mutate!(individual, operator, generation) -> individual

Mutate `genome` or `individual.genome` in place.  You only need to define the needed genome form.
"""
function mutate!(individual::Individual{G}, operator::MutationOperator{G},
                 generation::Integer) where {G}
    mutate!(individual.genome, operator, generation)
    individual
end


struct NoMutation <: MutationOperator{Any} end

mutate!(genome::Any, operator::NoMutation) = genome


struct LiftedMutation{G, M, I} <: MutationOperator{G}
    inner::M

    LiftedMutation{G}(operator::M) where {G, I, M<:MutationOperator{I}} = new{G, M, I}(operator)
    LiftedMutation{G, M}(args...) where {G, I, M<:MutationOperator{I}} = new{G, M, I}(M(args...))
end


struct BitFlipMutation <: MutationOperator{AbstractVector{Bool}}
    rate::Rate
end

function mutate!(genome::AbstractVector{Bool}, operator::BitFlipMutation, generation::Integer)
    for i in eachindex(genome)
        (rand() < operator.rate(generation)) && (genome[i] = !genome[i])
    end

    genome
end


struct PointwiseMutation{G<:AbstractVector} <: MutationOperator{G}
    rate::Rate
    tweak::Distribution{Univariate}

    PointwiseMutation{G}(rate, tweak::Distribution{Univariate, Discrete}) where {G<:AbstractVector{<:Integer}} =
        new{G}(rate, tweak)
    PointwiseMutation{G}(rate, tweak::Distribution{Univariate, Continuous}) where {G<:AbstractVector{<:AbstractFloat}} =
        new{G}(rate, tweak)
end

function mutate!(genome::G, operator::PointwiseMutation{G},
                 generation::Integer) where {G<:AbstractVector}
    for i in eachindex(genome)
        (rand() < operator.rate(generation)) && (genome[i] = rand(operator.tweak))
    end

    genome
end


struct AdditiveMutation{G<:AbstractVector, D<:Distribution{Univariate}} <: MutationOperator{G}
    rate::Rate
    tweak::D
    min::Real
    max::Real

    function AdditiveMutation{G}(rate, tweak::D, min, max) where {G, D<:Distribution{Univariate}}
        @assert (mean(tweak) == 0) "`tweak` should have zero mean!"
        new{G, D}(rate, tweak, min, max)
    end

    AdditiveMutation{G, D}(rate, r, min, max) where {G, D<:Uniform} =
        new{G, D}(rate, D(-r, r), min, max)
    AdditiveMutation{G, D}(rate, σ, min, max) where {G, D<:Normal} =
        new{G, D}(rate, D(0, σ), min, max)
    AdditiveMutation{G, D}(rate, r, min, max) where {G, D<:DiscreteUniform} =
        new{G, D}(rate, D(-r, r), min, max)
end

function mutate!(genome::G, operator::AdditiveMutation{G},
                 generation::Integer) where {G<:AbstractVector}
    for i in eachindex(genome)
        if rand() ≤ operator.rate(generation)
            genome[i] = rand(Truncated(Shifted(operator.tweak, genome[i]),
                                       operator.min, operator.max))
        end
    end

    genome
end

