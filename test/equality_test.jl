module Equality

using Test

using Darwin
import Darwin: crossover!, mutate!, selection

using LearningStrategies: learn!, MaxIter, strategy, Verbose
using Distributions

import Base: copy
import Random: AbstractRNG, rand, SamplerType


# This example is taken from Westley Argentum:
# https://github.com/WestleyArgentum/GeneticAlgorithms.jl

struct EqualityMonster
    # a + 2b + 3c + 4d + 5e = 42
    abcde::Vector{Int}
end

EqualityMonster() = EqualityMonster(Vector{Int}(undef, 5))

copy(m::EqualityMonster) = EqualityMonster(copy(m.abcde))
rand(rng::AbstractRNG, ::SamplerType{EqualityMonster}) = EqualityMonster(rand(rng, 0:42, 5))


@fitness function fitness(ent::EqualityMonster)
    # we want the expression `a + 2b + 3c + 4d + 5e - 42`
    # to be as close to 0 as possible
    score = sum(ent.abcde .* (1:5))
    1 / abs(score - 42)
end


const EMCrossover = LiftedCrossover{EqualityMonster, UniformCrossover{Vector{Int}, 2, 1}}

function crossover!((p₁, p₂)::NTuple{2, EqualityMonster}, strategy::EMCrossover)
    EqualityMonster.(crossover!((p₁.abcde, p₂.abcde), strategy.inner))
end


const EMMutation = LiftedMutation{EqualityMonster, PointwiseMutation{Vector{Int}}}

function mutate!(child::EqualityMonster, strat::EMMutation)
    mutate!(child.abcde, strat.inner)
    child
end


function run_with(selection, crossover, mutation, generations)
    initial_population = rand(Individual{EqualityMonster}, 64)
    model = GAModel(initial_population, fitness, selection, crossover, mutation)
    strat = strategy(Verbose(GAEvolver{EqualityMonster}()), MaxIter(generations))
    learn!(model, strat)
end


@testset "EqualityMonster" begin
    # @test isinf(fitness(result.population[fittest]))
    
    # let's go crazy and mutate 20% of the time
    r1 = run_with(PairWithBest{EqualityMonster, 1}(),
                  EMCrossover(),
                  EMMutation(0.2, DiscreteUniform(0, 42)),
                  200)
    @test sum(r1.fittest.genome.abcde .* (1:5)) == 42

    r2 = run_with(TournamentSelection{EqualityMonster, 5, 1, 2}(),
                  EMCrossover(),
                  EMMutation(0.2, DiscreteUniform(0, 42)),
                  300)
    @test sum(r2.fittest.genome.abcde .* (1:5)) == 42
end

end # module
