using Darwin
import Darwin: crossover, mutate!, selection, setup!

using LearningStrategies: learn!, MaxIter, strategy, Verbose

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


fitness = FitnessFunction{EqualityMonster}(function (ent::EqualityMonster)
    # we want the expression `a + 2b + 3c + 4d + 5e - 42`
    # to be as close to 0 as possible
    score = sum(ent.abcde .* (1:5))
    1 / abs(score - 42)
end)


const Selection = Tuple{Individual{EqualityMonster}, Individual{EqualityMonster}}
struct EMSelection <: SelectionStrategy{Selection} end

function selection(model::GAModel{T, Selection, F, EMSelection, Fc, Fm}) where {T, F, Fc, Fm}
    # simple naive groupings that pair the best entitiy with every other
    Iterators.zip(Iterators.repeated(model.best, length(model.population)), model.population)
end


struct EMCrossover <: CrossoverStrategy{Selection} end

function crossover(parents::Selection, ::EMCrossover)
    # grab each element from a random parent
    crossover_points = rand(Bool, 5)
    result = EqualityMonster()
    result.abcde[crossover_points] .= parents[1].genome.abcde[crossover_points]
    result.abcde[.~crossover_points] .= parents[2].genome.abcde[.~crossover_points]

    Individual(result)
end


struct EMMutation <: MutationStrategy{EqualityMonster}
    p::Float64
end

function mutate!(child::EqualityMonster, strat::EMMutation)
    if rand() < strat.p
        child.abcde[rand(1:5)] = rand(0:42)
    end
    
    child
end

initial_population = rand(Individual{EqualityMonster}, 64)

# let's go crazy and mutate 20% of the time
model = GAModel(initial_population, fitness, EMSelection(), EMCrossover(), EMMutation(0.2))
strat = strategy(Verbose(GAEvolver{EqualityMonster}()), MaxIter(200))

result = learn!(model, strat)


@testset "EqualityMonster" begin
    # @test isinf(fitness(result.population[fittest]))
    println(result.population)
    @test sum(result.best.genome.abcde .* (1:5)) == 42
end

