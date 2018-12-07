using Darwin
import Darwin: crossover, mutate!, selection, setup!,
    PairWithBestSelection

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


@fitness function fitness(ent::EqualityMonster)
    # we want the expression `a + 2b + 3c + 4d + 5e - 42`
    # to be as close to 0 as possible
    score = sum(ent.abcde .* (1:5))
    1 / abs(score - 42)
end


const SelectionResult = NTuple{2, Individual{EqualityMonster}}
const EMSelection = PairWithBestSelection{EqualityMonster}

struct EMCrossover <: CrossoverStrategy{SelectionResult} end

function crossover(parents::SelectionResult, ::EMCrossover)
    # grab each element from a random parent
    crossover_points = rand(Bool, 5)
    result = EqualityMonster()
    
    ongenome(parents[1], parents[2]) do g1, g2
        result.abcde[crossover_points] .= g1.abcde[crossover_points]
        result.abcde[.~crossover_points] .= g2.abcde[.~crossover_points]
    end
    
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
    @test sum(result.fittest.genome.abcde .* (1:5)) == 42
end

