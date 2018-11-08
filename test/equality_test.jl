using Darwin
import Darwin: selection, setup!, mutate!, crossover!

using LearningStrategies: learn!, Verbose

import Random: AbstractRNG, rand, SamplerType

# This example is taken from Westley Argentum:
# https://github.com/WestleyArgentum/GeneticAlgorithms.jl

struct EqualityMonster
    # a + 2b + 3c + 4d + 5e = 42
    abcde::Vector{Int}
end

EqualityMonster() = EqualityMonster(Vector{Int}(5))

rand(rng::AbstractRNG, ::SamplerType{EqualityMonster}) = EqualityMonster(rand(rng, 0:42, 5))


function fitness(ent::EqualityMonster)
    # we want the expression `a + 2b + 3c + 4d + 5e - 42`
    # to be as close to 0 as possible
    score = sum(ent.abcde .* (1:5))
    1 / abs(score - 42)
end


struct EMSelection <: SelectionStrategy end

function selection(population, ::EMSelection)
    # simple naive groupings that pair the best entitiy with every other
    fittest = argmax(fitness.(population))
    Iterators.repeated(fittest, length(population)), eachindex(population)
end


struct EMCrossover <: CrossoverStrategy end

function crossover!(children, parents, selection, ::EMCrossover)
    # grab each element from a random parent
    for (i, p1, p2) in zip(eachindex(children), selection...)
        crossover_points = rand(Bool, 5)
        abcde = similar(parents[p1].abcde)
        abcde[crossover_points] = parents[p1].abcde[crossover_points]
        abcde[.~crossover_points] = parents[p2].abcde[.~crossover_points]

        children[i] = EqualityMonster(abcde)
    end

    children
end


struct EMMutation <: MutationStrategy
    p::Float64
end

function mutate!(children, strat::EMMutation)
    for i in eachindex(children)
        if rand() < strat.p
            children[i].abcde[rand(1:5)] = rand(0:42)
        end
    end

    children
end

initial_population = rand(EqualityMonster, 64)

# let's go crazy and mutate 20% of the time
model = GAModel(initial_population, EMSelection(), EMCrossover(), EMMutation(0.2))

result = learn!(model, Verbose(GAEvolver{EqualityMonster}(200)))


@testset "EqualityMonster" begin
    fittest = result.population[argmax(fitness.(result.population))]
    # @test isinf(fitness(result.population[fittest]))
    @test sum(fittest.abcde .* (1:5)) == 42
end

