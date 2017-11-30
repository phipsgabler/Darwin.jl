using Darwin
using Distributions
import Base: rand

# This example is taken from Westley Argentum:
# https://github.com/WestleyArgentum/GeneticAlgorithms.jl/README.md

struct EqualityMonster
    # a + 2b + 3c + 4d + 5e = 42
    abcde::Vector{Int}
end

EqualityMonster() = EqualityMonster(Vector{Int}(5))

Base.rand(rng::AbstractRNG, ::Type{EqualityMonster}) = EqualityMonster(rand(rng, 0:42, 5))

function fitness(ent::EqualityMonster)
    # we want the expression `a + 2b + 3c + 4d + 5e - 42`
    # to be as close to 0 as possible
    score = dot(ent.abcde, 1:5)
    1 / abs(score - 42)
end

function selection(parents::Vector{EqualityMonster})
    fittest = indmax(fitness(e) for e in parents)
    [fittest, rand(indices(parents, 1))]
end

function crossover(parents::Vector{EqualityMonster})
    ent1, ent2 = tuple(parents...)
    
    # grab each element from a random parent
    child1, child2 = EqualityMonster(), EqualityMonster()
    crossover_points = rand(Bool, 5)
    child1.abcde[crossover_points] .= view(ent1.abcde, crossover_points)
    child1.abcde[.~crossover_points] .= view(ent2.abcde, .~crossover_points)
    child2.abcde[crossover_points] .= view(ent2.abcde, crossover_points)
    child2.abcde[.~crossover_points] .= view(ent1.abcde, .~crossover_points)

    [child1, child2]
end

function mutate!(ent::EqualityMonster)
    # let's go crazy and mutate 20% of the time
    if rand() < 0.2
        rand_element = rand(1:5)
        ent.abcde[rand_element] = rand(0:42)
    end
end


initial_population = rand(EqualityMonster, 16)
model = GAModel(initial_population, selection, crossover, mutate!, 500)
println(model)

result = evolve(model)
@show result
@show fitness.(result)

