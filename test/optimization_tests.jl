module Optimization

using Test

using Darwin
# import Darwin: crossover!, mutate!, selection

using LearningStrategies: learn!, MaxIter, strategy, Verbose
using Distributions

include("test_functions.jl")


const Entity = Vector{Float64}


function run_with(fitness, bounds, selection, crossover, mutation, generations)
    initial_population = Population([rand(bounds, 2) for _ in 1:128])
    model = PopulationModel(initial_population, fitness)
    strat = strategy(Verbose(GAStrategy(selection, crossover, mutation)),
                     MaxIter(generations))
    learn!(model, strat)
end

function test_with(fn, selection, crossover, mutation, generations)
    fitness = FitnessFunction{Entity}(function (x::Entity)
                                          -pack(fn, Entity)(x)
                                      end)
    Bounds = Uniform(bounds(fn)...)
    
    model = run_with(fitness, Bounds, selection, crossover, mutation, generations)

    @test fitness(model.fittest.genome) ≈ minimum(fn) atol = 0.01
    @test model.fittest.genome ≈ argmin(fn) atol = 0.15
end

function test_skip_with(fn, selection, crossover, mutation, generations)
    fitness = FitnessFunction{Entity}(function (x::Entity)
                                          -pack(fn, Entity)(x)
                                      end)
    Bounds = Uniform(bounds(fn)...)
    
    model = run_with(fitness, Bounds, selection, crossover, mutation, generations)

    @test_skip fitness(model.fittest.genome) ≈ minimum(fn) atol = 0.01
    @test_skip model.fittest.genome ≈ argmin(fn) atol = 0.15
end


@testset "Rosenbrock" begin
    rosenbrock = Rosenbrock{Float64}()
    
    test_with(rosenbrock,
              SoftmaxSelection{Entity, 2, 2}(ExponentialRate(1.0, 0.5, 0.999)),
              ArithmeticCrossover{Entity, 2, 2}(ConstantRate(0.7)),
              PointwiseMutation{Entity}(ConstantRate(0.2), Uniform(bounds(rosenbrock)...)),
              10_000)

    test_with(rosenbrock,
              TournamentSelection{Entity, 5, 1, 2}(),
              UniformCrossover{Entity, 2, 1}(LinearRate(0.8, 0.2, 1/20)),
              AdditiveMutation{Entity, Uniform}(ConstantRate(0.3), 2.0, -2.0, 2.0),
              10_000)

    test_skip_with(rosenbrock,
                   L1Selection{Entity, 2, 2}(),
                   ArithmeticCrossover{Entity, 2, 2}(ConstantRate(0.7)),
                   AdditiveMutation{Entity, Normal}(ConstantRate(0.3), 2.0, -2.0, 2.0),
                   10_000)
end


end # module
