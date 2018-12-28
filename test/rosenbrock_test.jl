module Rosenbrock

using Test

using Darwin
# import Darwin: crossover!, mutate!, selection

using LearningStrategies: learn!, MaxIter, strategy, Verbose
using Distributions


const Entity = Vector{Float64}

# https://en.wikipedia.org/wiki/Himmelblau%27s_function
rosenbrock(x::Entity; a::Float64 = 1.0, b::Float64 = 100.0) = (a - x[1])^2 + b * (x[2] - x[1]^2)^2
Bounds = Uniform(-2.0, 2.0)
# minimum 0 at [a, a²]

@fitness function fitness(x::Entity)
    -rosenbrock(x)
end


function run_with(selection, crossover, mutation, generations)
    initial_population = Individual.([rand(Bounds, 2) for _ in 1:128])
    model = GAModel(initial_population, fitness, selection, crossover, mutation)
    strat = strategy(Verbose(GAEvolver{Entity}()), MaxIter(generations))
    learn!(model, strat)
end


@testset "Rosenbrock" begin
    r1 = run_with(SoftmaxSelection{Entity, 2, 2}(ExponentialRate(1.0, 0.5, 0.999)),
                  ArithmeticCrossover{Entity, 2, 2}(ConstantRate(0.7)),
                  PointwiseMutation{Entity}(ConstantRate(0.2), Bounds),
                  10_000)

    @test rosenbrock(r1.fittest.genome) ≈ 0.0 atol = 0.01
    @test r1.fittest.genome ≈ [1.0, 1.0] atol = 0.15

    r2 = run_with(TournamentSelection{Entity, 5, 1, 2}(),
                  UniformCrossover{Entity, 2, 1}(LinearRate(0.8, 0.2, 1/20)),
                  AdditiveMutation{Entity, Uniform}(ConstantRate(0.3), 2.0, -2.0, 2.0),
                  10_000)

    @test rosenbrock(r2.fittest.genome) ≈ 0.0 atol = 0.01
    @test r2.fittest.genome ≈ [1.0, 1.0] atol = 0.1

    r3 = run_with(L1Selection{Entity, 2, 2}(),
                  ArithmeticCrossover{Entity, 2, 2}(ConstantRate(0.7)),
                  AdditiveMutation{Entity, Normal}(ConstantRate(0.3), 2.0, -2.0, 2.0),
                  10_000)

    @test_skip rosenbrock(r3.fittest.genome) ≈ 0.0 atol = 0.01
    @test_skip r2.fittest.genome ≈ [1.0, 1.0] atol = 0.1
end


# function list: https://www.sfu.ca/~ssurjano/optimization.html

# function rastrigin(x)
#     d = length(x)
#     10d + sum(x .^ 2 + 10cos(2π .* x))
# end
# bound_rastrigin = (-5, 5)
# minimum: 0 at [0, 0]

# function ackley(x; a = 20, b = 1/5, c = 2π)
#     d = length(x)
#     e + a - a * exp(-b * sqrt(sum(x .^ 2) / d)) - exp(sum(cos.(c .* x)) / d)
# end
# bound_ackley = (-30, 30)
# minimum 0 at [0, 0]

# chasm(x) = 10^3 * abs(x[1]) / (10^3 * abs(x[0]) + 1) + 10^(-2) * abs(x[1])
# bound_chasm = (-5, 5)

# himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
# bound_himmelblau = (-4, 4)
# 4 minima of 0, see https://en.wikipedia.org/wiki/Himmelblau%27s_function

end # module
