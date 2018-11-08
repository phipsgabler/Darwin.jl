using Darwin
import Darwin: selection, setup!, mutate!, crossover!

using Distributions

using LearningStrategies: learn!, Verbose


const Entity = Vector{Float64}

# https://en.wikipedia.org/wiki/Himmelblau%27s_function
rosenbrock(x::Entity; a = 1, b = 100) = (a - x[1])^2 + b * (x[2] - x[1]^2)^2
const Bounds = Uniform(-2.0, 2.0)
# minimum 0 at [a, a²]

fitness(x::Entity) = -rosenbrock(x)

function softmax(x)
    exps = exp.(x)
    exps ./= sum(exps)
    return exps
end


mutable struct RosenbrockSelection <: SelectionStrategy
    temperature::Float64
end

function selection(population, strat::RosenbrockSelection, generation)
    # softmax selection
    probabilities = softmax(fitness.(population) ./ strat.temperature)
    M = length(population) ÷ 2
    D = Categorical(probabilities)

    # tempering
    if strat.temperature ≥ 0.15
        strat.temperature *= 0.99
    end
    
    rand(D, M), rand(D, M)
end


mutable struct RosenbrockCrossover <: CrossoverStrategy
    rate::Float64
end

function crossover!(children, parents, selection, strat::RosenbrockCrossover, generation)
    # arithmetic crossover
    CI = Iterators.partition(eachindex(children), 2)
    
    for (ci, p1, p2) in zip(CI, selection...)
        if rand() < strat.rate
            mixing = rand()
            children[ci] .= [(1 - mixing) .* parents[p1] .+ mixing .* parents[p2],
                             (1 - mixing) .* parents[p2] .+ mixing .* parents[p1]]
        else
            children[ci] .= [parents[p1], parents[p2]]
        end
    end

    children
end


mutable struct RosenbrockMutation <: MutationStrategy
    rate::Float64
end

function mutate!(children, strat::RosenbrockMutation, generation)
    # uniform mutation of each component separately
    for child in children
        (rand() < strat.rate) && (child[1] = rand(Bounds))
        (rand() < strat.rate) && (child[2] = rand(Bounds))
    end

    if generation % 1000 == 0
        strat.rate *= 0.9
    end
    
    children
end

# function callback(evolver)
#     (evolver.model.selections.parameters[1] > 0.15) &&
#         (evolver.model.selections.parameters[1] *= 0.99)
#     if evolver.generation % 1000 == 0
#         evolver.model.mutate!.parameters[1] *= 0.9
#         println("Maximum fitness at generation ", evolver.generation,
#                 ": ", maximum(fitness.(evolver.solution.population)))
#     end
# end


const N = 128
initial_population = [rand(Bounds, 2) for _ in 1:N]
model = GAModel(initial_population, RosenbrockSelection(1.0),
                RosenbrockCrossover(0.7), RosenbrockMutation(0.3))

result = learn!(model, Verbose(GAEvolver{Entity}(8000)))

fittest = argmax(fitness.(result.population))
@test isapprox(rosenbrock(result.population[fittest]), 0.0, atol = 0.01)
# @show result.population[fittest]
@test all(isapprox.(result.population[fittest], [1.0, 1.0], atol = 0.1))


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

