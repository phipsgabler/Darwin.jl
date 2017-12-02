using Darwin
using Distributions

const Entity = Vector{Float64}
const Population = Vector{Entity}

# https://en.wikipedia.org/wiki/Himmelblau%27s_function
rosenbrock(x::Entity; a = 1, b = 100) = (a - x[1])^2 + b * (x[2] - x[1]^2)^2
const bounds = (-2.0, 2.0)
# minimum 0 at [a, a²]

fitness(x::Entity) = -rosenbrock(x)

@inline function softmax(x)
    exps = exp.(x)
    exps ./= sum(exps)
    return exps
end

function _selection(population::Population, selection_temperature)
    # softmax selection
    probabilities = softmax(fitness.(population) ./ selection_temperature)
    choices = rand(Categorical(probabilities), 2)
    return choices
end
selection = ParametrizedFunction(_selection, [1.0])

function _crossover(parents::Vector{Entity}, crossover_rate)
    # arithmetic crossover
    if rand() < crossover_rate
        mixing = rand()
        return [(1 - mixing) * parents[1] + mixing * parents[2],
                (1 - mixing) * parents[2] + mixing * parents[1]]
    else
        return parents
    end
end
crossover = ParametrizedFunction(_crossover, [0.7])

function _mutate!(x::Entity, mutation_rate)
    # uniform mutation of each component separately
    (rand() < mutation_rate) && (x[1] = rand(Uniform(bounds...)))
    (rand() < mutation_rate) && (x[2] = rand(Uniform(bounds...)))
end
mutate! = ParametrizedFunction(_mutate!, [0.3])

function callback(evolver)
    (evolver.model.selection.parameters[1] > 0.15) && (evolver.model.selection.parameters[1] *= 0.99)
    if evolver.generation % 1000 == 0
        evolver.model.mutate!.parameters[1] *= 0.9
        println(maximum(fitness.(evolver.solution.population)))
    end
end


const N = 128
initial_population = [rand(Uniform(bounds...), 2) for _ in 1:N]
model = GAModel(initial_population, selection, crossover, mutate!, 2)

result = evolve(model, 5000; verbose = true, callback = callback)
sample = rand(result.population, 10)
@show sample
@show fitness.(sample)


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

