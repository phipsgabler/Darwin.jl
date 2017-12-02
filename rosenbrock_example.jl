using Darwin
using Distributions

const Entity = Vector{Float64}
const Population = Vector{Entity}

# https://en.wikipedia.org/wiki/Himmelblau%27s_function
rosenbrock(x::Entity; a = 1, b = 100) = (a - x[1])^2 + b * (x[2] - x[1]^2)^2
const bounds = (-2.0, 2.0)
# minimum 0 at [a, a²]

fitness(x::Entity) = -rosenbrock(x)

function selection(population::Population)
    # @show population, fitness.(population)
    exps = exp.(fitness.(population))
    probabilities = exps ./ sum(exps)
    choices = rand(Categorical(probabilities), size(population))
    return choices
end

function crossover(population::Population)
    N = length(population)
    partners = randperm(N)
    mixing = rand(N)
    return mixing .* population .+ (1 .- mixing) .* population[partners]
end

function mutate!(x::Entity)
    if rand() < 0.05
        location = rand(Bool, size(x))
        x[location] .= rand(Uniform(bounds...), length(x))[location]
    end
    
    # mutation_mask = rand(Bernoulli(0.2), size(population)) .== 1
    # n_mutations = countnz(mutation_mask)
    # population[mutation_mask] .= rand(Uniform(bounds...), n_mutations)
end

const N = 128
initial_population = [rand(Uniform(bounds...), 2) for _ in 1:N]
model = GAModel(initial_population, selection, crossover, mutate!, N)
# println(model)

result = evolve(model, 1000; verbose = true)
@show fitness.(result.population)


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

