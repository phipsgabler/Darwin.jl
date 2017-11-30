using Darwin
using Distributions

const Entity = Vector{Float64}
const Population = Vector{Entity}

# https://en.wikipedia.org/wiki/Himmelblau%27s_function
rosenbrock(x::Entity; a = 1.0, b = 100.0) = (a - x[1])^2 + b * (x[2] - x[1]^2)^2
const bounds = (-2.0, 2.0)

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
model = GAModel(initial_population, selection, crossover, mutate!, 100, N)
println(model)

result = evolve(model)
@show result
@show fitness.(result)



# def rastrigin(x):
#     x = np.array(x)
#     return np.sum(x**2 + 10 - 10* np.cos(2* np.pi*x))
# bound_rastrigin = [-5,5]

# def rosenbrock(x):
#     x = np.array(x)
#     return (x[1]-x[0]**2)**2 + (x[0]-1)/2 + 2*np.sum(np.abs(x-1.5))
# bound_rosenbrock = [-2,2]

# def ackley(x):
#     x = np.array(x)
#     return np.exp(1) + 20 - 20*np.exp(-0.2*np.sqrt(1/2*np.sum(x**2))) - np.exp(0.5*np.sum(np.cos(2*np.pi*x)))
# bound_ackley = [-2,2]

# def chasm(x):
#     x = np.array(x)
#     return 1e3*np.abs(x[0])/(1e3*np.abs(x[0])+1) + 1e-2*np.abs(x[1])
# bound_chasm = [-5,5]

