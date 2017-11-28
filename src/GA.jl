using Distributions: Bernoulli, Categorical

struct GeneticProblem{T, P<:AbstractVector} <: AbstractEvolutionaryProblem
    initial_population::P
    generations::Int
    mutation_probability::Float64
    GeneticProblem{T, P}(ip::AbstractVector{T}, g::Int, mp::Float64) where {T, P} = new(ip, g, mp)
end

GeneticProblem(ip, g, mp) = GeneticProblem{eltype(ip), typeof(ip)}(ip, g, mp)

function mutate end

function crossover end

function fitness end

function softmax(x)
    # https://stats.stackexchange.com/a/163240
    # max_x = max(zero(x), maximum(x))
    # rebased_x = x - max_x
    # return rebased_x - np.logaddexp(-max_x, np.logaddexp.reduce(rebased_x))
    exps = exp.(x)
    exps ./ sum(exps)
end


function evolve{T, P}(problem::GeneticProblem{T, P})
    N = length(problem.initial_population)
    @assert iseven(N)
    
    populations = Vector{P}(problem.generations)
    populations[1] = problem.initial_population
    
    mutation() = rand(Bernoulli(problem.mutation_probability)) == 1
    
    for g = 2:problem.generations
        children = similar(problem.initial_population)
        parents = populations[g - 1]
        
        fitness_distribution = Categorical(softmax(fitness.(parents)))
        
        for n = 1:div(N, 2)
            p1, p2 = tuple(parents[rand(fitness_distribution, 2)]...)
            c1, c2 = crossover(p1, p2)

            mutation() || (c1 = mutate(c1))
            mutation() || (c2 = mutate(c2))

            children[n], children[n + 1] = c1, c2
        end

        populations[g] = children
    end
end
