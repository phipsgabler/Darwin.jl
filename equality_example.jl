import Darwin

struct EqualityMonster
    # a + 2b + 3c + 4d + 5e = 42
    abcde::Vector{Int}
end

EqualityMonster() = EqualityMonster(Vector{Int}(5))

function Darwin.fitness(ent::EqualityMonster)
    # we want the expression `a + 2b + 3c + 4d + 5e - 42`
    # to be as close to 0 as possible
    score = dot(ent.abcde, 1:5)
    abs(score - 42)
end

function Darwin.crossover(ent1::EqualityMonster, ent2::EqualityMonster)
    # grab each element from a random parent
    child1, child2 = EqualityMonster(), EqualityMonster()
    crossover_points = rand(Bool, 5)
    child1.abcde[crossover_points] .= view(ent1.abcde, crossover_points)
    child1.abcde[.~crossover_points] .= view(ent2.abcde, .~crossover_points)
    child2.abcde[crossover_points] .= view(ent2.abcde, crossover_points)
    child2.abcde[.~crossover_points] .= view(ent1.abcde, .~crossover_points)

    child1, child2
end

function Darwin.mutate!(ent::EqualityMonster)
    # let's go crazy and mutate 20% of the time
    # rand(Float64) < 0.8 && return

    rand_element = rand(1:5)
    ent.abcde[rand_element] = rand(0:42)
end


initial_population = [EqualityMonster(rand(0:42, 5)) for _ in 1:16]
problem = Darwin.GeneticProblem(initial_population, 1000, 0.2)

println(Darwin.evolve(problem))

