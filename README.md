# Darwin

[![Build
Status](https://travis-ci.org/phipsgabler/Darwin.jl.svg?branch=master)](https://travis-ci.org/phipsgabler/Darwin.jl)

This package tries to implement evolutionary algorithms (or "population methods", see Luke, 2013) with a very 
flexible interface.  Currently, only the [genetic algorithm](http://www.scholarpedia.org/article/Genetic_algorithm)
(GA) in its classic form is supported, but I plan to add other variations and evolution strategies (ES).

## Background/Alternatives

There are two existing Julia packages that I know of which implement genetic algorithms,
[Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) and
[GeneticAlgorithms.jl](https://github.com/WestleyArgentum/GeneticAlgorithms.jl).  I decided to write
a completely new on since I didn't really like the interfaces of both (GeneticAlgorithms requires to
write a module and uses task functions for grouping, and Evolutionary is less flexible and exposes
much of internal specifics, IMHO).  

Instead, I wanted to have something like in the style
[DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl), with a very
flexible, high level interface and convenient control possibilities.  Originally I tried to replicate 
their iterator interface, but then switched to the more general interface of 
[LearningStrategies.jl](https://github.com/JuliaML/LearningStrategies.jl), which is explained below.

Additionally, I took efforts to implement other parts in a combinator style as well, e.g. genetic
operators, lifting of them, rate parameters, etc.

## Usage

In the style of `LearningStrategies`, you need to define define two things for optimizing a problem: 

- A model, which contains the data and specification of the problem -- _what is be optimized_.  In this package, 
  this will be a subtype of `AbstractEvolutionaryModel`, holding at least the population and fitness.
- A `LearningStrategy` which determines _how the model should be optimized_.  This corresponds to a certain
  algorithm, such as `GAStrategy`.  The strategy includes the genetic operators and internal values, e.g. caches.

Let's look at an example for GAs:

```julia
model = GAModel(initial_population, fitness)
strat = strategy(GAStrategy(selection, crossover, mutation), MaxIter(generations))
learn!(model, strat)
```

By calling `learn!(model, strat)`, you execute the following loop on `model`:
  
```julia
setup!(strat, model)
for (generation, _d) in Iterators.repeated(nothing)
    update!(model, strat, generation, _d)
    hook(strat, model, d_, generation)
    finished(strat, model, d_, generation) && break
end
cleanup!(strat, model)
```

(The presence of the unused `_d` is a leftover from `LearningStrategies`, but we usually have no data 
be read iteratively in population methods).  The main part of a strategy consists the `update!` function,
 in which the genetic operators are applied.  Termination through `finished` is mostly left to 
meta-strategies like `LearningStrategies.MaxIter`, which can be combined with an evolutionary algorithm 
in an arbitrary way:

```julia
strategy(Verbose(strat), MaxIter(g))
```

would be the most common one, specifying termination after `g` generations and printing stuff about `strat` 
(see [here](https://github.com/JuliaML/LearningStrategies.jl#built-in-strategies) for about other strategies).

### Types

Mostly everything is parametrized by a first type parameter `T` for the "things which should be optimized" -- 
that is, the genome which is used.  

Internally in models and such, not `T` itself is used, but a wrapper `Individual{T}`, which has an additional 
field for caching the fitness of a genome.  Thus you don't have deal with fitness caching yourself.

There are two other type synonyms related to `Individual`:

```julia
const Population{T} = Vector{Individual{T}}
const Family{T, N} = NTuple{N, Individual{T}}
```

Usually, the only place where you have to deal with `Individuals` is where you set up the initial population
for your problem, like the following:

```julia
Population(rand(Genome, N))
```

The type `T` should implement `rand`, `copy`, and have an `AbstractFitness{T}` object associated with it.


## Strategies

### Genetic Algorithm

The genetic algorithm has the following abstract structure:

```julia
while !terminated
  evaluate_fitness!(population)
  
  for parents in selections(population)
    for child in crossover(copy.(parents))
      push!(new_population, mutate!(child))
    end
  end
  
  population = new_population
end
```

Because we _iterate_ over the parent selection and crossed-over children, there's some 
flexibility in how to divide and rebuild a population.  To be exact, the strategy for
the GA is defined like this:

```julia
mutable struct GAStrategy{T, P, K,
                          Fs<:SelectionOperator{>:T, P, K},
                          Fc<:CrossoverOperator{>:T, K, P},
                          Fm<:MutationOperator{>:T}} <: L.LearningStrategy
```
                          
The type parameters reflect the constraints on number of parents and children: `T` is the type 
of genomes, `P` the ratio of population to families (parent tuples), and `K` the number of 
children produced from each family.  

Thus, a `SelectionOperator{T, 1, 2}` will select two parents for each population member.
Similarly, a `SelectionOperator{T, 2, 1}` will select `length(population) ÷ 2` single 
parents, and `GAStrategy{T, 2, 2}` as many pairs of parents (the last variant being the
one used in most literature, AFAIK).

These selected parents are then passed to a crossover operator of opposite characteristics, 
producing `P` children out of the `K` parents.

This interface allows one to generalize the selection/crossover structure, while ensuring to
keep the population size constant (except for rounding errors when the populuation size is
not a multiple of `P`).

A simple GA run might thus look as follows:

```julia
initial_population = rand(Individual{Genome}, N)
model = GAModel(initial_population, fitness)
strat = strategy(Verbose(GAStrategy(selection, crossover, mutation)),
                 MaxIter(generations))
learn!(model, strat)
```

For this to work, you need to have some type `Genome` with `copy` and `rand` defined on it,
a fitness function of type `AbstractFitness`, and selection, crossover, and mutation operators
of suitable characteristics (with the numbers matching).


## Fitness

Evolutionary models in general require a fitness function for assigning quality to the
individuals in question.  Since Julia functions are not parametrized by their input and 
output types, an `AbstractFitness{T}` is required by this package, representing a fitness
operator on genomes of type `T`.  Such an operator should be callable on values of `T`.

The return type should be `Float64`, since that is what is cached in `Individual{T}`
and can represent any total order anyway.  In theory, it is enough to return something 
that can be compared using `less` and converted to `Float64`. 

Since in most cases, the fitness will be implemented by a simple function, there is a
macro `@fitness` which will produce a constant of type 
`FitnessFunction{T, F} <: AbstractFitness{T}` wrapping that function:

```julia
@fitness function fitness(x::Entity)
    -rosenbrock(x)
end
```

expands to

```julia
const fitness = FitnessFunction{Entity}(function ##fitness#2342(x::Entity)
    -rosenbrock(x)
end)
```

## Selection Operators

## Crossover Operators

## Mutation Operators

## Literature

Most useful is Luke (2013), as it contains a very consistent, detailed, but clear overview of very many common
algorithms, strategies, and operators.  Most of my implementations are quite literally taken from there.

- R. Poli, W. B. Langdon, N. F. McPhee, and J. R. Koza, A field guide to genetic programming. Morrisville: Lulu Press, 2008.
- S. Luke, Essentials of metaheuristics: a set of undergraduate lecture notes, Second edition, Online version 2.0. Morrisville: Lulu Press, 2013.
- H.-G. Beyer, “Evolution strategies,” Scholarpedia, vol. 2, no. 8, p. 1965, Aug. 2007.
- D. Dasgupta and Z. Michalewicz, Eds., Evolutionary Algorithms in Engineering Applications. Berlin, Heidelberg: Springer Berlin Heidelberg, 1997.
- I. Rechenberg, “Evolutionsprozesse,” in Simulationsmethoden in der Medizin und Biologie, B. Schneider and U. Ranft, Eds. Berlin: Springer, 1978, pp. 83--114.
- J. H. Holland, “Genetic algorithms,” Scholarpedia, vol. 7, no. 12, p. 1482, Dec. 2012.
- D. J. Montana, “Strongly typed genetic programming,” Evolutionary computation, vol. 3, no. 2, pp. 199–230, 1995.


