# Darwin

[![Build
Status](https://travis-ci.org/phipsgabler/Darwin.jl.svg?branch=master)](https://travis-ci.org/phipsgabler/Darwin.jl)

This package tries to implement evolutionary algorithms with a very flexible interface.  Currently,
only [genetic algorithms](http://www.scholarpedia.org/article/Genetic_algorithms) in their classic
form are supported.

## Usage

### Genetic Algorithms

At the simplest, you specify a `GAModel` and call `evolve` on it:

```julia
model = GAModel(initial_population, selection, crossover, mutate!, 2)
result = evolve(model, 100; verbose = true)
```

Now `result` will contain a `GAResult` with a field `population` for the population of the last
generation, which hopefully will contain very fit individuals; you can check that with

```julia
@show fitness.(result.population)
```

The model is specified by an initial population, which is an `AbstractArray` of usually randomly
chosen individuals.  `selection`, `crossover`, and `mutate!` are functions (or at least callables)
with the following conventions:

- `selection` takes the current population and returns a vector of indices which are chosen to
  breed.  This is the place where you should use the fitness functions of the problem.
  The size of the selection should be `matingfactor`. 
- `crossover` receives a choice of parents of size `matingfactor` and should perform whatever you
  want for breeding.  It should not modify the parents, but produce new individuals (so if you don't
  want to do any crossover, at least copy the parents!).
- `mutate!` should mutate one individual in-place.

Note that you don't have to explicitely give fitness as an argument to the model -- it is only
implicitely used in `selection`, where you usually will select fitter individuals more likely.

In this way, randomness is completely in the hand of the user -- if you want to make an experiment
repeatable, choose a fixed seed for these three functions.  Also, since evaluation of fitness is not
explicitely handled by internal functions, you are free to, e.g., parallelize it in your own ways.

### Evolver interface

This is inspired by the DiffEq [integrator
interface](http://docs.juliadiffeq.org/stable/basics/integrator.html).  If you want to have finer
control over the evolution, you can create a `GAEvolver` by `init(model)`, and then perform the
steps manually by calling `evolvestep!` repeatedly.  The `GAEvolver` contains all current
information, especially in the fields `generation` and `solution`, the latter of which holds the
current population.

`GAEvolver` is iterable, so, for example, to evolve for 100 steps, printing the best fitness each
time, you can do the following:

```julia
evolver = init(model)
for step in take(evolver, 100)
	println(maximum(fitness.(evolver.solution.population)))
end
```

(There is a slight caveat in that iterating is mutable, and the `state` argument of the interface is
essentially a dummy; but you can at any time create a fresh evolver by calling `init` again.)

## Background/Alternatives

There are two existing Julia packages that I know of which implement genetic algorithms,
[Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) and
[GeneticAlgorithms.jl](https://github.com/WestleyArgentum/GeneticAlgorithms.jl).  I decided to write
a completely new on since I didn't really like the interfaces of both (GeneticAlgorithms requires to
write a module and uses task functions for grouping, and Evolutionary is less flexible and exposes
much of internal specifics, IMHO).  Instead, I wanted to have something like in the style
[DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl), with a very
flexible, hight level interface and convenient control possibilities.

Note that this is only a draft.  I don't claim to do things right.


## ToDo

- Testing
- Illustrated examples with notebooks
- Multiple callbacks (like `CallbackSet` for DiffEq), `terminate!` function for the evolver
- More flexible interface with `selection`/`crossover`, avoiding specification of `matingfactor`
- Predefined operators for selection, mutation and crossover
- Evolution strategies
