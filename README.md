# Darwin

[![Build
Status](https://travis-ci.org/phipsgabler/Darwin.jl.svg?branch=master)](https://travis-ci.org/phipsgabler/Darwin.jl)

This package tries to implement evolutionary algorithms (or "population methods", see Luke, 2013)
with a very flexible interface.  Currently, only the [genetic
algorithm](http://www.scholarpedia.org/article/Genetic_algorithm) (GA) in its classic form is
supported, but I plan to add other variations and evolution strategies (ES).

## Background/Alternatives

There are two existing Julia packages that I know of which implement genetic algorithms,
[Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) and
[GeneticAlgorithms.jl](https://github.com/WestleyArgentum/GeneticAlgorithms.jl).  I decided to write
a completely new on since I didn't really like the interfaces of both (GeneticAlgorithms requires to
write a module and uses task functions for grouping, and Evolutionary is less flexible and exposes
much of internal specifics, IMHO).  

Instead, I wanted to have something like in the style
[DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl), with a very
flexible, high level interface and convenient control possibilities.  Originally I tried to
replicate their iterator interface, but then switched to the more general interface of
[LearningStrategies.jl](https://github.com/JuliaML/LearningStrategies.jl), which is explained below.

Additionally, I took efforts to implement other parts in a combinator style as well, e.g. genetic
operators, lifting of them, rate parameters, etc.

## Usage

In the style of `LearningStrategies`, you need to define define two things for optimizing a problem:

- A model, which contains the data and specification of the problem -- _what is be optimized_.  In
  this package, this will be a subtype of `AbstractEvolutionaryModel`, holding at least the
  population and fitness.
- A `LearningStrategy` which determines _how the model should be optimized_.  This corresponds to a
  certain algorithm, such as `GAStrategy`.  The strategy includes the genetic operators and internal
  values, e.g. caches.

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

(The presence of the unused `_d` is a leftover from `LearningStrategies`, but we usually have no
data be read iteratively in population methods).  The main part of a strategy consists the `update!`
function, in which the genetic operators are applied.  Termination through `finished` is mostly left
to meta-strategies like `LearningStrategies.MaxIter`, which can be combined with an evolutionary
algorithm in an arbitrary way:

```julia
strategy(Verbose(strat), MaxIter(g))
```

would be the most common one, specifying termination after `g` generations and printing stuff about
`strat` (see [here](https://github.com/JuliaML/LearningStrategies.jl#built-in-strategies) for about
other strategies).

### Types

Mostly everything is parametrized by a first type parameter `G` for the "things which should be
optimized" -- that is, the genome which is used.

Internally in models and such, not `G` itself is used, but a wrapper `Individual{G}`, which has an
additional field for caching the fitness of a genome.  Thus you don't have deal with fitness caching
yourself.

There are two other type synonyms related to `Individual`:

```julia
const Population{G} = Vector{Individual{G}}
const Family{G, N} = NTuple{N, Individual{G}}
```

Usually, the only place where you have to deal with `Individuals` is where you set up the initial
population for your problem, like the following:

```julia
Population(rand(Genome, N))
```

The type `G` should implement `rand`, `copy`, and have an `AbstractFitness{G}` object associated
with it.

#### Rate Parameters

The abstract type `Rate` should be used at places for rate parameters, such as tempering factors or
mutation rates.  An implementation of `Rate` should be callable on positive `Integer`s, representing
the rate at a certain generation.

There are three pre-provided schemes:

```julia
ConstantRate(α)
LinearRate(initial_rate, final_rate, slope)
ExponentialRate(initial_rate, final_rate, decay)
```

The first just returns the constant `α` all the time.  Linear and exponential rates start at their
`initial_rate`s, and decrease until `final_rate`.


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

Because we _iterate_ over the parent selection and crossed-over children, there's some flexibility
in how to divide and rebuild a population.  To be exact, the strategy for the GA is defined like
this:

```julia
mutable struct GAStrategy{G, P, K,
                          Fs<:SelectionOperator{>:G, P, K},
                          Fc<:CrossoverOperator{>:G, K, P},
                          Fm<:MutationOperator{>:G}} <: L.LearningStrategy
```
                          
The type parameters reflect the constraints on number of parents and children: 

- `G` is the type of genomes,
- `P` the ratio of population to families (parent tuples), or equivalently the number of individuals
  produced by a crossover,
- `K` the number of children produced from each selected family and consumed by a crossover operator.

Thus, a `SelectionOperator{G, 1, 2}` will select two parents for each population member.  Similarly,
a `SelectionOperator{G, 2, 1}` will select `length(population) ÷ 2` single parents, and
`GAStrategy{G, 2, 2}` as many pairs of parents (the last variant being the one used in most
literature, AFAIK).

These selected parents are then passed to a crossover operator of opposite characteristics,
producing `P` children out of the `K` parents.

This interface allows one to generalize the selection/crossover structure, while ensuring to keep
the population size constant (except for rounding errors when the populuation size is not a multiple
of `P`).

A simple GA run might thus look as follows:

```julia
initial_population = rand(Individual{Genome}, N)
model = GAModel(initial_population, fitness)
strat = strategy(Verbose(GAStrategy(selection, crossover, mutation)),
                 MaxIter(generations))
learn!(model, strat)
```

For this to work, you need to have some type `Genome` with `copy` and `rand` defined on it, a
fitness function of type `AbstractFitness`, and selection, crossover, and mutation operators of
suitable characteristics (with the numbers matching).


## Fitness Functions

Evolutionary models in general require a fitness function for assigning quality to the individuals
in question.  Since Julia functions are not parametrized by their input and output types, an
`AbstractFitness{G}` is required by this package, representing a fitness operator on genomes of type
`G`.  Such an operator should be callable on values of `G`.

The return type should be `Float64`, since that is what is cached in `Individual{G}` and can
represent any total order anyway.  In theory, it is enough to return something that can be compared
using `less` and converted to `Float64`.

Since in most cases, the fitness will be implemented by a simple function, there is a macro
`@fitness` which will produce a constant of type `FitnessFunction{G, F} <: AbstractFitness{G}`
wrapping that function:

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

Selection operators inherit from the abstract type `SelectionOperator{G, P, K}`, with the parameters
described above.  

To define your own selection operator, you need to define a type inheriting from `SelectionOperator`
with the right type parameters.  If your operator only works for certain kinds of genomes, or for
certain numbers `P` and `K`, you should make those constant in the inheriting clause.  You then have
to implement the actual selection by overloading the function `selection` with one of the following
signatures:

```julia
selection(population::Population{G}, operator::YourSelection{G, P, K}) where {G, P, K}
selection(population::Population{G}, operator::YourSelection{G, P, K}, generation::Integer) where {G, P, K}
```

The second method defaults to the first one and can be used if the generation is used in calculating
the selection (e.g. if you use some kind of rate parameter).  The `selection` function should return
an iterable of `Family{G, K}`, which is a `K`-`NTuple` of `Individual{G}` (which is already the type
of `population`).

Usually you can use a custom `YourSelectionIterator` for this, which implements the actual selection
logic, and have `selection` just set up this iterator.  In this way, you can in the iterator focus
on one `Family` at a time, which is easier to think about. As an example, the main parts of
`TournamentSelection` are implemented in such a style:

```julia
selection(population::Population{G}, operator::TournamentSelection{G, S, P, K}) where {G, S, P, K} = 
    TournamentSelectionIterator{S, P, K}(population)
    
function iterate(itr::TournamentSelectionIterator{M, S, P, K}, state = 0) where {M, S, P, K}
    if state ≥ M
        return nothing
    else
        ntuple(i -> maximumby(fitness, randview(itr.population, S)), Val{K}()), state + 1
    end
end
```

Note that the `ntuple` function is used with a `Val` parameter here.  `randview` is a custom
function returning a random view of size `S`.

If you need to perform some initialization of the operator with the model, you can overload
`setup!(operator, model)`; this is, for example, used to access the `fittest` property of a model in
`PairWithBestSelection`.

### Pair-With-Best Selection

```julia
PairWithBestSelection{G, P} <: SelectionOperator{G, P, 2}
PairWithBestSelection{G, P}()
```

Selects `length(population) ÷ P` 2-tuples of random individuals, paired with the currently best
individual (obtained from the `GAModel` through its `fittest` property).

### Fitness-Proportionate Selection

```
FitnessProportionateSelection{G, P, K, F} <: SelectionOperator{G, P, K}
```

Selects random `K`-tuples from the population, according to a categorical distribution over
individuals given by `t(fitness.(population))`, where `t(fs) = transform(fs,
temperature(generation))`.  `transform` needs to be a function turning a vector of arbitrary fitness
values into a discrete probability distribution, with `temperature` as a second parameter.

There are two pre-provided transforms with factory constructors:

```julia
SoftmaxSelection{G, P, K}(rate = 1.0)
L1Selection{G, P, K}()
```

which use the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) with temperature,
and rescaling by its sum (taking care of ties and negative values).

### [Tournament Selection](https://en.wikipedia.org/wiki/Tournament_selection)

```julia
TournamentSelection{G, S, P, K} <: SelectionOperator{G, P, K}
```

Selects `K` times the best of `S` randomly chosen individuals.


## Crossover Operators

Crossover operators inherit from the abstract type `CrossoverOperator{G, K, P}`, with the parameters
described above: the operator consumes `K` individuals and procudes `P` new ones.

To define your own crossover operator, you need to define a type inheriting from `CrossoverOperator`
with the right type parameters.  If your operator only works for certain kinds of genomes, or for
certain numbers `P` and `K`, you should make those constant in the inheriting clause.  You then have
to implement the actual crossover function by overloading `crossover` with one of the following
signatures:

```julia
crossover!(parents::NTuple{K, G}, operator::YourCrossover{G, K, P}, generation::Integer)
```

which should return an `NTuple{P, G}`.  This function is automatically lifted to `Family{K, G}`, so
you don’t have to care about lifting `Individual`s.

If you need to perform some initialization of the operator with the model, you can overload
`setup!(operator, model)`.

### Arithmetic Crossover

The operators

```julia
ArithmeticCrossover{G, K, P}(rate::Rate)
```

will produce, with probability `rate`, produce cross-wise convex combinations between the parents,
using an arbitrary mixing coefficient.  Currently, only `K == P == 2` is supported; this means that
the mixed result will be

```julia
((1 - mixing) .* parents[1] .+ mixing .* parents[2], (1 - mixing) .* parents[2] .+ mixing .* parents[1])
```

`G` must be a subtype of `AbstractVector` with elements supporting the necessary arithmetic.

### Uniform Crossover

The operators

```julia
UniformCrossover{G, K, P}(p::Rate = 0.5)
```

for an `AbstractVector` `G` will independently choose each child element among all parent elements
at the same index, depending on a Bernoulli choice with parameter `p`.  Currently, the variants
`UniformCrossover{G, 2, 1}` and `UniformCrossover{G, 2, 2}` are supported.

### Lifting

If your genome type is a wrapper `G` about an "inner genome" `IG`, e.g. an array, you can reuse
existing crossover operators by using

```julia
const YourCrossover = LiftedCrossover{G, SomeOperator{IG, K, P}}
```

and manually defining the lifted function like

```julia
function crossover!((p₁, p₂)::NTuple{2, G}, operator::YourCrossover, generation::Integer)
    G.(crossover!((p₁.inner_genome, p₂.inner_genome), operator.inner, generation))
end
```

where `inner_genome` is the field you actually want the mutation to happen on.


## Mutation Operators

Mutation operators inherit from the abstract type `MutationOperator{G}`, where `G` is the type of
genome that is mutated.  Every operator implementation consists of a subtype of `MutationOperator`,
possibly with a specialization of the genome type, and a method

```julia
mutate!(genome::Genome, operator::YourMutation{Genome}, generation::Integer)
```

(possibly parametric in `Genome`, depending on what kinds of things are supported).  This method
should execute the mutation in-place on the genome and return it (copying is done independently
before, by the strategy).

Mutation operators will likely involve `Rate` parameters, which should be called depending on
`generation`.  The implementation is automatically lifted to

```julia
mutate!(individual::Individual{Genome}, operator::YourMutation{Genome}, generation::Integer)
```

so you don't need to care about updating fitness and the like.

If you need to perform some initialization of the operator with the model, you can overload
`setup!(operator, model)`.

### Trivial mutation

In case you want to leave out mutation completely, there is ` NoMutation <: MutationOperator{Any}`
which just returns the genome as-is.

### Pointwise mutation

The operators

```julia
PointwiseMutation{G}(rate::Rate, tweak::Distribution)
```

for `G<:AbstractVector` will independently with probability `rate` replace element of an array
independently by a sample from `tweak`.

### Additive mutation/"convolution"

The operators

```julia
AdditiveMutation{G}(rate::Rate, tweak::Distribution, min, max)
```

for `G<:AbstractVector` will independently with probability `rate` modify each element of an array
by adding a sample of `tweak`, which must be a distribution with zero mean.

There are convenience constructors

```julia
AdditiveMutation{G, Uniform}(rate, r, min, max)
AdditiveMutation{G, Normal}(rate, σ, min, max)
AdditiveMutation{G, DiscreteUniform}(rate, r, min, max)
```

for specific distributions, which allow you to specify their scale parameters directly.

This operator generalizes what is called "bounded uniform" and "Gaussian convolution" in Luke
(2013).

### Bitflip mutation

The operator

```julia
BitFlipMutation(rate::Rate)
```

for `AbstractVector{Bool}` will independently with probability `rate` replace element of an array
independently its negation.

### Lifting

If your genome type is a wrapper `G` about an "inner genome" `IG`, e.g. an array, you can reuse
existing mutation operators by using

```julia
const YourMutation = LiftedMutation{G, SomeOperator{IG})
```

and manually defining the lifted mutation like

```julia
function mutate!(genome::G, operator::YourMutation, generation::Integer)
    mutate!(genome.inner_genome, operator.inner, generation)
    genome
end
```

where `inner_genome` is the field you actually want the mutation to happen on.



## Literature

Most useful is Luke (2013), as it contains a very consistent, detailed, but clear overview of very
many common algorithms, strategies, and operators.  Most of my implementations are quite literally
taken from there.

- R. Poli, W. B. Langdon, N. F. McPhee, and J. R. Koza, A field guide to genetic
  programming. Morrisville: Lulu Press, 2008.
- S. Luke, Essentials of metaheuristics: a set of undergraduate lecture notes, Second edition,
  Online version 2.0. Morrisville: Lulu Press, 2013.
- H.-G. Beyer, “Evolution strategies,” Scholarpedia, vol. 2, no. 8, p. 1965, Aug. 2007.
- D. Dasgupta and Z. Michalewicz, Eds., Evolutionary Algorithms in Engineering Applications. Berlin,
  Heidelberg: Springer Berlin Heidelberg, 1997.
- I. Rechenberg, “Evolutionsprozesse,” in Simulationsmethoden in der Medizin und Biologie,
  B. Schneider and U. Ranft, Eds. Berlin: Springer, 1978, pp. 83--114.
- J. H. Holland, “Genetic algorithms,” Scholarpedia, vol. 7, no. 12, p. 1482, Dec. 2012.
- D. J. Montana, “Strongly typed genetic programming,” Evolutionary computation, vol. 3, no. 2,
  pp. 199–230, 1995.


