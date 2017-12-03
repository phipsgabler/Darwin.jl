# Darwin

[![Build
Status](https://travis-ci.org/phipsgabler/Darwin.jl.svg?branch=master)](https://travis-ci.org/phipsgabler/Darwin.jl)

## Installation

## Background/Alternatives

There are two existing Julia packages that I know of which implement genetic algorithms,
[Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) and
[GeneticAlgorithms.jl](https://github.com/WestleyArgentum/GeneticAlgorithms.jl).  I decided to write
a completely new on since I didn't really like the interfaces of both (GeneticAlgorithms requires to
write a model and uses task functions, and Evolutionary is less flexible and exposes much of
internal specifics, IMHO).  Instead, I wanted to have something like in the style
[DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl), with a very
flexible, hight level interface and convenient control possibilities.

Note that this is only a draft.  I don't claim to do things right.


## ToDo

- Testing
- Illustrated examples with notebooks
- Predefined operators for selection, mutation and crossover
- Evolution strategies
