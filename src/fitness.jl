export assess!,
    AbstractFitness,
    # @fitness,
    FitnessFunction,
    setup!

abstract type AbstractFitness{T} <: Function end

setup!(f::AbstractFitness) = f


"""
    assess!(individual, f::AbstractFitness) -> FitnessValue

Calculate fitness of `individual`, returning a `FitnessValue`.  Stores the result of assessment in 
the individual, so the actual calculation is done at most once.
"""
function assess!(individual::Individual{T}, f::AbstractFitness{>:T}) where {T}
    if individual.fitnessvalue === nothing
        individual.fitnessvalue = f(individual.genome)
    end

    individual.fitnessvalue
end


struct FitnessFunction{T, F} <: AbstractFitness{T}
    evaluate::F

    FitnessFunction{T}(f::F) where {T, F} = new{T, F}(f)
end

(f::FitnessFunction{T})(genome::T) where {T} = f.evaluate(genome)


# macro fitness(fundef)
    # :(FitnessFunction{$(<gettype(fundef)>)}($(esc(fundef))))
# end


## can't do this because of https://github.com/JuliaLang/julia/issues/14919
# """
#     (f::Fitness)(individual[, generation]) -> number

# Calculate fitness of `individual`, returning a number.
# """
# (f::F)(individual) where {F<:Fitness} = NaN
# (f::F)(individual, generation) where {F<:Fitness} = f(individual)
