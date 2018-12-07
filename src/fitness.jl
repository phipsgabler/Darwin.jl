using MacroTools
import MacroTools: @q, combinearg, combinedef

export assess!,
    AbstractFitness,
    @fitness,
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



macro fitness(fundef::Expr)
    def = splitdef(fundef)
    @assert (length(def[:args]) == 1) "Only single argument functions allowed!"
    argname, argtype, slurp, default = splitarg(def[:args][1])

    fname = def[:name]
    def[:name] = gensym(def[:name])
    # def[:args] = [combinearg(argname, esc(argtype), slurp, default)]
    fun = combinedef(def)
    
    return @q const $(esc(fname)) = FitnessFunction{$(esc(argtype))}($(esc(fun)))
end


## can't do this because of https://github.com/JuliaLang/julia/issues/14919
# """
#     (f::Fitness)(individual[, generation]) -> number

# Calculate fitness of `individual`, returning a number.
# """
# (f::F)(individual) where {F<:Fitness} = NaN
# (f::F)(individual, generation) where {F<:Fitness} = f(individual)
