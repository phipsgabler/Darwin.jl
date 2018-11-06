abstract type Fitness end

setup!(f::Fitness) = f

## can't do this because of https://github.com/JuliaLang/julia/issues/14919
# """
#     (f::Fitness)(individual[, generation]) -> number

# Calculate fitness of `individual`, returning a number.  Can do caching internally.
# """
# (f::F)(individual) where {F<:Fitness} = NaN
# (f::F)(individual, generation) where {F<:Fitness} = f(individual)
