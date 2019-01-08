import Base: minimum, argmin

abstract type TestFunction{T, D} <: Function end

# @generated function pack(f::TestFunction{T, D}, V::Type{<:AbstractVector{T}}) where {T, D}
#     quote
#         function $(Symbol(nameof(f), "_wrapper"))(x::V)
#             f($((:(x[$i]) for i = 1:D)...))
#         end
#     end
# end


function pack(f::TestFunction{T, D}, ::Type{V}) where {T, D, V<:AbstractVector{T}}
    function (x::V)
        f(x...)
    end
end


# https://en.wikipedia.org/wiki/Rosenbrock_function
struct Rosenbrock{T<:AbstractFloat} <: TestFunction{T, 2}
    a::T
    b::T

    Rosenbrock{T}(; a::T = 1.0, b::T = 100.0) where T = new{T}(a, b)
end

(r::Rosenbrock{T})(x::T, y::T) where T = (r.a - x)^2 + r.b * (y - x^2)^2
bounds(::Rosenbrock{T}) where T = (-2one(T), 2one(T))
minimum(::Rosenbrock{T}) where T = zero(T)
argmin(r::Rosenbrock{T}) where T = [r.a, r.a^2]


struct Rastrigin{T, D} <: TestFunction{T, D} end

(r::Rastrigin{T, D})(x::Vararg{T, D}) where {T, D} = 10D + sum(x .^ 2 .- 10 .* cospi.(2 .* x))
bounds(::Rastrigin{T}) where T = (-6one(T), 6one(T))
minimum(::Rastrigin{T}) where T = zero(T)
argmin(r::Rastrigin{T, D}) where {T, D} = fill(zero(T), D)


struct Ackley{T, D} <: TestFunction{T, D}
    a::T
    b::T
    c::T

    Ackley{T, D}(; a::T = 20.0, b::T = 0.2, c::T = 2π) where {T, D} = new{T, D}(a, b, c)
end

(r::Ackley{T, D})(x::Vararg{T, D}) where {T, D} =
    -r.a * exp(-r.b * sqrt(sum(x .^ 2) / D)) - exp(sum(cos.(r.c .* x)) / D) + r.a + exp(1)
bounds(::Ackley{T}) where T = (-32.768one(T), 32.768one(T))
minimum(::Ackley{T}) where T = zero(T)
argmin(r::Ackley{T, D}) where {T, D} = fill(zero(T), D)


struct Chasm{T} <: TestFunction{T, 2} end

(r::Chasm{T})(x::T, y::T) where T = 10^3 * abs(y) / (10^3 * abs(x) + 1) + 10^(-2) * abs(y)
bounds(::Chasm{T}) where T = (-5one(T), 5one(T))
minimum(::Chasm{T}) where T = zero(T)
argmin(r::Chasm{T}) where T = [-one(T), zero(T)]


# https://en.wikipedia.org/wiki/Himmelblau%27s_function
# This function has _four_ local minima!
struct Himmelblau{T<:AbstractFloat} <: TestFunction{T, 2}
    a::T
    b::T

    Himmelblau{T}(; a::T = 1.0, b::T = 100.0) where T = new{T}(a, b)
end

(r::Himmelblau{T})(x::T, y::T) where {T} = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
bounds(::Himmelblau{T}) where {T} = (-6one(T), 6one(T))
minimum(::Himmelblau{T}) where {T} = zero(T)



# Other resources:
# https://www.sfu.ca/~ssurjano/optimization.html
# https://en.wikipedia.org/wiki/Test_functions_for_optimization
