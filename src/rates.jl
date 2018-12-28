export ConstantRate,
    ExponentialRate,
    LinearRate,
    Rate


abstract type Rate end


struct ConstantRate <: Rate
    α::Float64
end

(rate::ConstantRate)(::Integer) = rate.α


struct LinearRate <: Rate
    initial_rate::Float64
    final_rate::Float64
    slope::Float64
end

(rate::LinearRate)(generation::Integer) =
    max(rate.initial_rate - generation * rate.slope, rate.final_rate)


struct ExponentialRate <: Rate
    initial_rate::Float64
    final_rate::Float64
    decay::Float64
end

(rate::ExponentialRate)(generation::Integer) =
    max(rate.initial_rate * rate.decay ^ (generation - 1), rate.final_rate)
