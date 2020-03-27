using DistributionsAD
using Distributions
using ForwardDiff
using AdvancedHMC

using Minijyro

#using AdvancedMH
#using MCMCChains

@jyro function model(xs::Vector{Float64})
    w ~ Normal(0, 1)
    for (i, x) in enumerate(xs)
        {:y => i} ~ Normal(w * x, 1)
    end
end

# Generate some data
true_w = 0.5
xs = convert(Vector{Float64}, collect(1:10))
ys = true_w .* xs
data = Dict((:y => i) => y for (i, y) in enumerate(ys))

condition!(model, data)
samples, stats = hmc(model, 0.05, 10, 1000, (xs,))
