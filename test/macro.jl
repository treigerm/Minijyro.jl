using DistributionsAD
using Distributions
using ReverseDiff

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

function density(w, data)
    data = convert(Dict{Any,Any}, data)
    data[:w] = w
    handle!(model, ConditionHandler(data))
    handle!(model, LogJointHandler())
    t = model(xs)
    return t[:logjoint]
end

#m = DensityModel(x -> density(x, data))
#spl = RWMH(Normal(0,1))
#chain = sample(m, spl, 2000; param_names=["w"], chain_type=Chains)

ForwardDiff.gradient(x -> density(x[1], data), [0.0])
