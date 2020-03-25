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

D = 1
initial_w = [0.0]

n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, x -> density(x[1], data), ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_eps = find_good_eps(hamiltonian, initial_w)
integrator = Leapfrog(initial_eps)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(Preconditioner(metric), NesterovDualAveraging(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, initial_w, n_samples, adaptor, n_adapts; progress=true)
