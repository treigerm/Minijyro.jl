using Flux
using LinearAlgebra: I # Identity matrix
using Distributions
using Random
using Plots

using Minijyro

Random.seed!(42)
theme(:ggplot2)

# Convenience functions
function get_num_params(nn)
    # For each parameter vector count the number of elements and collect the total
    # sum.
    return mapreduce(p -> prod(size(p)), +, Flux.params(nn))
end

function set_params!(nn, weights)
    new_params = Flux.params([])
    i = 1
    for p in Flux.params(nn)
        num_elements = prod(size(p))
        new_p = weights[i:i+num_elements-1]
        push!(new_params, reshape(new_p, size(p)))
        i += num_elements
    end
    Flux.loadparams!(nn, new_params)
end

@jyro function model(xs, neural_net, sigma)
    num_params = get_num_params(neural_net)
    w ~ MvNormal(num_params, 1.0)

    set_params!(neural_net, w)
    y ~ MvNormal(neural_net(xs), sigma*I)
end

D = 1 # Dimensionality of data
N = 100 # Number of data points
sigma = 0.1 # Noise level
dnn = Chain(Dense(D, 5, tanh), Dense(5, 1), x -> vec(x))

# Create some data
xs = collect(range(-3, stop=3, length=N))
xs = reshape(xs, (D,N))
ys = cos.(xs) + sigma * randn(D,N)
ys = reshape(ys, (N,))

display(scatter(xs[1,:], ys, grid=:none))

# Run NUTS sampler
cond_model = condition(model, Dict(:y => ys))
trace!(cond_model)
samples, stats = nuts(cond_model, (xs, dnn, sigma), 2000; autodiff=:reverse, progress=true)

# Plot posterior samples
p = plot(xlims=(-10,10))
for posterior_sample in samples[end-500:end]
    wide_xs = range(-10, stop=10, length=400)
    wide_xs = reshape(wide_xs, (1,400))
    set_params!(dnn, posterior_sample)
    preds = dnn(wide_xs)
    plot!(p, wide_xs[1,:], preds, color=:red, alpha=0.3)
end
display(scatter!(p, xs[1,:], ys, color=:blue, grid=:none, legend=false))
