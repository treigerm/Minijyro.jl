using Flux
using LinearAlgebra: I # Identity matrix
using DistributionsAD
using Distributions

using Minijyro

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
        i += num_elements # TODO: Check for no off by one error.
    end
    Flux.loadparams!(nn, new_params)
end

@jyro function model(xs, neural_net, sigma)
    num_params = get_num_params(neural_net)
    w ~ MvNormal(num_params, 1.0)

    set_params!(neural_net, w)
    y ~ MvNormal(neural_net(xs), sigma*I)
end

function predict(m, w, xs)
    t = condition(m, Dict(:w => w))(xs)
    return t[:msgs][:y][:value]
end

# Generate some random data.
D = 2 # Dimensionality of data
N = 100 # Number of data points
sigma = 0.1 # Noise level TODO: Set a reasonable value for this.
dnn = Chain(Dense(D, 5, relu), Dense(5, 1), x -> vec(x))

xs = randn(D, N)
true_trace = trace(model)(xs, dnn, sigma)
ys = true_trace[:msgs][:y][:value]

# Do inference.
condition!(model, Dict(:y => ys))
trace!(model)

# TODO: Tune these values.
samples, stats = hmc(model, 0.05, 10, 1000, (xs, dnn, sigma), autodiff=:reverse); # TODO: name arguments.