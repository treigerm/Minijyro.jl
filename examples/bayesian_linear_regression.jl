using Distributions
using LinearAlgebra: I # Identity matrix
using Random

using Minijyro

Random.seed!(42)

# Generate some data.
N = 100
D = 5
true_w = randn(D)
X = randn(N, D)
noise = 0.1 * randn(N)
y_obs = X * true_w + noise

@jyro function model(xs)
    D = size(xs)[2]
    w ~ MvNormal(zeros(D), I)
    y ~ MvNormal(xs * w, 0.1*I)
end

cond_model = condition(model, Dict(:y => y_obs))
samples, stats = nuts(cond_model, (X,), 1000)

@show abs.(true_w - mean(samples))
