# Minijyro

A simple probabilistic programming language in Julia based on effect handlers.
The name comes from the fact that this project is largely based on the ideas from
[Pyro's effect handlers](http://pyro.ai/examples/effect_handlers.html)
and their [Mini-Pyro implementation](http://pyro.ai/examples/minipyro.html).

The design goals of this language are:

- Allow for concise definition of sample statements using `~` syntax
- Use effect handlers to implement simple operations such as conditioning and
    computing the log joint probability
- Leverage existing Julia packages such as [Distributions.jl](https://github.com/JuliaStats/Distributions.jl),
    [AdvancedHMC](https://github.com/TuringLang/AdvancedHMC.jl) and
    [Flux](https://github.com/FluxML/Flux.jl)

**NOTE**: This is not meant to be a serious PPL to be used by anyone. If you are
interested in probabilistic programming in Julia have a look at
[Turing.jl](https://github.com/TuringLang/Turing.jl),
[Gen](https://github.com/probcomp/Gen) and
[Soss.jl](https://github.com/cscherrer/Soss.jl).

## Example: Bayesian Linear Regression

A simple model taken from [Colin Caroll's tour of PPL APIs](https://colcarroll.github.io/ppl-api/).

```julia
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
```
