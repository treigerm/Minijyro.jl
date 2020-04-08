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

## Behind the Scenes

Here is a high-level overview of the inner workings of Minijyro. For more details
I recommend first reading through the links to the Pyro documentation from above
and then through full the source code of Minijyro.

Each `~` expression is translated into a call to

```julia
function sample!(
    trace::Dict,
    handlers_stack::Array{AbstractHandler,1},
    name::Any,
    dist::Distributions.Distribution
)
    if length(handlers_stack) == 0
        return rand(dist)
    end

    initial_msg = Dict(
        :fn => rand,
        :args => (dist, ),
        :name => name,
        :value => nothing,
        :is_observed => false,
        :done => false,
        :stop => false,
        :continuation => nothing
    )
    msg = apply_stack!(trace, handlers_stack, initial_msg)
    return msg[:value]
end
```

See [dsl.jl](https://github.com/treigerm/Minijyro.jl/blob/master/src/dsl.jl) for
the full implementation of the `@jyro` macro.
`apply_stack!` is used to apply all effect handlers that are active at
the given sample site:

```julia
function apply_stack!(
    trace::Dict,
    handlers_stack::Array{AbstractHandler,1},
    msg::Dict
)
    @assert length(handlers_stack) > 0

    handler_counter = 0
    # Loop through handlers from top of the stack to the bottom.
    for handler in handlers_stack[end:-1:1]
        handler_counter += 1
        process_message!(trace, handler, msg)
        if msg[:stop]
            break
        end
    end

    if !(msg[:value] != nothing || msg[:done])
        msg[:value] = msg[:fn](msg[:args]...)
    end

    # Loop through handlers from bottom of the stack to the top.
    # If we exited the first loop early then we will start looping from the
    # handler which caused the loop to exit.
    for handler in handlers_stack[end-handler_counter+1:end]
        postprocess_message!(trace, handler, msg)
    end

    if msg[:continuation] != nothing
        msg[:continuation](trace, msg)
    end

    return msg
end
```

Effect handlers are subtypes of `AbstractHandler`:

```julia
abstract type AbstractHandler end

function enter!(trace::Dict, h::AbstractHandler)
    return
end

function exit!(trace::Dict, h::AbstractHandler)
    return
end

function process_message!(trace::Dict, h::AbstractHandler, msg::Dict)
    return
end

function postprocess_message!(trace::Dict, h::AbstractHandler, msg::Dict)
    return
end
```

For example, conditioning on data can be implemented as:

```julia
struct ConditionHandler <: AbstractHandler
    data::Dict
end

function process_message!(trace::Dict, h::ConditionHandler, msg::Dict)
    if haskey(h.data, msg[:name])
        msg[:value] = h.data[msg[:name]]
        msg[:stop] = true
        msg[:is_observed] = true
    end
end
```
