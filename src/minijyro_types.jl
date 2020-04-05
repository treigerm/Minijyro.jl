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

"""
    MinijyroModel

A generative function with associated effect handlers which are stored in
`handlers_stack`. A `MinijyroModel` can be called like a normal Julia function
and will execute the `model_fn` with the effect handlers in `handlers_stack`.

`MinijyroModel`s should be constructed using the `@jyro` macro.

# Fields
- `handlers_stack::Array{AbstractHandler,1}`: the stack of handlers applied to the
    model
- `model_fn::Function`: generated function
"""
struct MinijyroModel
    handlers_stack::Array{AbstractHandler,1}
    model_fn::Function
end

function (model::MinijyroModel)(args...)
    # TODO: Possibly give nicer error message when args are given in wrong type.
    return model.model_fn(model.handlers_stack, args...)
end

Base.copy(m::MinijyroModel) = MinijyroModel(copy(m.handlers_stack), m.model_fn)
