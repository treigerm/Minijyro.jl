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

struct MinijyroModel
    handlers_stack::Array{AbstractHandler,1}
    model_fn::Function
end

function (model::MinijyroModel)(args...)
    # TODO: Possibly give nicer error message when args are given in wrong type.
    return model.model_fn(model.handlers_stack, args...)
end

Base.copy(m::MinijyroModel) = MinijyroModel(copy(m.handlers_stack), m.model_fn)
