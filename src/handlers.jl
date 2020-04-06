"""
    TraceHandler

Saves the messages from each sample site in a dictionary with the keys as the
name of the sample site. All the messages are saved in `trace[:msgs]`.
"""
struct TraceHandler <: AbstractHandler end

function enter!(trace::Dict, h::TraceHandler)
    trace[:msgs] = Dict()
end

function postprocess_message!(trace::Dict, h::TraceHandler, msg::Dict)
    trace[:msgs][msg[:name]] = copy(msg)
end

"""
    LogJointHandler

Computes the sum of the log densities of all sample sites. The result is saved
in trace[:logjoint].
"""
struct LogJointHandler <: AbstractHandler end

function enter!(trace::Dict, h::LogJointHandler)
    trace[:logjoint] = 0.0
end

function postprocess_message!(trace::Dict, h::LogJointHandler, msg::Dict)
    dist = msg[:args][1]
    if msg[:value] != nothing
        trace[:logjoint] = trace[:logjoint] + logpdf(dist, msg[:value])
    else
        # TODO: Warning might be annoying when running the queue handler.
        @warn "Encountered site with value nothing." msg[:name]
    end
end

"""
    ConditionHandler

Conditions the model on the given data. This means that the sample sites
included in `data` will be recorded as observed and the model will always
generate traces which agree with the data.

# Fields
- `data::Dict`: data to condition on saved as `{site_name => value}`
"""
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

"""
    ReplayHandler

Makes sure the model generates the same value as given in `trace`. Only the sites
that do not appear in `trace` will be randomly sampled.

# Fields
- `trace::Dict`: trace with values used for replay
"""
struct ReplayHandler <: AbstractHandler
    trace::Dict
end

function process_message!(trace::Dict, h::ReplayHandler, msg::Dict)
    if haskey(h.trace[:msgs], msg[:name])
        msg[:value] = h.trace[:msgs][msg[:name]][:value]
    end
end

"""
    EscapeHandler

Interrupts the execution of model when `escape_fn` evaluates to `true`. Specifically,
an `EscapeException` is thrown to interrupt the execution.

# Fields
- `escape_fn::Function`: function that takes a message as an argument and returns
    `true` at the sample site to escape from
"""
struct EscapeHandler <: AbstractHandler
    escape_fn::Function
end

"""
    EscapeException

Raised by `EscapeHandler`.

# Fields
- `trace::Dict`: trace of the model execution that has been interrupted
- `name::Any`: name of the sample site that caused interruption
"""
struct EscapeException <: Exception
    trace::Dict
    name::Any
end

function process_message!(trace::Dict, h::EscapeHandler, msg::Dict)
    if h.escape_fn(msg)
        msg[:done] = true
        cont(t, m) = throw(EscapeException(t, m[:name]))
        msg[:continuation] = cont
    end
end

"""
    handle!(model::MinijyroModel, handler::AbstractHandler)

Pushes the `handler` on top of the stack of `model`.

# Arguments
- `model::MinijyroModel`
- `handler::AbstractHandler`
"""
function handle!(model::MinijyroModel, handler::AbstractHandler)
    push!(model.handlers_stack, handler)
end

"""
    handle(model::MinijyroModel, handler::AbstractHandler)

Same as `handle!` but does not mutate the `model` and instead returns a new
model with the added `handler`.

# Arguments
- `model::MinijyroModel`
- `handler::AbstractHandler`

# Returns
- `MinijyroModel`
"""
function handle(model::MinijyroModel, handler::AbstractHandler)
    # Same has handle! but do not alter original model.
    m = copy(model)
    push!(m.handlers_stack, handler)
    return m
end

# Generated convenience functions
handlers = [
    :TraceHandler,
    :LogJointHandler,
    :ConditionHandler,
    :ReplayHandler,
    :EscapeHandler
]

function get_function_name(type_name::Symbol)
    # Takes handler type and returns an adequate function name
    # Example: LogJointHandler -> log_joint
    s = String(type_name)
    prefix = split(s, "Handler")[1]
    regex = r"([a-z])([A-Z]+)"
    subsititution = s"\1_\2"
    fn_name = lowercase(replace(prefix, regex => subsititution))
    return Symbol(fn_name)
end

for h in handlers
    # TODO: Can we do kwargs as well?
    h_fn_name = get_function_name(h)
    # Function name for when we do mutation.
    h_fn_name_mut = Symbol(h_fn_name, "!")
    @eval begin
        export $(h_fn_name_mut), $(h_fn_name)

        function $(h_fn_name_mut)(model, args...)
            handle!(model, $(h)(args...))
        end

        function $(h_fn_name)(model, args...)
            return handle(model, $(h)(args...))
        end
    end
end

"""
    queue(model::MinijyroModel, queue::Array{Dict{Any,Any},1})

Will replay the model with the first trace in `queue`. Once we encounter a
discrete sample site which was not in the trace the function execution is
interrupted and for each value in the support of the new sample site we create
a new trace from the old one and add it to the queue.

This effect handler can be used to perform breadth-first enumeration of all
possible traces.

# Arguments
- `model::MinijyroModel`: model
- `queue::Array{Dict{Any,Any},1}`: array of traces

# Returns
- `MinijyroModel`
"""
function queue(model::MinijyroModel, queue::Array{Dict{Any,Any},1})
    # NOTE: This type of effect handler is actually more powerful because we can
    #       change the model_fn.
    max_tries = Int(1e6) # Make sure we do not have infinite loops.
    function _fn(handlers_stack, args...)
        # TODO: This cannot deal with the fact that model_fn might have been
        # been changed. Alternative is to pass model to the model_fn.
        m = MinijyroModel(handlers_stack, model.model_fn)
        for i in 1:max_tries
            top_trace = popfirst!(queue)
            try
                queue_m = handle(m, ReplayHandler(top_trace))
                # Escape when we see a new variable that is not in the trace.
                discr_esc(msg) = !haskey(top_trace[:msgs], msg[:name])
                queue_m = handle(queue_m, EscapeHandler(discr_esc))
                queue_m = handle(queue_m, TraceHandler())

                full_trace = queue_m(args...)
                # We only get to the return call if there has been no escape i.e.
                # if the top_trace has been a full trace.
                return full_trace
            catch e
                if isa(e, EscapeException)
                    for val in support(e.trace[:msgs][e.name][:args][1])
                        new_trace = deepcopy(e.trace)
                        new_trace[:msgs][e.name][:value] = val

                        # Make sure that we do not escape again when using this
                        # trace for replay.
                        new_trace[:msgs][e.name][:done] = false
                        new_trace[:msgs][e.name][:continuation] = nothing
                        push!(queue, new_trace)
                    end
                else
                    throw(e)
                end
            end
        end
    end

    return MinijyroModel(model.handlers_stack, _fn)
end
