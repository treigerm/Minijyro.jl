struct TraceHandler <: AbstractHandler end

function enter!(trace::Dict, h::TraceHandler)
    trace[:msgs] = Dict()
end

function postprocess_message!(trace::Dict, h::TraceHandler, msg::Dict)
    trace[:msgs][msg[:name]] = copy(msg)
end

struct LogJointHandler <: AbstractHandler end

function enter!(trace::Dict, h::LogJointHandler)
    trace[:logjoint] = 0.0
end

function postprocess_message!(trace::Dict, h::LogJointHandler, msg::Dict)
    dist = msg[:args][1]
    # TODO: What to do if value is nothing.
    if msg[:value] != nothing
        trace[:logjoint] = trace[:logjoint] + logpdf(dist, msg[:value])
    end
end

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

struct ReplayHandler <: AbstractHandler
    trace::Dict
end

function process_message!(trace::Dict, h::ReplayHandler, msg::Dict)
    if haskey(h.trace[:msgs], msg[:name])
        msg[:value] = h.trace[:msgs][msg[:name]][:value]
    end
end

struct EscapeHandler <: AbstractHandler
    escape_fn::Function
end

struct EscapeException <: Exception
    trace::Dict
    name
end

function process_message!(trace::Dict, h::EscapeHandler, msg::Dict)
    if h.escape_fn(msg)
        msg[:done] = true
        cont(t, m) = throw(EscapeException(t, m[:name]))
        msg[:continuation] = cont
    end
end

# TODO: Maybe move this into handlers.jl?
function handle!(model::MinijyroModel, handler)
    # TODO: Type for handler.
    push!(model.handlers_stack, handler)
end

function handle(model::MinijyroModel, handler)
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
    h_fn_name = get_function_name(h)
    # TODO: Can we do kwargs as well?
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

function queue(model::MinijyroModel, queue)
    max_tries = Int(1e6) # TODO: Do we need this?
    function _fn(handlers_stack, args...)
        # TODO: This cannot deal with the fact that model_fn might have been
        # been changed. Alternative is to pass model to the model_fn.
        m = MinijyroModel(handlers_stack, model.model_fn)
        for i in 1:max_tries
            top_trace = popfirst!(queue)
            try
                queue_m = handle(m, ReplayHandler(top_trace))
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
