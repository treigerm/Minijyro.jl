# TODO: Macros to add to trace

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

# TODO: Does this need to be mutable?
# TODO: Maybe rename into something like RecordMessagesHandler
# TODO: do not save trace and stuff in inside trace handler but instead use the trace to save stuff
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
    trace[:logjoint] = trace[:logjoint] + logpdf(msg[:args][1], msg[:value])
end

struct ConditionHandler <: AbstractHandler
    data::Dict
end

function process_message!(trace::Dict, h::ConditionHandler, msg::Dict)
    if haskey(h.data, msg[:name])
        msg[:value] = h.data[msg[:name]]
        msg[:stop] = true
    end
end

# TODO: Queue effect handler.

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

function queue(model::MinijyroModel, queue)
    # TODO: Make sure this function does not change model argument but creates new one.
    max_tries = 10 # TODO: Do we need this?
    function _fn(handlers_stack, args...)
        hstack = deepcopy(handlers_stack)
        for i in 1:max_tries
            top_trace = pop!(queue)
            try
                # TODO: How to make sure that the handlers are going to be removed?
                # NOTE: If we define custom handler functions this should work.
                push!(hstack, ReplayHandler(top_trace))
                discr_esc(msg) = !haskey(top_trace[:msgs], msg[:name])
                push!(hstack, EscapeHandler(discr_esc))
                push!(hstack, TraceHandler())

                full_trace = model.model_fn(hstack, args...)
                # We only get to the return call if there has been no escape i.e.
                # if the top_trace has been a full trace.
                return full_trace
            catch e
                if isa(e, EscapeException)
                    for val in support(e.trace[:msgs][e.name][:args][1])
                        new_trace = deepcopy(e.trace)
                        new_trace[:msgs][e.name][:value] = val
                        new_trace[:msgs][e.name][:done] = false
                        new_trace[:msgs][e.name][:continuation] = nothing
                        push!(queue, new_trace)
                    end
                else
                    throw(e)
                end
            finally
                # TODO: How to get rid of this?
                pop!(hstack)
                pop!(hstack)
                pop!(hstack)
            end
        end
    end

    return MinijyroModel(model.handlers_stack, _fn)
end
