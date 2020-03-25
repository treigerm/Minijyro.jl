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
# TODO: Escape effect handler. Requires changing apply_stack to have continuiation.
# TODO: Replay effect handler.

struct ReplayHandler <: AbstractHandler
    trace::Dict
end

function process_message!(trace::Dict, h::ReplayHandler, msg::Dict)
    if haskey(h.trace[:msgs], msg[:name])
        msg[:value] = h.trace[:msgs][msg[:name]][:value]
    end
end
