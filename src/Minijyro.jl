module Minijyro

using Distributions

# TODO: Do we want to export the enter!, exit!, sample! and apply_stack!?
# TODO: Reexport Distributions, DistributionsAD and AutoDiff
export sample!,
    apply_stack!,
    AbstractHandler,
    TraceHandler,
    LogJointHandler,
    ConditionHandler,
    ReplayHandler,
    EscapeHandler,
    EscapeException,
    queue,
    enter!,
    exit!,
    @jyro,
    handle!,
    handle,
    discrete_enumeration,
    hmc,
    log_prob


include("minijyro_types.jl")
include("dsl.jl")
include("handlers.jl")
include("inference.jl")


function sample!(trace::Dict, handlers_stack::Array{AbstractHandler,1}, name::Any, dist::Distributions.Distribution)
    if length(handlers_stack) == 0
        return rand(dist)
    end

    # TODO: What are the fields we really need for our use case.
    initial_msg = Dict(
        :fn => rand,
        :args => (dist, ),
        :name => name,
        :value => nothing,
        :is_observed => false
    )
    msg = apply_stack!(trace, handlers_stack, initial_msg)
    return msg[:value]
end

function apply_stack!(trace::Dict, handlers_stack::Array{AbstractHandler,1}, msg::Dict)
    pointer = 1
    for (p, handler) in enumerate(handlers_stack[end:-1:1])
        pointer = p
        process_message!(trace, handler, msg)
        if get(msg, :stop, false)
            break
        end
    end

    if !(get(msg, :value, nothing) != nothing || get(msg, :done, false))
        msg[:value] = msg[:fn](msg[:args]...)
    end

    for handler in handlers_stack[end-pointer+1:end]
        postprocess_message!(trace, handler, msg)
    end

    if get(msg, :continuation, nothing) != nothing
        msg[:continuation](trace, msg)
    end

    return msg
end
end # module
