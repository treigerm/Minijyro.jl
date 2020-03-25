module Minijyro

using Distributions

# TODO: Do we want to export the enter! and exit! functions?
export sample!,
    apply_stack!,
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
    handle!

include("handlers.jl")
include("dsl.jl")

function sample!(trace, handlers_stack, name, dist)
    # TODO: Type annotations.
    if length(handlers_stack) == 0
        return rand(dist)
    end

    # TODO: Does this need to be a dict?
    # TODO: What are the fields we really need for our use case.
    initial_msg = Dict(
        :fn => rand,
        :args => (dist, ),
        :name => name,
        :value => nothing
    )
    msg = apply_stack!(trace, handlers_stack, initial_msg)
    return msg[:value]
end

function apply_stack!(trace, handlers_stack, msg)
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

    if length(handlers_stack) === 0
        # TODO: Can we make this more pretty?
        # NOTE: We should not need to have to deal with the case that handlers_stack is empty
        return msg
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
