module Minijyro

using Distributions

export sample!,
    MinijyroModel,
    AbstractHandler,
    TraceHandler,
    LogJointHandler,
    ConditionHandler,
    ReplayHandler,
    EscapeHandler,
    EscapeException,
    queue,
    @jyro,
    handle!,
    handle,
    discrete_enumeration,
    hmc,
    nuts,
    enter!,
    exit!


include("minijyro_types.jl")
include("dsl.jl")
include("handlers.jl")
include("inference.jl")


"""
    sample!(args...)

Allows sampling from probability distributions with side effects specified by
effect handlers in `handlers_stack`.

# Arguments
- `trace::Dict`: dict that can be used to store things in a global state
- `handlers_stack::Array{AbstractHandler,1}`: stack of effect handlers
- `name::Any`: unique name for the sample site
- `dist::Distributions.Distribution`: distribution to sample from
"""
function sample!(
    trace::Dict,
    handlers_stack::Array{AbstractHandler,1},
    name::Any,
    dist::Distributions.Distribution
)
    if length(handlers_stack) == 0
        return rand(dist)
    end

    # NOTE: In this simplified case we could replace :args with :dist because
    #       our "effect" is always sampling from a distribution.
    # NOTE: This could be a custom type but is a Dict for simplicity.
    initial_msg = Dict(
        :fn => rand,
        :args => (dist, ),
        :name => name,
        :value => nothing,
        :is_observed => false,
        :done => false, # Use if we still want other handlers to be active.
        :stop => false, # Use if we want handlers below on the stack not to be active.
        :continuation => nothing # Function to be called after effect handlers have been applied.
    )
    msg = apply_stack!(trace, handlers_stack, initial_msg)
    return msg[:value]
end

"""
    apply_stack!(trace::Dict, handlers_stack::Array{AbstractHandler,1}, msg::Dict)

Apply all the effect handlers from `handlers_stack` to the effect described in
`msg`.

# Arguments
- `trace::Dict`: dict which can be used to store things in a global state
- `handlers_stack::Array{AbstractHandler,1}`: stack of effect handlers
- `msg::Dict`: dict describing the effectful computation

# Returns
- `Dict`: message after effect handlers are applied
"""
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
end # module
