import MacroTools
using Base

const TRACE_SYMBOL = :trace
const STACK_SYMBOL = :handlers_stack

# TODO: Make handlers stack a stack of AbstractHandlers.
# TODO: Maybe use actual stack and not an array?
struct MinijyroModel
    handlers_stack::Array{Any,1}
    model_fn::Function
end

function (model::MinijyroModel)(args...)
    # TODO: Possibly give nicer error message when args are given in wrong type.
    return model.model_fn(model.handlers_stack, args...)
end

Base.copy(m::MinijyroModel) = MinijyroModel(copy(m.handlers_stack), m.model_fn)

macro jyro(expr)
    # TODO: Check for return.
    fn_dict = MacroTools.splitdef(expr)
    pushfirst!(fn_dict[:args], STACK_SYMBOL)

    body = translate_tilde(fn_dict[:body])

    # TODO: Is it okay to assign h to the loop variable?
    fn_dict[:body] = quote
        $TRACE_SYMBOL = Dict()
        for h in $(STACK_SYMBOL)[end:-1:1]
            enter!($TRACE_SYMBOL, h)
        end
        $(body.args...)
        for h in $STACK_SYMBOL
            exit!($TRACE_SYMBOL, h)
        end
        return $TRACE_SYMBOL
    end

    model_name = fn_dict[:name]
    fn_name = gensym(model_name)
    fn_dict[:name] = fn_name
    fn_expr = esc(MacroTools.combinedef(fn_dict))

    return quote
        $fn_expr

        $(esc(model_name)) = MinijyroModel([], $(esc(fn_name)))
    end
end

function translate_tilde(expr)
    # TODO: Handler cases in loops.
    # NOTE: Code adapted from https://github.com/probcomp/Gen/blob/master/src/dsl/dsl.jl
    MacroTools.postwalk(expr) do e
        if MacroTools.@capture(e, {addr_} ~ rhs_)
            # TODO: How to handle variable assignment?
            :(Minijyro.sample!($TRACE_SYMBOL, $STACK_SYMBOL, $(addr), $rhs))
        elseif MacroTools.@capture(e, lhs_ ~ rhs_)
            name_symbol = QuoteNode(lhs)
            :($lhs = Minijyro.sample!($TRACE_SYMBOL, $STACK_SYMBOL, $name_symbol, $rhs))
        else
            e
        end
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

# TODO: Macro for handling code segments
