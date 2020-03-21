import MacroTools

const TRACE_SYMBOL = :trace
const STACK_SYMBOL = :handlers_stack

macro model(exp)
    # TODO: Check for return.
    body = translate_tilde(exp.args[2])

    exp.args[2] = quote
        $TRACE_SYMBOL = Dict()
        $STACK_SYMBOL = []
        $(body.args[2])
        return $TRACE_SYMBOL
    end

    show(exp)
    return esc(:($exp))
end

function translate_tilde(expr)
    # TODO: Handler cases in loops.
    MacroTools.postwalk(expr) do e
        if MacroTools.@capture(e, lhs_ ~ rhs_)
            name_symbol = QuoteNode(lhs)
            :($lhs = Minijyro.sample!($TRACE_SYMBOL, $STACK_SYMBOL, $name_symbol, $rhs))
        else
            e
        end
    end
end
