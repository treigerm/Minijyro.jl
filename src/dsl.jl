import MacroTools
using Base

const TRACE_SYMBOL = :trace
const STACK_SYMBOL = :handlers_stack
const RETURN_KEY = :(:_return)

macro jyro(expr)
    fn_dict = MacroTools.splitdef(expr)
    pushfirst!(fn_dict[:args], STACK_SYMBOL)

    loop_var = gensym()
    body = translate_tilde(fn_dict[:body])
    body, has_return = translate_return(body, loop_var)

    fn_dict[:body] = quote
        $TRACE_SYMBOL = Dict{Any,Any}($(RETURN_KEY) => nothing)
        for $(loop_var) in $(STACK_SYMBOL)[end:-1:1]
            enter!($TRACE_SYMBOL, $(loop_var))
        end
        $(body.args...)
        # NOTE: If the last line of (body.args...) is already a return statement
        #       then the compiler is smart enough to remove this code.
        for $(loop_var) in $STACK_SYMBOL
            exit!($TRACE_SYMBOL, $(loop_var))
        end
        return $TRACE_SYMBOL
    end


    model_name = fn_dict[:name]
    fn_name = gensym(model_name)
    fn_dict[:name] = fn_name
    fn_dict[:rtype] = Dict{Any,Any} # Change return type of generated function to a Dict.
    fn_expr = esc(MacroTools.combinedef(fn_dict))

    return quote
        $fn_expr

        $(esc(model_name)) = MinijyroModel(AbstractHandler[], $(esc(fn_name)))
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

function translate_return(expr, loop_var)
    has_return = false
    expr = MacroTools.postwalk(expr) do e
        if MacroTools.@capture(e, return r_)
            has_return = true
            quote
                for $(loop_var) in $STACK_SYMBOL
                    exit!($TRACE_SYMBOL, $(loop_var))
                end
                $(TRACE_SYMBOL)[$(RETURN_KEY)] = $r
                return $TRACE_SYMBOL
            end
        else
            e
        end
    end
    return (expr, has_return)
end

# TODO: Macro for handling code segments
