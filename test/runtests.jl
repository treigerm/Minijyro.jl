using Minijyro
using Distributions
using Test
using Random

Random.seed!(42)

# TODO Check whether we use testset properly.

@testset "sample function" begin
    # TODO: Check that handlers stack is not altered.
    trace = Dict()
    handlers_stack = []
    name = :normal
    dist = Normal()
    @test isa(Minijyro.sample!(trace, handlers_stack, name, dist), Float64)
end

@testset "apply stack function" begin
    # TODO: Checksome basic trace handler functionality.
    trace = Dict()
    stack = []
    msg = Dict(
        :fn => rand,
        :args => (Normal(),),
        :value => nothing
    )

    @test isa(apply_stack!(trace, stack, msg)[:value], Float64)
    @test isa(apply_stack!(trace, stack, msg), Dict)
end

@testset "handle" begin
    m = Minijyro.MinijyroModel([], x -> rand(Normal(), x))

    handle!(m, TraceHandler())
    @test length(m.handlers_stack) == 1
    @test isa(m.handlers_stack[1], TraceHandler)

    new_m = handle(m, LogJointHandler())
    @test length(m.handlers_stack) == 1
    @test length(new_m.handlers_stack) == 2
    @test isa(new_m.handlers_stack[1], TraceHandler)
    @test isa(new_m.handlers_stack[2], LogJointHandler)
end

@testset "trace" begin
    trace = Dict()
    trace_handler = TraceHandler()
    handlers_stack = [trace_handler]

    name = :normal
    dist = Normal()
    enter!(trace, handlers_stack[1])
    val = Minijyro.sample!(trace, handlers_stack, name, dist)

    @test val == trace[:msgs][:normal][:value]
end

@testset "logjoint" begin
    trace = Dict()
    logjoint_handler = LogJointHandler()
    handlers_stack = [logjoint_handler]

    name = :normal
    dist = Normal()
    enter!(trace, handlers_stack[1])
    val = Minijyro.sample!(trace, handlers_stack, name, dist)

    @test logpdf(dist, val) == trace[:logjoint]
end

@testset "condition" begin
    trace = Dict()
    logjoint_handler = LogJointHandler()
    condition_handler = ConditionHandler(Dict(:normal => 0.0))
    handlers_stack = [condition_handler, logjoint_handler]

    name = :normal
    dist = Normal()
    enter!(trace, handlers_stack[2])
    val = Minijyro.sample!(trace, handlers_stack, name, dist)

    @test val == 0.0
    @test trace[:logjoint] == logpdf(dist, 0.0)
    # TODO: Add test for is observed.    
end

@testset "replay handler" begin
    # First sampling.
    @jyro function model1()
        x ~ Normal()
    end

    handle!(model1, TraceHandler())
    trace1 = model1()

    # Second sampling.
    @jyro function model2()
        x ~ Normal()
        y ~ Normal()
    end

    handle!(model2, ReplayHandler(trace1))
    handle!(model2, TraceHandler())
    trace2 = model2()

    @test trace1[:msgs][:x][:value] == trace2[:msgs][:x][:value]
end

@testset "escape handler" begin
    @jyro function model()
        x ~ Normal()
        {:y => 1} ~ Normal()
        z ~ Normal()
    end

    # Escape when we reach variable y.
    handle!(model, EscapeHandler(x -> x[:name] == (:y => 1)))
    handle!(model, TraceHandler())
    try
        t = model()
    catch e
        # TODO: Should this test be in a finally block?
        if isa(e, EscapeException)
            t = e.trace

            @test e.name == (:y => 1)
            @test haskey(t[:msgs], :x)
            @test isa(t[:msgs][:x][:value], Float64)
            @test t[:msgs][e.name][:value] == nothing
            @test !haskey(t[:msgs], :z)
        else
            throw(e)
        end
    end
end

@testset "queue" begin
    @jyro function model()
        a ~ Bernoulli(0.5)
        b ~ Bernoulli(0.5)
    end

    true_enum = Set([(true,true), (false,true), (true,false), (false,false)])
    empty_trace = Dict()
    empty_trace[:msgs] = Dict()
    q = [empty_trace]

    enum_model = queue(model, q)
    generated_enum = Set()
    while length(q) > 0
        t = enum_model()
        push!(generated_enum, (t[:msgs][:a][:value], t[:msgs][:b][:value]))
    end

    @test true_enum == generated_enum
end

@testset "enumeration" begin
    @jyro function model()
        a ~ Bernoulli(0.5)
        b ~ Bernoulli(0.5)
        c ~ Bernoulli(0.5)
        # Use this as a deterministic function.
        s ~ DiscreteNonParametric([a + b + c], [1.0])
    end

    #handle!(model, ConditionHandler(Dict(:s => 2)))
    dist = discrete_enumeration(model, :s)
    analytic_dist = DiscreteNonParametric([0, 1, 2, 3], [0.125, 0.375, 0.375, 0.125])
    @test dist == analytic_dist
    # TODO: Tests.
end

@testset "generated" begin
    @jyro function model()
        a ~ Normal()
    end

    conditioned_model = condition(model, Dict(:a => 1.0))
    trace!(model)
    t = model()
    cond_t = trace(conditioned_model)()

    @test cond_t[:msgs][:a][:value] == 1.0
    @test cond_t[:msgs][:a][:value] != t[:msgs][:a][:value] # This should technically never happen.
end
