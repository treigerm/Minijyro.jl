@testset "handle" begin
    @jyro function m()
        x ~ Normal()
    end

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
    @jyro function m()
        x ~ Normal()
    end

    handle!(m, TraceHandler())
    t = m()
    @test haskey(t, :msgs)
    @test haskey(t[:msgs], :x)
    @test isa(t[:msgs][:x][:value], Float64)
end

@testset "logjoint" begin
    @jyro function m()
        x ~ Normal()
    end

    trace!(m)
    log_joint!(m)
    t = m()
    val = t[:msgs][:x][:value]

    @test logpdf(Normal(), val) == t[:logjoint]
end

@testset "condition" begin
    @jyro function m()
        x ~ Normal()
    end

    condition!(m, Dict(:x => 0.0))
    log_joint!(m)
    trace!(m)

    t = m()
    val = t[:msgs][:x][:value]

    @test val == 0.0
    @test t[:logjoint] == logpdf(Normal(), 0.0)
    @test t[:msgs][:x][:is_observed]
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

@testset "generated handlers" begin
    @jyro function model()
        a ~ Normal()
    end

    conditioned_model = condition(model, Dict(:a => 1.0))
    trace!(model)
    t = model()
    cond_t = trace(conditioned_model)()

    # Traces from conditioned model should be different.
    @test cond_t[:msgs][:a][:value] == 1.0
    @test cond_t[:msgs][:a][:value] != t[:msgs][:a][:value] # This should technically never happen.
end
