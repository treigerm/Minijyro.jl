using Minijyro
using Distributions
using Test

# TODO: Set random seed.

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
end
