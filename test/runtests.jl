using Minijyro
using Distributions
using Test
using Random

Random.seed!(42)

@testset "Minijyro tests" begin
    include("test_handlers.jl")
    include("test_inference.jl")

    @testset "sample function" begin
        trace = Dict()
        handlers_stack = AbstractHandler[]
        name = :normal
        dist = Normal()
        @test isa(Minijyro.sample!(trace, handlers_stack, name, dist), Float64)
    end
end
