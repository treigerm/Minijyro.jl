@testset "enumeration" begin
    @jyro function model()
        a ~ Bernoulli(0.5)
        b ~ Bernoulli(0.5)
        c ~ Bernoulli(0.5)
        # Use this as a deterministic function.
        s ~ DiscreteNonParametric([a + b + c], [1.0])
        return s
    end

    dist = discrete_enumeration(model, :_return)
    analytic_dist = DiscreteNonParametric([0, 1, 2, 3], [0.125, 0.375, 0.375, 0.125])
    @test dist == analytic_dist
    dist_s = discrete_enumeration(model, :s)
    @test dist_s == analytic_dist
end
