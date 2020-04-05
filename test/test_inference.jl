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

@testset "helper functions" begin
    D = 2
    @jyro function model()
        x ~ Normal()
        y ~ MvNormal(D, 1)
        z ~ Normal()
    end

    condition!(model, Dict(:z => 0))
    param_info, num_params = Minijyro.get_param_info(model, ())

    @test num_params == (D + 1)
    @test !haskey(param_info, :z)
    @test param_info[:x][:is_scalar]
    @test !param_info[:y][:is_scalar]
    @test param_info[:y][:num_elems] == D
    @test param_info[:y][:shape] == (D,)

    params_dict = Dict()
    new_params = [0.0, 1.0, 1.0]
    Minijyro.load_params!(params_dict, param_info, new_params)

    params_dict2 = Dict()
    Minijyro.load_params!(params_dict2, param_info, new_params)

    # Because we loop through a dict we do not know the order of keys beforehand
    # but it should always be consistent.
    @test params_dict[:x] == params_dict2[:x]
    @test params_dict[:y] == params_dict2[:y]
end
