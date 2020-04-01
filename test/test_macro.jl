using Logging

@testset "return type" begin
    @jyro function m()::Int
        x ~ Normal()
        return 2
    end

    @test typeof(m) == MinijyroModel
    m_return_types = Base.return_types(m.model_fn)
    @test length(m_return_types) == 1
    @test m_return_types[1] == Dict{Any,Any}

    t = m()
    @test haskey(t, :_return)
    @test t[:_return] == 2

    @jyro function m2()
        x ~ Normal()
    end

    m_return_types = Base.return_types(m2.model_fn)
    @test length(m_return_types) == 1
    @test m_return_types[1] == Dict{Any,Any}

    t = m2()
    @test haskey(t, :_return)
    @test t[:_return] == nothing
end

@testset "multiple return statements" begin
    @jyro function m(switch::Bool)
        x ~ Normal()
        if switch
            return 1
        end

        y ~ Normal()
        return 2
    end


    trace!(m)
    t = m(true)
    @test t[:_return] == 1
    @test haskey(t[:msgs], :x)
    @test !haskey(t[:msgs], :y)

    t = m(false)
    @test t[:_return] == 2
    @test haskey(t[:msgs], :x)
    @test haskey(t[:msgs], :y)
end
