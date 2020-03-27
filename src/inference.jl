using StatsFuns: logsumexp
using AdvancedHMC
using ForwardDiff

function discrete_enumeration(model::MinijyroModel, site_name)
    # Site name is the name of the variable we are interested in.
    empty_trace = Dict()
    empty_trace[:msgs] = Dict()
    q = [empty_trace]

    log_joint!(model)
    enum_model = queue(model, q)
    samples = Dict{Int,Float64}()
    while length(q) > 0
        trace = enum_model()
        val = trace[:msgs][site_name][:value]
        if !haskey(samples, val)
            samples[val] = 0.0
        end
        samples[val] += exp(trace[:logjoint])
    end

    vals = collect(keys(samples))
    probs = collect(values(samples))
    probs = probs ./ sum(probs)
    return DiscreteNonParametric(vals, probs)
end

function hmc(model::MinijyroModel, step_size, n_leapfrog, n_samples, model_args)
    # TODO: How to handle args passed to model?

    # TODO: This assumes a "static" model with fixed dimensionality.

    t = trace(model)(model_args...)
    param_names = []
    for msg in values(t[:msgs])
        if !msg[:is_observed]
            push!(param_names, msg[:name])
        end
    end
    dim = length(param_names)
    initial_params = rand(dim)

    function density(params)
        params_dict = Dict()
        for (i, name) in enumerate(param_names)
            params_dict[name] = params[i]
        end
        m = log_joint(condition(model, params_dict))
        return m(model_args...)[:logjoint]
    end

    # Construct Hamiltonian system
    metric = UnitEuclideanMetric(dim)
    hamiltonian = Hamiltonian(metric, density, ForwardDiff)

    integrator = Leapfrog(step_size)
    proposal = StaticTrajectory(integrator, n_leapfrog)
    return sample(hamiltonian, proposal, initial_params, n_samples)
end
