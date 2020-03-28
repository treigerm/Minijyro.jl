using StatsFuns: logsumexp
using AdvancedHMC
using ForwardDiff
using ReverseDiff

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
    # TODO: This assumes a "static" model with fixed dimensionality.
    t = trace(model)(model_args...)
    param_info = Dict()
    num_params = 0 # Total number of parameters
    for msg in values(t[:msgs])
        if !msg[:is_observed]
            val = msg[:value]
            d = Dict{Symbol,Any}(
                :is_scalar => isa(val, Number)
            )
            if !d[:is_scalar]
                # TODO: Assume it is an array. Maybe have an assert.
                d[:shape] = size(val)
                d[:num_elems] = prod(size(val))
            end
            param_info[msg[:name]] = d
            num_params += d[:is_scalar] ? 1 : d[:num_elems]
        end
    end
    initial_params = randn(num_params)

    function density(params)
        params_dict = Dict()
        # Precompute mapping param_name -> (length, shape)
        i = 1
        for name in keys(param_info)
            if param_info[name][:is_scalar]
                params_dict[name] = params[i]
                i += 1
            else
                num_elems = param_info[name][:num_elems]
                params_dict[name] = reshape(params[i:i+num_elems-1], param_info[name][:shape])
                i += num_elems # TODO: Check for no off by one error.
            end
        end
        m = log_joint(condition(model, params_dict))
        return m(model_args...)[:logjoint]
    end

    function grad_density(x)
        # TODO: Use tape.
        score = density(x)
        grad = ReverseDiff.gradient(density, x)
        return (score, grad)
    end

    # Construct Hamiltonian system
    metric = UnitEuclideanMetric(length(initial_params))
    # TODO: More autodiff options.
    hamiltonian = Hamiltonian(metric, density, grad_density)

    integrator = Leapfrog(step_size)
    proposal = StaticTrajectory(integrator, n_leapfrog)
    return sample(hamiltonian, proposal, initial_params, n_samples)
end
