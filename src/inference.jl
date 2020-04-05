using StatsFuns: logsumexp
using AdvancedHMC
using ForwardDiff
using ReverseDiff

"""
    discrete_enumeration(model::MinijyroModel, site_name::Any)

Computes the marginal distribution of `site_name` by enumerating all possible
execution traces. This requires that all sample sites have discrete values.

# Arguments
- `model::MinijyroModel`: model which to perform inference in
- `site_name::Any`: site name which to compute marginal distribution of; use `:_return`
    to compute marginal distribution of the return value

# Returns
- `Distributions.DiscreteNonParametric`: the marginal distribution
"""
function discrete_enumeration(model::MinijyroModel, site_name::Any)
    # Site name is the name of the variable we are interested in.
    empty_trace = Dict()
    empty_trace[:msgs] = Dict()
    q = [empty_trace]

    log_joint!(model)
    enum_model = queue(model, q)
    samples = Dict{Int,Float64}()
    while length(q) > 0
        trace = enum_model()
        val = site_name == :_return ? trace[:_return] : trace[:msgs][site_name][:value]
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

function get_param_info(model::MinijyroModel, model_args::Tuple)
    # For each unobserved site get information on whether it is a scalar or not
    # and if it is not a scalar then get its shape and the number of elements in
    # there.
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
                @assert isa(val, AbstractArray) "Sampled values must either be Number or AbstractArray but found $(typeof(val))."
                d[:shape] = size(val)
                d[:num_elems] = prod(size(val))
            end
            param_info[msg[:name]] = d
            num_params += d[:is_scalar] ? 1 : d[:num_elems]
        end
    end

    return (param_info, num_params)
end

function load_params!(params_dict::Dict, param_info::Dict, params::AbstractArray)
    # Load parameters from params into params_dict.
    i = 1
    for name in keys(param_info)
        if param_info[name][:is_scalar]
            params_dict[name] = params[i]
            i += 1
        else
            num_elems = param_info[name][:num_elems]
            params_dict[name] = reshape(params[i:i+num_elems-1], param_info[name][:shape])
            i += num_elems
        end
    end
end

function get_grad_fn(density::Function, autodiff::Symbol)
    if autodiff == :reverse
        function grad_density(x)
            results = DiffResults.GradientResult(x)
            ReverseDiff.gradient!(results, density, x)
            return (DiffResults.value(results), DiffResults.gradient(results))
        end
        return grad_density
    elseif autodiff == :forward
        return ForwardDiff
    else
        error("Autodiff option $autodiff not supported")
    end
end

"""
    hmc(args...; kwargs...)

Runs static Hamiltonian Monte Carlo with a fixed integrator step size and fixed
number of integrator steps.

# Arguments
- `model::MinijyroModel`: model to do inference in
- `model_args::Tuple`: arguments for the `model` to be called with
- `step_size::Float64`: integrator step size
- `n_leapfrog::Int`: number of steps taken by leapfrog integrator
- `n_samples::Int`: number of samples to take from posterior

# Keywords
- `autodiff::Symbol=:forward`: either `:forward` or `:reverse` to select autodiff mode
"""
function hmc(
    model::MinijyroModel,
    model_args::Tuple,
    step_size::Float64,
    n_leapfrog::Int,
    n_samples::Int;
    autodiff::Symbol=:forward
)
    # NOTE: This assumes a "static" model with fixed dimensionality.
    param_info, num_params = get_param_info(model, model_args)
    initial_params = randn(num_params)

    function density(params)
        params_dict = Dict()
        load_params!(params_dict, param_info, params)
        m = log_joint(condition(model, params_dict))
        return m(model_args...)[:logjoint]
    end

    grad_fn = get_grad_fn(density, autodiff)

    # Construct Hamiltonian system
    metric = UnitEuclideanMetric(length(initial_params))
    hamiltonian = Hamiltonian(metric, density, grad_fn)

    integrator = Leapfrog(step_size)
    proposal = StaticTrajectory(integrator, n_leapfrog)
    return AdvancedHMC.sample(hamiltonian, proposal, initial_params, n_samples)
end

"""
    nuts(args...; kwargs...)

Run the NUTS algorithm to get posterior samples from `model`.

# Arguments
- `model::MinijyroModel`: model to do inference in
- `model_args::Tuple`: arguments for the `model` to be called with
- `num_samples::Int`: number of samples to take from posterior

# Keywords
- `num_warmup::Int=1000`: number of warmup steps to burn-in the chain; these samples
    will be discarded
- `autodiff::Symbol=:forward`: either `:forward` or `:reverse` to select autodiff mode
- `progress::Bool=true`: whether to display progress meter or not
"""
function nuts(
    model::MinijyroModel,
    model_args::Tuple,
    num_samples::Int;
    num_warmup::Int=1000,
    autodiff::Symbol=:forward,
    progress::Bool=true
)
    # NOTE: This assumes a "static" model with fixed dimensionality.
    param_info, num_params = get_param_info(model, model_args)
    initial_params = randn(num_params)

    function density(params)
        params_dict = Dict()
        load_params!(params_dict, param_info, params)
        m = log_joint(condition(model, params_dict))
        return m(model_args...)[:logjoint]
    end

    grad_fn = get_grad_fn(density, autodiff)

    metric = DiagEuclideanMetric(length(initial_params))
    hamiltonian = Hamiltonian(metric, density, grad_fn)

    initial_step_size = find_good_eps(hamiltonian, initial_params)
    integrator = Leapfrog(initial_step_size)

    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(
        Preconditioner(metric), NesterovDualAveraging(0.8, integrator)
    )

    return AdvancedHMC.sample(
        hamiltonian,
        proposal,
        initial_params,
        num_samples,
        adaptor,
        num_warmup;
        progress=progress
    )
end
