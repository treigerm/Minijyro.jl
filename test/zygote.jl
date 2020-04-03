using DistributionsAD
using Distributions
using Zygote

struct DataStruct
    data
end

function density(x)
    d = Dict()
    data = Dict()
    data[:x] = x
    d[:logjoint] = 0.0
    # TODO: Zygote cannot handle the [end:-1:1] bit.
    for h in Zygote.nograd([data][end:-1:1])
        d[:logjoint] = d[:logjoint] + logpdf(Normal(), h[:x])
    end
    return d[:logjoint]
end

Zygote.gradient(density, 1.0)
