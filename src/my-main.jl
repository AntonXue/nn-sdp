#
main_start_time = time()

#
include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/deep-sdp.jl"); using .DeepSdp
# include("core/split-sdp.jl"); using .SplitSdp
# include("core/admm-sdp.jl"); using .AdmmSdp
include("parsers/nnet-parser.jl"); using .NNetParser
include("parsers/vnnlib-parser.jl"); using .VnnlibParser
include("utils.jl"); using .Utils
# include("tests.jl"); using .Tests
using LinearAlgebra
using JuMP
using Random

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

# This combination makes a difference for 2-stride and all-stride

# This pair sees some differences with 1.15^2
xdims = [2; 3; 4; 5; 4; 3; 2]
ffnet = randomNetwork(xdims, σ=0.8)

# xdims = [2; 3; 4; 5; 6; 7; 8; 7; 6; 5; 4; 3; 2]
# xdims = [2; 10; 10; 10; 10; 2]
# xdims = [2; 20; 20; 20; 20; 20; 20; 2]
# xdims = [2; 50; 50; 50; 50; 50; 2]
# xdims = [2; 10; 10; 10; 10; 10; 2]
# xdims = [2; 20; 20; 20; 20; 20; 2]
# xdims = [2; 30; 30; 30; 30; 30; 2]


# Plot some trajectories

xcenter = ones(ffnet.xdims[1])
ε = 0.1
input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
runAndPlotRandomTrajectories(1000, ffnet, x1min=input.x1min, x1max=input.x1max)
x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

# Safety instance
safety = safetyNormBound(1.15^2, xdims)
safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
println("finally aobut to start important stuff after " * string(time() - main_start_time))
opts0 = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband=0)
opts1 = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband=1)
opts5 = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband=5)
opts10 = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband=10)
opts15 = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband=15)
optsall = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband=nothing)

res0 = DeepSdp.run(safety_inst, opts0)
res1 = DeepSdp.run(safety_inst, opts0)
res5 = DeepSdp.run(safety_inst, opts5)
res10 = DeepSdp.run(safety_inst, opts10)
res15 = DeepSdp.run(safety_inst, opts15)
resall = DeepSdp.run(safety_inst, optsall)

# Admm Options
#=
println("finally about to do the ADMM stuff! after " * string(time() - main_start_time))

split_opts = SplitSdpOptions(β=1, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=true)
admm_opts = AdmmSdpOptions(β=1, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=true)
admm_params = initParams(safety_inst, admm_opts)

admm_cache, _ = precomputeCache(admm_params, safety_inst, admm_opts)
iter_params, _ = AdmmSdp.admm(admm_params, admm_cache, admm_opts)

z1 = AdmmSdp.makezk(1, iter_params.ωs[1], admm_cache)
z2 = AdmmSdp.makezk(2, iter_params.ωs[2], admm_cache)
z3 = AdmmSdp.makezk(3, iter_params.ωs[3], admm_cache)

Z1dim = Int(round(sqrt(length(z1))))
Z2dim = Int(round(sqrt(length(z2))))
Z3dim = Int(round(sqrt(length(z3))))

Z1 = Symmetric(reshape(z1, (Z1dim, Z1dim)))
Z2 = Symmetric(reshape(z2, (Z2dim, Z2dim)))
Z3 = Symmetric(reshape(z3, (Z3dim, Z3dim)))
=#


# sn, an, ac, sy, ay = Tests.testAdmmCache()
