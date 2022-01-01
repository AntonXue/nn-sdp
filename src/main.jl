#
main_start_time = time()

#
include("header.jl"); using .Header
include("common.jl"); using .Common
include("intervals.jl"); using .Intervals
include("partitions.jl"); using .Partitions
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
include("split-sdp.jl"); using .SplitSdp
include("admm-sdp.jl"); using .AdmmSdp
include("tests.jl"); using .Tests
using LinearAlgebra
using JuMP
using Random

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

# This combination makes a difference for 2-stride and all-stride
xdims = [2; 3; 4; 5; 4; 3; 2]
# xdims = [2; 3; 4; 5; 6; 7; 8; 7; 6; 5; 4; 3; 2]
# xdims = [2; 10; 10; 10; 10; 2]

ffnet = randomNetwork(xdims, σ=1.0)

# Plot some trajectories

# xcenter = ones(ffnet.xdims[1])
# ε = 0.1
# input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
# runAndPlotRandomTrajectories(1000, ffnet, x1min=input.x1min, x1max=input.x1max)
# x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

# Safety instance
# safety = safetyNormBound(10^2, xdims)
# safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

# Admm Options
println("finally about to do the ADMM stuff! after " * string(time() - main_start_time))

# split_opts = SplitSdpOptions(β=2, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=true)
# admm_opts = AdmmSdpOptions(β=2, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=true)
# admm_params = initParams(safety_inst, admm_opts)
# admm_cache = precomputeCache(admm_params, safety_inst, admm_opts)
# iter_params = AdmmSdp.admm(admm_params, admm_cache, admm_opts)



sn, an, ac, sy, ay = Tests.testAdmmCache()
