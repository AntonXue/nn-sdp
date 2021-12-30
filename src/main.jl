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

xcenter = ones(ffnet.xdims[1])
ε = 0.1
input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
# runAndPlotRandomTrajectories(1000, ffnet, x1min=input.x1min, x1max=input.x1max)
x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

# Safety instance
safety = safetyNormBound(5^2, xdims)
safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

# Reachability Instance
hplane = HyperplaneSet(normal=[0.0; 1.0])
reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane)

# SplitSdp options
split_opts1 = SplitSdpOptions(β=1, verbose=true, x_intervals=x_intvs, slope_intervals=slope_intvs)
split_opts2 = SplitSdpOptions(β=2, verbose=true, x_intervals=x_intvs, slope_intervals=slope_intvs)
split_opts3 = SplitSdpOptions(β=3, verbose=true, x_intervals=x_intvs, slope_intervals=slope_intvs)
split_opts4 = SplitSdpOptions(β=4, verbose=true, x_intervals=x_intvs, slope_intervals=slope_intvs)

# Admm Options
admm_opts = AdmmSdpOptions(β=2)


