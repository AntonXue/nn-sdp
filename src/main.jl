#
include("header.jl"); using .Header
include("common.jl"); using .Common
include("intervals.jl"); using .Intervals
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
include("split-sdp.jl"); using .SplitSdp
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

ffnet = randomNetwork(xdims, σ=0.5)

# Plot some trajectories

xcenter = ones(ffnet.xdims[1])
ε = 0.1
# input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
input = BoxInput(x1min=-ones(xdims[1]), x1max=ones(xdims[1]))
# runAndPlotRandomTrajectories(1000, ffnet, x1min=input.x1min, x1max=input.x1max)
x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

# Safety instance
norm2 = 5^2
safety = safetyNormBound(norm2, xdims)
safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

# SplitSdp options
split_opts = SplitSdpOptions(β=4, verbose=true)

Tests.testY(safety_inst, split_opts)


