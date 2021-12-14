#
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
# include("split-sdp.jl"); using .SplitSdp
# include("tests.jl"); using .Tests

using LinearAlgebra
using JuMP
using Random

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

# Tihs combination makes a difference for 2-stride and all-stride
# xdims = [2; 3; 4; 5; 4; 3; 2]
# xdims = [2; 3; 4; 5; 6; 7; 6; 5; 4; 3; 2]
xdims = [2; 10; 10; 10; 10; 2]
# xdims = [2; 20; 20; 20; 20; 2]
# xdims = [2; 40; 40; 40; 40; 40; 40; 2]
# xdims = [2; 20; 20; 20; 20; 20; 2]

ffnet = randomNetwork(xdims, σ=0.8)

# Plot some trajectories

xcenter = ones(ffnet.xdims[1])
ε = 0.01
input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
# input = BoxInput(x1min=-ones(xdims[1]), x1max=ones(xdims[1]))
runAndPlotRandomTrajectories(10000, ffnet, x1min=input.x1min, x1max=input.x1max)

norm2 = 50^2
safety = safetyNormBound(norm2, xdims)
safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

normals = [[0;1], [1;1], [1;0], [1;-1], [0;-1], [-1;-1], [-1;0], [-1;1]]
reach_set = HyperplaneSet(normals=normals)
reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=reach_set)

#

plain_opts = DeepSdpOptions(use_xintervals=false, use_localized_slopes=false, verbose=true)
xintv_opts = DeepSdpOptions(use_xintervals=true, use_localized_slopes=false, verbose=true)
sintv_opts = DeepSdpOptions(use_xintervals=false, use_localized_slopes=true, verbose=true)
all_opts = DeepSdpOptions(use_xintervals=true, use_localized_slopes=true, verbose=true)


safety_soln = DeepSdp.run(safety_inst, all_opts)

reach_soln = DeepSdp.run(reach_inst, all_opts)

