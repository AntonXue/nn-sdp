#
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
include("split-sdp.jl"); using .SplitSdp
# include("tests.jl"); using .Tests

using LinearAlgebra
using JuMP
using Random

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

# This combination makes a difference for 2-stride and all-stride
# xdims = [2; 3; 4; 5; 4; 3; 2]
# xdims = [2; 3; 4; 5; 6; 7; 6; 5; 4; 3; 2]
xdims = [2; 10; 10; 10; 10; 2]
# xdims = [2; 20; 20; 20; 20; 2]

ffnet = randomNetwork(xdims, σ=0.5)

# Plot some trajectories

xcenter = ones(ffnet.xdims[1])
ε = 0.01
input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
# input = BoxInput(x1min=-ones(xdims[1]), x1max=ones(xdims[1]))
runAndPlotRandomTrajectories(10000, ffnet, x1min=input.x1min, x1max=input.x1max)

norm2 = 50^2
safety = safetyNormBound(norm2, xdims)
safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

hplane1 = HyperplaneSet(normal=[0.0; 1.0])
hplane2 = HyperplaneSet(normal=[1.0; 1.0])
hplane3 = HyperplaneSet(normal=[1.0; 0.0])
hplane4 = HyperplaneSet(normal=[1.0; -1.0])
hplane5 = HyperplaneSet(normal=[0.0; -1.0])
hplane6 = HyperplaneSet(normal=[-1.0; -1.0])
hplane7 = HyperplaneSet(normal=[-1.0; 0.0])
hplane8 = HyperplaneSet(normal=[-1.0; 1.0])

reach_inst1 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane1)
reach_inst2 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane2)
reach_inst3 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane3)
reach_inst4 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane4)
reach_inst5 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane5)
reach_inst6 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane6)
reach_inst7 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane7)
reach_inst8 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane8)

#
plain_opts = DeepSdpOptions(use_xintervals=false, use_localized_slopes=false, verbose=true)
xintv_opts = DeepSdpOptions(use_xintervals=true, use_localized_slopes=false, verbose=true)
sintv_opts = DeepSdpOptions(use_xintervals=false, use_localized_slopes=true, verbose=true)
all_opts = DeepSdpOptions(use_xintervals=true, use_localized_slopes=true, verbose=true)

deep_safety_soln = DeepSdp.run(safety_inst, all_opts)
deep_reach_soln1 = DeepSdp.run(reach_inst1, all_opts)

#
split_opts1 = SplitSdpOptions(β=1, verbose=true)
split_opts2 = SplitSdpOptions(β=2, verbose=true)
split_opts3 = SplitSdpOptions(β=3, verbose=true)

split_safety_soln1 = SplitSdp.run(safety_inst, split_opts1)
split_reach_soln3 = SplitSdp.run(reach_inst1, split_opts3)




#=
reach_hplanes = [
  ([0.0, 1.0], 3.66),
  ([1.0, 1.0], -6.75),
  ([1.0, 0.0], -10.41),
  ([1.0, -1.0], -14.02),
  ([0.0, -1.0], -3.40),
  ([-1.0, -1.0], 7.26),
  ([-1.0, 0.0], 10.67),
  ([-1.0, 1.0], 14.16)]
=#

# reach_hplanes = split_soln1.values[:hplanes]

# plt1 = plotReachPolytope(xfs, split_soln1.values[:hplanes], imgfile="~/Desktop/foo1.png")
# plt2 = plotReachPolytope(xfs, split_soln2.values[:hplanes], imgfile="~/Desktop/foo2.png")
# plt3 = plotReachPolytope(xfs, split_soln3.values[:hplanes], imgfile="~/Desktop/foo3.png")



