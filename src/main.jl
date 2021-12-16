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
# xdims = [2; 3; 4; 5; 4; 3; 2]
# xdims = [2; 3; 4; 5; 6; 7; 6; 5; 4; 3; 2]
xdims = [2; 10; 10; 10; 10; 2]
# xdims = [2; 20; 20; 20; 20; 2]

ffnet = randomNetwork(xdims, σ=0.5)

# Plot some trajectories

xcenter = ones(ffnet.xdims[1])
ε = 0.1
input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
# input = BoxInput(x1min=-ones(xdims[1]), x1max=ones(xdims[1]))
# runAndPlotRandomTrajectories(10000, ffnet, x1min=input.x1min, x1max=input.x1max)

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
x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
rand_x_intvs, _, rand_slope_intvs = randomizedPropagation(input.x1min, input.x1max, ffnet, 100000)
xfs = randomTrajectories(10000, ffnet, x1min=input.x1min, x1max=input.x1max)

deep_opts = DeepSdpOptions(x_intervals=nothing, slope_intervals=slope_intvs, verbose=true)

only_x_opts = DeepSdpOptions(x_intervals=x_intvs, slope_intervals=nothing, verbose=true)
only_slope_opts = DeepSdpOptions(x_intervals=nothing, slope_intervals=slope_intvs, verbose=true)
none_opts = DeepSdpOptions(x_intervals=nothing, slope_intervals=nothing, verbose=true)
deep_opts = DeepSdpOptions(x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=true)

only_slope_soln = DeepSdp.run(reach_inst1, only_slope_opts)
only_x_soln = DeepSdp.run(reach_inst1, only_x_opts)
none_soln = DeepSdp.run(reach_inst1, none_opts)
deep_soln = DeepSdp.run(reach_inst1, deep_opts)



#=
deep_reach_soln2 = DeepSdp.run(reach_inst2, deep_opts)
deep_reach_soln3 = DeepSdp.run(reach_inst3, deep_opts)
deep_reach_soln4 = DeepSdp.run(reach_inst4, deep_opts)
deep_reach_soln5 = DeepSdp.run(reach_inst5, deep_opts)
deep_reach_soln6 = DeepSdp.run(reach_inst6, deep_opts)
deep_reach_soln7 = DeepSdp.run(reach_inst7, deep_opts)
deep_reach_soln8 = DeepSdp.run(reach_inst8, deep_opts)
=#

#=
split_opts = SplitSdpOptions(β=3, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=true)
split_reach_soln1 = SplitSdp.run(reach_inst1, split_opts)
split_reach_soln2 = SplitSdp.run(reach_inst2, split_opts)
split_reach_soln3 = SplitSdp.run(reach_inst3, split_opts)
split_reach_soln4 = SplitSdp.run(reach_inst4, split_opts)
split_reach_soln5 = SplitSdp.run(reach_inst5, split_opts)
split_reach_soln6 = SplitSdp.run(reach_inst6, split_opts)
split_reach_soln7 = SplitSdp.run(reach_inst7, split_opts)
split_reach_soln8 = SplitSdp.run(reach_inst8, split_opts)
=#



#=
println("about to run randomized ones!")

rand_deep_opts = DeepSdpOptions(x_intervals=rand_x_intvs, slope_intervals=rand_slope_intvs, verbose=true)
rand_deep_reach_soln1 = DeepSdp.run(reach_inst1, rand_deep_opts)
rand_deep_reach_soln2 = DeepSdp.run(reach_inst2, rand_deep_opts)
rand_deep_reach_soln3 = DeepSdp.run(reach_inst3, rand_deep_opts)
rand_deep_reach_soln4 = DeepSdp.run(reach_inst4, rand_deep_opts)
rand_deep_reach_soln5 = DeepSdp.run(reach_inst5, rand_deep_opts)
rand_deep_reach_soln6 = DeepSdp.run(reach_inst6, rand_deep_opts)
rand_deep_reach_soln7 = DeepSdp.run(reach_inst7, rand_deep_opts)
rand_deep_reach_soln8 = DeepSdp.run(reach_inst8, rand_deep_opts)
=#

# println("Getting ready to plot!")

# Reach hyperplanes
#=
reach_hplanes = [
  (reach_inst1.reach_set.normal, deep_reach_soln1.values[:h]),
  (reach_inst2.reach_set.normal, deep_reach_soln2.values[:h]),
  (reach_inst3.reach_set.normal, deep_reach_soln3.values[:h]),
  (reach_inst4.reach_set.normal, deep_reach_soln4.values[:h]),
  (reach_inst5.reach_set.normal, deep_reach_soln5.values[:h]),
  (reach_inst6.reach_set.normal, deep_reach_soln6.values[:h]),
  (reach_inst7.reach_set.normal, deep_reach_soln7.values[:h]),
  (reach_inst8.reach_set.normal, deep_reach_soln8.values[:h])
]
=#

#=
split_reach_hplanes = [
  (reach_inst1.reach_set.normal, split_reach_soln1.values[:h]),
  (reach_inst2.reach_set.normal, split_reach_soln2.values[:h]),
  (reach_inst3.reach_set.normal, split_reach_soln3.values[:h]),
  (reach_inst4.reach_set.normal, split_reach_soln4.values[:h]),
  (reach_inst5.reach_set.normal, split_reach_soln5.values[:h]),
  (reach_inst6.reach_set.normal, split_reach_soln6.values[:h]),
  (reach_inst7.reach_set.normal, split_reach_soln7.values[:h]),
  (reach_inst8.reach_set.normal, split_reach_soln8.values[:h])
]
=#

#=
rand_reach_hplanes = [
  (reach_inst1.reach_set.normal, rand_deep_reach_soln1.values[:h]),
  (reach_inst2.reach_set.normal, rand_deep_reach_soln2.values[:h]),
  (reach_inst3.reach_set.normal, rand_deep_reach_soln3.values[:h]),
  (reach_inst4.reach_set.normal, rand_deep_reach_soln4.values[:h]),
  (reach_inst5.reach_set.normal, rand_deep_reach_soln5.values[:h]),
  (reach_inst6.reach_set.normal, rand_deep_reach_soln6.values[:h]),
  (reach_inst7.reach_set.normal, rand_deep_reach_soln7.values[:h]),
  (reach_inst8.reach_set.normal, rand_deep_reach_soln8.values[:h])
]
=#




# plt1 = plotReachPolytope(xfs, reach_hplanes, imgfile="~/Desktop/deep-reach.png")
# plt2 = plotReachPolytope(xfs, split_reach_hplanes, imgfile="~/Desktop/split-reach.png")


#
#=
# split_opts = SplitSdpOptions(β=1, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=true)
split_opts = SplitSdpOptions(β=1, x_intervals=rand_x_intvs, slope_intervals=rand_slope_intvs, verbose=true)

split_reach_soln1 = SplitSdp.run(reach_inst1, split_opts)
split_reach_soln2 = SplitSdp.run(reach_inst2, split_opts)
split_reach_soln3 = SplitSdp.run(reach_inst3, split_opts)
split_reach_soln4 = SplitSdp.run(reach_inst4, split_opts)
split_reach_soln5 = SplitSdp.run(reach_inst5, split_opts)
split_reach_soln6 = SplitSdp.run(reach_inst6, split_opts)
split_reach_soln7 = SplitSdp.run(reach_inst7, split_opts)
split_reach_soln8 = SplitSdp.run(reach_inst8, split_opts)

split_reach_hplanes = [
  (reach_inst1.reach_set.normal, split_reach_soln1.values[:h]),
  (reach_inst2.reach_set.normal, split_reach_soln2.values[:h]),
  (reach_inst3.reach_set.normal, split_reach_soln3.values[:h]),
  (reach_inst4.reach_set.normal, split_reach_soln4.values[:h]),
  (reach_inst5.reach_set.normal, split_reach_soln5.values[:h]),
  (reach_inst6.reach_set.normal, split_reach_soln6.values[:h]),
  (reach_inst7.reach_set.normal, split_reach_soln7.values[:h]),
  (reach_inst8.reach_set.normal, split_reach_soln8.values[:h])
]

plt = plotReachPolytope(xfs, split_reach_hplanes, imgfile="~/Desktop/beta1-rand.png")
=#


