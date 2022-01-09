#
main_start_time = time()

#
include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
# include("core/deep-sdp.jl"); using .DeepSdp
include("core/split-sdp.jl"); using .SplitSdp
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

# This pair sees a difference with 300^2, and when tband = 15
# xdims = [2; 20; 20; 20; 20; 20; 2]
# ffnet = randomNetwork(xdims, σ=0.8)


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

opts1 = SplitSdpOptions(β=1, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband_func=((x,y) -> y))
opts2 = SplitSdpOptions(β=2, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband_func=((x,y) -> y))
opts3 = SplitSdpOptions(β=3, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband_func=((x,y) -> y))
opts4 = SplitSdpOptions(β=4, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband_func=((x,y) -> y))

println("solving res1")
res1 = SplitSdp.run(safety_inst, opts1)
println(res1)
println("")

println("solving res2")
res2 = SplitSdp.run(safety_inst, opts2)
println(res2)
println("")

println("solving res3")
res3 = SplitSdp.run(safety_inst, opts3)
println(res3)
println("")

println("solving res4")
res4 = SplitSdp.run(safety_inst, opts4)
println(res4)
println("")

#=
println("solving res1")
res1 = DeepSdp.run(safety_inst, opts1)
println(res1)

println("solving res5")
res5 = DeepSdp.run(safety_inst, opts5)
println(res5)

println("solving res10")
res10 = DeepSdp.run(safety_inst, opts10)
println(res10)

println("solving res15")
res15 = DeepSdp.run(safety_inst, opts15)
println(res15)

println("solving resall")
resall = DeepSdp.run(safety_inst, optsall)
println(resall)
=#

