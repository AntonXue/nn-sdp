
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
# include("split-deep-sdp-a.jl"); using .SplitDeepSdpA
# include("split-deep-sdp-b.jl"); using .SplitDeepSdpB
# include("admm-deep-sdp.jl"); using .AdmmDeepSdp
include("lip-sdp.jl"); using .LipSdp
using LinearAlgebra
using JuMP
using Random

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

# Tihs combination makes a difference for 2-stride and all-stride
# xdims = [2; 10; 11; 12; 11; 10; 2]
# norm2 = 1100.0^2
xdims = [2; 30; 30; 30; 2]
# xdims = [2; 3; 3; 2]
norm2 = 20000.0^2

# This combination makes a different with a 2-stride vs 3-stride and all-stride
# xdims = [2; 3; 4; 5; 6; 5; 4; 3; 2]
# norm2 = 170.0^2

ffnet = randomReluNetwork(xdims)
K = ffnet.K
input = inputUnitBox(xdims)
safety = safetyNormBound(norm2, xdims)

#

plotRandomTrajectories(10000, ffnet)

inst = VerificationInstance(net=ffnet, input=input, safety=safety)
opts1 = VerificationOptions(stride=1)
opts2 = VerificationOptions(stride=2)
optsAll = VerificationOptions(stride=(ffnet.K-1))

#

#=
println("Beginning DeepSdp stuff (stride=1)")
soln1 = DeepSdp.run(inst, opts1)
println("DeepSdp solve time: " * string(soln1.solve_time) * ", " * soln1.status)

println("")

println("Beginning DeepSdp stuff (stride=2)")
soln2 = DeepSdp.run(inst, opts2)
println("DeepSdp solve time: " * string(soln2.solve_time) * ", " * soln2.status)
=#

println("")

all_start_time = time()

#=
println("Beginning DeepSdp stuff (stride=all)")
solnA = DeepSdp.run(inst, optsAll)
println("DeepSdp solve time: " * string(solnA.solve_time) * ", " * solnA.status)
=#

println("")

println("Beginning LipSdp stuff")
# model, M2, A, B = LipSdp.setup(ffnet, optsAll)
solnlip = LipSdp.run(ffnet)
println("LipSdp solve time: " * string(solnlip.solve_time) * ", " * solnlip.status)


all_time = time() - all_start_time
println("all time: " * string(all_time))

