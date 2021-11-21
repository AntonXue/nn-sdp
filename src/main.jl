
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
# include("split-deep-sdp-a.jl"); using .SplitDeepSdpA
# include("split-deep-sdp-b.jl"); using .SplitDeepSdpB
# include("admm-deep-sdp.jl"); using .AdmmDeepSdp
using LinearAlgebra
using JuMP
using Random

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

xdims = [2; 3; 4; 5; 6; 5; 4; 3; 2]


ffnet = randomReluNetwork(xdims)
K = ffnet.K
input = inputUnitBox(xdims)
safety = safetyNormBound(170.0 ^2, xdims)

plotRandomTrajectories(10000, ffnet)

inst = VerificationInstance(net=ffnet, input=input, safety=safety)
opts2 = VerificationOptions(stride=2)
optsAll = VerificationOptions(stride=(ffnet.K-1))

#
# with xdims = [2; 3; 4; 5; 6; 5; 4; 3; 2], norm^2 <= 170 makes a difference between 2-stride and all-stride

println("Beginning DeepSdp stuff (stride=2)")
soln2 = DeepSdp.run(inst, opts2)
println("DeepSdp solve time: " * string(soln2.solve_time) * ", " * soln2.status)

println("Beginning DeepSdp stuff (stride=all)")
solnA = DeepSdp.run(inst, optsAll)
println("DeepSdp solve time: " * string(solnA.solve_time) * ", " * solnA.status)

