
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
# include("split-deep-sdp-a.jl"); using .SplitDeepSdpA
# include("split-deep-sdp-b.jl"); using .SplitDeepSdpB
# include("admm-deep-sdp.jl"); using .AdmmDeepSdp
# include("lip-sdp.jl"); using .LipSdp
using LinearAlgebra
using JuMP
using Random

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

# Tihs combination makes a difference for 2-stride and all-stride
# xdims = [2; 3; 4; 5; 4; 3; 2]
xdims = [2; 10; 10; 10; 10; 2]
ffnet = randomNetwork(xdims, σ=0.8)

# Plot some trajectories
runAndPlotRandomTrajectories(10000, ffnet)


input = inputUnitBox(xdims)
norm2=4000
safety = safetyNormBound(norm2, xdims)

inst = VerificationInstance(net=ffnet, input=input, safety=safety, β=3, pattern=BandedPattern(tband=1000))


opts1 = DeepSdpOptions(setupMethod=SimpleSetup(), verbose=true)
soln1 = DeepSdp.run(inst, opts1)
println(soln1)



