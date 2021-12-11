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
xdims = [2; 3; 4; 5; 6; 7; 6; 5; 4; 3; 2]
# xdims = [2; 40; 40; 40; 40; 40; 40; 2]
# xdims = [2; 20; 20; 20; 20; 20; 2]

ffnet = randomNetwork(xdims, σ=0.8)

# Plot some trajectories
runAndPlotRandomTrajectories(10000, ffnet)

input = inputUnitBox(xdims)
norm2 = 0.5
safety = safetyNormBound(norm2, xdims)
safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

normals = [[0;1], [1;1], [1;0], [1;-1], [0;-1], [-1;-1], [-1;0], [-1;1]]
hplanes = HyperplanesConstraint(normals=normals)
reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, hplanes=hplanes)

deepopts = DeepSdpOptions(verbose=true)



# reach_soln = DeepSdp.run(reach_inst, deepopts)

#=
inst1 = VerificationInstance(net=ffnet, input=input, safety=safety, β=1, pattern=BandedPattern(tband=10))
inst2 = VerificationInstance(net=ffnet, input=input, safety=safety, β=2, pattern=BandedPattern(tband=10))
inst3 = VerificationInstance(net=ffnet, input=input, safety=safety, β=3, pattern=BandedPattern(tband=10))
inst4 = VerificationInstance(net=ffnet, input=input, safety=safety, β=4, pattern=BandedPattern(tband=10))
inst5 = VerificationInstance(net=ffnet, input=input, safety=safety, β=5, pattern=BandedPattern(tband=10))
=#

# big_deepopts = DeepSdpOptions(setupMethod=BigSetup(), verbose=true)
# sumx_deepopts = DeepSdpOptions(setupMethod=SumXSetup(), verbose=true)
# solndeep = DeepSdp.run(inst, deepopts)
# println(solndeep)

# sumx_splitopts = SplitSdpOptions(setupMethod=SumXThenSplitSetup(), verbose=true)
# solnsplit = SplitSdp.run(inst, splitopts)
# println(solnsplit)



# Tests.testXSumWhole(inst)


