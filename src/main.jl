
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
include("split-deep-sdp-a.jl"); using .SplitDeepSdpA
include("split-deep-sdp-b.jl"); using .SplitDeepSdpB
# include("admm-deep-sdp.jl"); using .AdmmDeepSdp
using JuMP
using Random


# xdims = [2; 20; 30; 20; 20; 30; 2]
xdims = [10; 14; 12; 8]
zdims = [xdims[1:end-1]; 1]

relunet = randomReluNetwork(xdims)
pbox = inputUnitBox(xdims)
safety = safetyNormBound(8, xdims)
inst = VerificationInstance(net=relunet, input=pbox, safety=safety)

println("Beginning DeepSdp stuff")
soln = DeepSdp.run(inst)

println("Beginning SplitDeepSdpA stuff")
solna = SplitDeepSdpA.run(inst)

println("Beginning SplitDeepSdpB stuff")
solnb = SplitDeepSdpB.run(inst)


