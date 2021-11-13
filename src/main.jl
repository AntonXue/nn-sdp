
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSDP


xdims = [2; 10; 12; 14; 12; 10; 2]
relunet = randomReluNetwork(xdims)
pbox = inputUnitBox(xdims)
safety = safetyNormBound(100, xdims)

soln = DeepSDP.run(relunet, pbox, safety)

