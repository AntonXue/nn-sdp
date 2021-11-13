
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSDP
include("split-deep-sdp-a.jl"); using .SplitDeepSDPa


xdims = [2; 20; 30; 20; 20; 30; 2]
zdims = [xdims[1:end-1]; 1]


relunet = randomReluNetwork(xdims)
pbox = inputUnitBox(xdims)
safety = safetyNormBound(100, xdims)

println("Beginning DeepSDP stuff")
soln1 = DeepSDP.run(relunet, pbox, safety)

println("Beginning SplitDeepSDPa stuff")
solna = SplitDeepSDPa.run(relunet, pbox, safety)

