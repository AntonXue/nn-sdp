
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("deep-sdp.jl"); using .DeepSdp
include("split-deep-sdp-a.jl"); using .SplitDeepSdpA
include("split-deep-sdp-b.jl"); using .SplitDeepSdpB
include("admm-deep-sdp.jl"); using .AdmmDeepSdp
using LinearAlgebra
using JuMP
using Random


xdims = [2; 20; 30; 20; 20; 30; 15; 2]
# xdims = [9; 8; 7; 6; 5; 6; 7; 8; 9]
# xdims = [10; 14; 12; 8]
# xdims = [2; 3; 4; 5; 6; 5; 4; 3; 2]

# xdims = [8; 9; 10; 11; 10; 9; 8]

zdims = [xdims[1:end-1]; 1]

ffnet = randomReluNetwork(xdims)
input = inputUnitBox(xdims)
safety = safetyNormBound(7.75, xdims) # 8.0 is SAT, 7.75 is not with xdims = [10; 14; 12; 8]
inst = VerificationInstance(net=ffnet, input=input, safety=safety)
K = ffnet.K

#=
println("Beginning DeepSdp stuff")
soln = DeepSdp.run(inst)

println("Beginning SplitDeepSdpA stuff")
solna = SplitDeepSdpA.run(inst)

println("Beginning SplitDeepSdpB stuff")
solnb = SplitDeepSdpB.run(inst)
=#

params = AdmmDeepSdp.initParams(inst)

println("Precompute start")
precomp_start_time = time()

cache = AdmmDeepSdp.precompute(params, inst)


precomp_total_time = time() - precomp_start_time

println("Precompute time: " * string(precomp_total_time))

println("")

admm_start_time = time()
new_params = AdmmDeepSdp.admm(params, cache)

admm_total_time = time() - admm_start_time
println("Admm iter time: " * string(admm_total_time))


