
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

# Seed is set, but make sure all random calls happen in the same sequence
Random.seed!(1234)

# xdims = [2; 20; 30; 20; 20; 30; 15; 2]
# xdims = [9; 8; 7; 6; 5; 6; 7; 8; 9]
xdims = [10; 14; 12; 8]
# xdims = [2; 3; 4; 5; 6; 5; 4; 3; 2]

# xdims = [2; 40; 40; 40; 40; 40; 2]

# xdims = [8; 9; 10; 11; 10; 9; 8]

zdims = [xdims[1:end-1]; 1]

ffnet = randomReluNetwork(xdims)
input = inputUnitBox(xdims)
safety = safetyNormBound(8.0, xdims) # 8.0 is SAT, 7.8 is not with xdims = [10; 14; 12; 8]
inst = VerificationInstance(net=ffnet, input=input, safety=safety)
K = ffnet.K

#=
println("Beginning DeepSdp stuff")
soln = DeepSdp.run(inst)
println("DeepSdp solve time: " * string(soln.solve_time) * ", " * soln.status)

println("Beginning SplitDeepSdpA stuff")
solna = SplitDeepSdpA.run(inst)
println("SplitDeepSdpA solve time: " * string(solna.solve_time) * ", " * solna.status)

println("")

println("Beginning SplitDeepSdpB stuff")
solnb = SplitDeepSdpB.run(inst)
println("SplitDeepSdpB solve time: " * string(solnb.solve_time) * ", " * solnb.status)

println("")
=#

# ADMM tests
#
start_params = AdmmDeepSdp.initParams(inst)

println("Precompute start")
precomp_start_time = time()
cache = AdmmDeepSdp.precompute(start_params, inst)
precomp_total_time = time() - precomp_start_time
println("Precompute time: " * string(precomp_total_time))

println("")

println("Beginning ADMM iterations")
admm_start_time = time()
new_params = AdmmDeepSdp.admm(start_params, cache)

admm_total_time = time() - admm_start_time
println("Admm iter time: " * string(admm_total_time))


### test for ADMM coherence

# Extract the final Zks that would have been generated with new_params


#=
solnbγ = value.(solnb.model[:γ])
solnbω1 = Hc(1, new_params.γdims) * solnbγ
solnbω2 = Hc(2, new_params.γdims) * solnbγ
solnbω3 = Hc(3, new_params.γdims) * solnbγ

solnbz1 = AdmmDeepSdp.zk(1, solnbω1, cache)
solnbz2 = AdmmDeepSdp.zk(2, solnbω2, cache)
solnbz3 = AdmmDeepSdp.zk(3, solnbω3, cache)

solnbZ1 = reshape(solnbz1, (Int(round(sqrt(length(solnbz1)))), Int(round(sqrt(length(solnbz1))))))
solnbZ2 = reshape(solnbz2, (Int(round(sqrt(length(solnbz2)))), Int(round(sqrt(length(solnbz2))))))
solnbZ3 = reshape(solnbz3, (Int(round(sqrt(length(solnbz3)))), Int(round(sqrt(length(solnbz3))))))
=#



new_Zks = Vector{Any}()
for k = 1:K
  ωk = Hc(k, new_params.γdims) * new_params.γ
  zk = cache.Js[k] * ωk + cache.zaffs[k]
  dim = Int(round(sqrt(length(zk))))
  push!(new_Zks, reshape(zk, (dim, dim)))
end

Js = cache.Js
zaffs = cache.zaffs

for k = 1:K
  ωk = Hc(k, new_params.γdims) * new_params.γ
  _Zkb = SplitDeepSdpB.Zk(k, ωk, new_params.γdims, new_params.zdims, input, safety, ffnet)
  _zkb = vec(_Zkb)
  _zkadmm = Js[k] * ωk + zaffs[k]
  println("norm diff at k = " * string(k) * ": " * string(norm(_zkb - _zkadmm)))
end

