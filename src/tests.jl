module Tests

include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/deep-sdp.jl"); using .DeepSdp
include("core/split-sdp.jl"); using .SplitSdp
include("core/admm-sdp.jl"); using .AdmmSdp
include("parsers/nnet-parser.jl"); using .NNetParser
include("utils.jl"); using .Utils

using LinearAlgebra
using Random
using JuMP
using Mosek
using MosekTools

# Set up vec(Z) in two different ways and see if things make sense
function testAdmmCache(verbose=true)
  # Don't touch!!
  Random.seed!(12345)
  xdims = [2; 6; 8; 10; 8; 6; 2]
  ffnet = randomNetwork(xdims, σ=0.5)

  xcenter = ones(ffnet.xdims[1])
  x1min = xcenter .- 1e-2
  x1max = xcenter .+ 1e-2
  input = BoxInput(x1min=x1min, x1max=x1max)
  x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
  
  # Some block matrices
  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Ein = [E1; Ea]
  Eout = [E1; EK; Ea]

  # The safety instance
  safety = outputSafetyNorm2(1.0, 1.0, 100, xdims)
  safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

  admm_safety_opts = AdmmSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs)
  admm_safety_params = AdmmSdp.initParams(safety_inst, admm_safety_opts)
  admm_safety_cache = AdmmSdp.precompute(safety_inst, admm_safety_params, admm_safety_opts)

  # Make some random variables for the safety problem
  safety_β = admm_safety_opts.β
  safety_γvardims = makeγvardims(safety_β, safety_inst, admm_safety_opts.tband_func)
  safety_γin = rand(safety_γvardims[1])
  safety_γks = [rand(d) for d in safety_γvardims[3]]
  safety_γ = vcat([safety_γin; safety_γks]...)
  safety_p3 = admm_safety_cache.J * safety_γ + admm_safety_cache.zaff

  safety_Xin = makeXin(safety_γin, input, ffnet)
  safety_Xout = makeXout(safety.S, ffnet)
  safety_Xs = Vector{Any}()
  for k in 1:(admm_safety_params.num_cliques+1)
    qxdim = Qxdim(k, safety_β, ffnet.zdims)
    xqinfo = Xqinfo(
      ffnet = ffnet,
      ϕout_intv = selectϕoutIntervals(k, safety_β, x_intvs),
      slope_intv = selectSlopeIntervals(k, safety_β, slope_intvs),
      tband = admm_safety_opts.tband_func(k, qxdim))
    Xk = makeXqγ(k, safety_β, safety_γks[k], xqinfo)
    push!(safety_Xs, Xk)
  end

  safety_p2 = (Ein' * safety_Xin * Ein) + (Eout' * safety_Xout * Eout)
  for k = 1:(admm_safety_params.num_cliques+1)
    EXk = [E(k, safety_β, ffnet.zdims); Ea]
    safety_p2 = safety_p2 + (EXk' * safety_Xs[k] * EXk)
  end
  safety_p2 = vec(safety_p2)


  # The verification instance
  hplane = HyperplaneSet(normal=[1.0; 1.0])
  reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane)

  admm_reach_opts = AdmmSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs)
  admm_reach_params = AdmmSdp.initParams(reach_inst, admm_reach_opts)
  admm_reach_cache = AdmmSdp.precompute(reach_inst, admm_reach_params, admm_reach_opts)
  
  # Make some random variables for the reachability problem
  
  safety_maxdiff = maximum(abs.(safety_p2 - safety_p3))
  println("safety maxdiff: " * string(safety_maxdiff))
  @assert safety_maxdiff <= 1e-13

  return safety_p2, safety_p3
end

end # End module

