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
using Printf

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
  inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

  admm_opts = AdmmSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs)
  admm_params = AdmmSdp.initParams(inst, admm_opts)
  admm_cache, cache_time = AdmmSdp.precompute(inst, admm_params, admm_opts)

  @printf("addm cache time: %.3f\n", cache_time)

  # Make some random variables for the safety problem
  β = admm_opts.β
  γvardims = makeγvardims(β, inst, admm_opts.tband_func)
  γin = rand(γvardims[1])
  γks = [rand(d) for d in γvardims[3]]
  γ = vcat([γin; γks]...)
  p3 = admm_cache.J * γ + admm_cache.zaff

  Xin = makeXin(γin, input, ffnet)
  Xout = makeXout(safety.S, ffnet)
  Xs = Vector{Any}()
  for k in 1:(admm_params.num_cliques+1)
    qxdim = Qxdim(k, β, ffnet.zdims)
    xqinfo = Xqinfo(
      ffnet = ffnet,
      ϕout_intv = selectϕoutIntervals(k, β, x_intvs),
      slope_intv = selectSlopeIntervals(k, β, slope_intvs),
      tband = admm_opts.tband_func(k, qxdim))
    Xk = makeXqγ(k, β, γks[k], xqinfo)
    push!(Xs, Xk)
  end

  p2 = (Ein' * Xin * Ein) + (Eout' * Xout * Eout)
  for k = 1:(admm_params.num_cliques+1)
    EXk = [E(k, β, ffnet.zdims); Ea]
    p2 = p2 + (EXk' * Xs[k] * EXk)
  end
  p2 = vec(p2)
  
  maxdiff = maximum(abs.(p2 - p3))
  @printf("safety maxdiff: %.3f\n", maxdiff)
  @assert maxdiff <= 1e-13

  return p2, p3
end

end # End module

