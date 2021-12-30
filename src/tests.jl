module Tests

using ..Header
using ..Common
using ..Partitions
using ..DeepSdp
using ..SplitSdp
using ..Intervals
using ..Utils

using LinearAlgebra
using Random
using JuMP
using Mosek
using MosekTools


# Test that that Z safety construction by Xk, Yk, and Zk are equivalent.
function testSafetyZk(inst :: SafetyInstance, opts :: SplitSdpOptions, verbose=true)
  ffnet = inst.ffnet
  input = inst.input
  safety = inst.safety

  γdims, ξvardims = makeγdims(opts.β, inst)
  ξindim, ξsafedim, ξkdims = ξvardims
  
  ξin = abs.(randn(ξindim))
  ξsafe = abs.(randn(ξsafedim))
  ξs = [abs.(randn(ξkdim)) for ξkdim in ξkdims]

  # Some helpful block index matrices
  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Ein = [E1; Ea]
  Esafe = [E1; EK; Ea]

  # Construct the X components
  num_Xs = inst.ffnet.K - opts.β
  Xks = Vector{Any}()
  Xs = [makeXqξ(k, opts.β, ξs[k], ffnet) for k in 1:num_Xs]
  Xin = makeXinξ(ξin, input, ffnet)
  Xsafe = makeSafetyXsafeξ(safety, ffnet)

  ZXs = (Ein' * Xin * Ein) + (Esafe' * Xsafe * Esafe)
  for k in 1:num_Xs
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    ZXs = ZXs + (EXk' * Xs[k] * EXk)
  end

  # Now construct Z via the Ys
  num_cliques = inst.ffnet.K - opts.β - 1
  @assert num_cliques > 1
  Ys = Vector{Any}()
  for k = 1:num_cliques
    if num_cliques == 1 && k == 1
      @assert length(ξs) == 2
      γ1 = [ξin; ξsafe; ξs[1]; ξs[2]]
      Y1 = makeYk(k, opts.β, γ1, ξvardims, inst)
      push!(Ys, Y1)

    elseif k == 1
      γ1 = [ξin; ξsafe; ξs[1]]
      Y1 = makeYk(k, opts.β, γ1, ξvardims, inst)
      push!(Ys, Y1)

    elseif k == num_cliques
      γp = [ξs[end-1]; ξs[end]]
      Yp = makeYk(k, opts.β, γp, ξvardims, inst)
      push!(Ys, Yp)

    else
      γk = ξs[k]
      Yk = makeYk(k, opts.β, γk, ξvardims, inst)
      push!(Ys, Yk)
    end
  end

  ZYs = sum(Ec(k, opts.β, ffnet.zdims)' * Ys[k] * Ec(k, opts.β, ffnet.zdims) for k in 1:num_cliques)

  # Now construct each Zk
  Ωinvs = makeΩinvs(opts.β, ffnet.zdims)
  Zs = Vector{Any}()
  for k = 1:num_cliques
    Zk = makeZk(k, opts.β, Ys, Ωinvs, ffnet.zdims)
    push!(Zs, Zk)
  end

  ZZs = sum(Ec(k, opts.β, ffnet.zdims)' * Zs[k] * Ec(k, opts.β, ffnet.zdims) for k in 1:num_cliques)

  maxdiffXYs = maximum(abs.(ZXs - ZYs))
  maxdiffYZs = maximum(abs.(ZYs - ZZs))

  if verbose; println("maxdiff XY: " * string(maxdiffXYs)) end
  if verbose; println("maxdiff YZ: " * string(maxdiffYZs)) end

  @assert maxdiffXYs <= 1e-13 && maxdiffYZs <= 1e-13
end

# Test that that Z reachability construction by Xk, Yk, and Zk are equivalent.
function testReachZk(inst :: ReachabilityInstance, opts :: SplitSdpOptions, verbose=true)
  ffnet = inst.ffnet
  input = inst.input
  hplane = inst.reach_set

  @assert hplane isa HyperplaneSet

  γdims, ξvardims = makeγdims(opts.β, inst)
  ξindim, ξsafedim, ξkdims = ξvardims
  
  ξin = abs.(randn(ξindim))
  ξsafe = abs.(randn(ξsafedim))
  ξs = [abs.(randn(ξkdim)) for ξkdim in ξkdims]

  # Some helpful block index matrices
  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Ein = [E1; Ea]
  Esafe = [E1; EK; Ea]

  # Construct the X components
  num_Xs = inst.ffnet.K - opts.β
  Xks = Vector{Any}()
  Xs = [makeXqξ(k, opts.β, ξs[k], ffnet) for k in 1:num_Xs]
  Xin = makeXinξ(ξin, input, ffnet)
  Xsafe = makeHyperplaneReachXsafeξ(ξsafe, hplane, ffnet)

  ZXs = (Ein' * Xin * Ein) + (Esafe' * Xsafe * Esafe)
  for k in 1:num_Xs
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    ZXs = ZXs + (EXk' * Xs[k] * EXk)
  end

  # Now construct Z via the Ys
  num_cliques = inst.ffnet.K - opts.β - 1
  @assert num_cliques > 1
  Ys = Vector{Any}()
  for k = 1:num_cliques
    if num_cliques == 1 && k == 1
      @assert length(ξs) == 2
      γ1 = [ξin; ξsafe; ξs[1]; ξs[2]]
      Y1 = makeYk(k, opts.β, γ1, ξvardims, inst)
      push!(Ys, Y1)

    elseif k == 1
      γ1 = [ξin; ξsafe; ξs[1]]
      Y1 = makeYk(k, opts.β, γ1, ξvardims, inst)
      push!(Ys, Y1)

    elseif k == num_cliques
      γp = [ξs[end-1]; ξs[end]]
      Yp = makeYk(k, opts.β, γp, ξvardims, inst)
      push!(Ys, Yp)

    else
      γk = ξs[k]
      Yk = makeYk(k, opts.β, γk, ξvardims, inst)
      push!(Ys, Yk)
    end
  end

  ZYs = sum(Ec(k, opts.β, ffnet.zdims)' * Ys[k] * Ec(k, opts.β, ffnet.zdims) for k in 1:num_cliques)

  # Now construct each Zk
  Ωinvs = makeΩinvs(opts.β, ffnet.zdims)
  Zs = Vector{Any}()
  for k = 1:num_cliques
    Zk = makeZk(k, opts.β, Ys, Ωinvs, ffnet.zdims)
    push!(Zs, Zk)
  end

  ZZs = sum(Ec(k, opts.β, ffnet.zdims)' * Zs[k] * Ec(k, opts.β, ffnet.zdims) for k in 1:num_cliques)

  maxdiffXYs = maximum(abs.(ZXs - ZYs))
  maxdiffYZs = maximum(abs.(ZYs - ZZs))

  if verbose; println("maxdiff XY: " * string(maxdiffXYs)) end
  if verbose; println("maxdiff YZ: " * string(maxdiffYZs)) end

  @assert maxdiffXYs <= 1e-13 && maxdiffYZs <= 1e-13
end

# Test that the P1 and P2 give the expected values for reachability
function testP1P2reach(verbose=true)
  Random.seed!(12345)

  #xdims = [2; 3; 4; 5; 4; 3; 2]
  xdims = [2; 2; 2; 2; 2; 2]
  ffnet = randomNetwork(xdims, σ=1.0)

  xcenter = ones(ffnet.xdims[1])
  ε = 0.1
  input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
  runAndPlotRandomTrajectories(10000, ffnet, x1min=input.x1min, x1max=input.x1max)

  hplane = HyperplaneSet(normal=[0.0; 1.0])
  reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane)

  #=
  hplane1 = HyperplaneSet(normal=[0.0; 1.0])
  hplane2 = HyperplaneSet(normal=[1.0; 1.0])
  hplane3 = HyperplaneSet(normal=[1.0; 0.0])
  hplane4 = HyperplaneSet(normal=[1.0; -1.0])
  hplane5 = HyperplaneSet(normal=[0.0; -1.0])
  hplane6 = HyperplaneSet(normal=[-1.0; -1.0])
  hplane7 = HyperplaneSet(normal=[-1.0; 0.0])
  hplane8 = HyperplaneSet(normal=[-1.0; 1.0])

  reach_inst1 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane1)
  reach_inst2 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane2)
  reach_inst3 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane3)
  reach_inst4 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane4)
  reach_inst5 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane5)
  reach_inst6 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane6)
  reach_inst7 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane7)
  reach_inst8 = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane8)
  =#

  x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

  deep_opts = DeepSdpOptions(x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=verbose)
  split_opts1 = SplitSdpOptions(β=1, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=verbose)
  split_opts2 = SplitSdpOptions(β=2, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=verbose)
  split_opts3 = SplitSdpOptions(β=3, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=verbose)
  split_opts4 = SplitSdpOptions(β=4, x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=verbose)

  # Let us just test the hplane1 reachability for simplicity

  #=
  deep_soln = DeepSdp.run(reach_inst, deep_opts)
  println("\n\n")
  =#

  split_soln1 = SplitSdp.run(reach_inst, split_opts1)
  println("\n\n")

  split_soln2 = SplitSdp.run(reach_inst, split_opts2)
  println("value: " * string(split_soln2.values[:h]))
  println("\n\n")

  split_soln3 = SplitSdp.run(reach_inst, split_opts3)
  println("value: " * string(split_soln3.values[:h]))
  println("\n\n")

  # split_soln4 = SplitSdp.run(reach_inst, split_opts4)
  # println("value: " * string(split_soln4.values[:h]))
  # println("\n\n")

  return split_soln1, split_soln2, split_soln3

  # return deep_soln, split_soln1, split_soln2, split_soln3, split_soln4


end


end # End module

