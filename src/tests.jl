module Tests

include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/deep-sdp.jl"); using .DeepSdp
include("core/split-sdp.jl"); using .SplitSdp
include("core/admm-sdp.jl"); using .AdmmSdp
include("parsers/nnet-parser.jl"); using .NNetParser
include("parsers/vnnlib-parser.jl"); using .VnnlibParser
include("utils.jl"); using .Utils

using LinearAlgebra
using Random
using JuMP
using Mosek
using MosekTools

# Test that that Z safety construction by Xk, Yk, and Zk are equivalent.
function _testZk(inst :: QueryInstance, opts :: SplitSdpOptions; verbose=true)
  ffnet = inst.ffnet
  input = inst.input

  # The random variables
  γdims, ξvardims = makeγdims(opts.β, inst, opts.tband_func)
  ξindim, ξsafedim, ξkdims = ξvardims
  
  ξin = abs.(randn(ξindim))
  ξsafe = abs.(randn(ξsafedim))
  ξs = [abs.(randn(ξkdim)) for ξkdim in ξkdims]

  # Slightly different setups depending on instance
  if inst isa SafetyInstance
    safety = inst.safety
    Xsafe = makeSafetyXsafeξ(safety, ffnet)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    hplane = inst.reach_set
    Xsafe = makeHyperplaneReachXsafeξ(ξsafe, hplane, ffnet)
  else
    error("unrecognized instance: " * string(inst))
  end

  # Some helpful block index matrices
  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Ein = [E1; Ea]
  Esafe = [E1; EK; Ea]

  # Construct the X components
  num_Xs = inst.ffnet.K - opts.β
  Xs = Vector{Any}()
  for k = 1:num_Xs
    qxdim = Qxdim(k, opts.β, ffnet.zdims)
    xqinfo = Xqinfo(
      ffnet = inst.ffnet,
      ϕout_intv = selectϕoutIntervals(k, opts.β, opts.x_intvs),
      slope_intv = selectSlopeIntervals(k, opts.β, opts.slope_intvs),
      tband = opts.tband_func(k, qxdim))
    Xk = makeXqξ(k, opts.β, ξs[k], xqinfo)
    push!(Xs, Xk)
  end
  Xin = makeXinξ(ξin, input, ffnet)

  ZXs = (Ein' * Xin * Ein) + (Esafe' * Xsafe * Esafe)
  for k in 1:num_Xs
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    ZXs = ZXs + (EXk' * Xs[k] * EXk)
  end

  # Now construct Z via the Ys
  num_cliques = inst.ffnet.K - opts.β - 1
  @assert num_cliques >= 1

  yinfo = Yinfo(
    inst = inst,
    num_cliques = num_cliques,
    x_intvs = opts.x_intvs,
    slope_intvs = opts.slope_intvs,
    ξvardims = ξvardims,
    tband_func = opts.tband_func)

  Ys = Vector{Any}()
  for k = 1:num_cliques
    if num_cliques == 1 && k == 1
      @assert length(ξs) == 2
      γ1 = [ξin; ξsafe; ξs[1]; ξs[2]]
      Y1 = makeYk(k, opts.β, γ1, yinfo)
      push!(Ys, Y1)

    elseif k == 1
      γ1 = [ξin; ξsafe; ξs[1]]
      Y1 = makeYk(k, opts.β, γ1, yinfo)
      push!(Ys, Y1)

    elseif k == num_cliques
      γp = [ξs[end-1]; ξs[end]]
      Yp = makeYk(k, opts.β, γp, yinfo)
      push!(Ys, Yp)

    else
      γk = ξs[k]
      Yk = makeYk(k, opts.β, γk, yinfo)
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

  @assert maxdiffXYs <= 1e-12 && maxdiffYZs <= 1e-12
end

function testZk(verbose=true)
  Random.seed!(1234)
  xdims = [2;3;4;5;6;5;4;3;2]
  ffnet = randomNetwork(xdims, σ=0.8)

  xcenter = ones(ffnet.xdims[1])
  ε = 0.1
  input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
  x_intvs, ϕintvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

  safety = safetyNormBound(20^2, xdims)
  safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
  opts1 = SplitSdp.SplitSdpOptions(β=1, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)
  opts2 = SplitSdp.SplitSdpOptions(β=2, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)
  opts3 = SplitSdp.SplitSdpOptions(β=3, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)

  if verbose; println("testing safety instances") end
  _testZk(safety_inst, opts1, verbose=verbose)
  _testZk(safety_inst, opts2, verbose=verbose)
  _testZk(safety_inst, opts3, verbose=verbose)

  if verbose; println("testing reachability instances") end
  hplane = HyperplaneSet(normal=[0.0; 1.0])
  reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane)
  _testZk(reach_inst, opts1, verbose=verbose)
  _testZk(reach_inst, opts2, verbose=verbose)
  _testZk(reach_inst, opts3, verbose=verbose)
end

# Test that the P1 and P2 give the expected values for reachability
function testP1P2reach(verbose=true)
  # Don't touch!!
  Random.seed!(12345)
  xdims = [2; 6; 8; 10; 8; 6; 2]
  ffnet = randomNetwork(xdims, σ=0.4)

  xcenter = ones(ffnet.xdims[1])
  ε = 0.01
  input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
  runAndPlotRandomTrajectories(10000, ffnet, x1min=input.x1min, x1max=input.x1max)

  hplane = HyperplaneSet(normal=[0.0; 1.0])
  reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane)

  x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

  deep_opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)
  split_opts1 = SplitSdpOptions(β=1, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)
  split_opts2 = SplitSdpOptions(β=2, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)
  split_opts3 = SplitSdpOptions(β=3, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)
  split_opts4 = SplitSdpOptions(β=4, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)

  # Let us just test the hplane1 reachability for simplicity

  println("running DeepSDP")
  deep_soln = DeepSdp.run(reach_inst, deep_opts)

  println("running SplitSdp with β=1")
  split_soln1 = SplitSdp.run(reach_inst, split_opts1)

  println("running SplitSdp with β=2")
  split_soln2 = SplitSdp.run(reach_inst, split_opts2)

  println("running SplitSdp with β=3")
  split_soln3 = SplitSdp.run(reach_inst, split_opts3)

  println("running SplitSdp with β=4")
  split_soln4 = SplitSdp.run(reach_inst, split_opts4)

  @assert string(deep_soln.termination_status) == "OPTIMAL"
  @assert string(split_soln1.termination_status) == "SLOW_PROGRESS"
  @assert string(split_soln2.termination_status) == "OPTIMAL"
  @assert string(split_soln3.termination_status) == "OPTIMAL"
  @assert string(split_soln4.termination_status) == "OPTIMAL"

  @assert deep_soln.objective_value > 0.036
  @assert abs(deep_soln.objective_value - split_soln4.objective_value) <= 1e-4
  @assert split_soln2.objective_value > split_soln3.objective_value > split_soln4.objective_value

  return deep_soln, split_soln1, split_soln2, split_soln3, split_soln4
end

# Helper function for testing cache
function _setupSafetyViaCache(model, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  setup_start_time = time()
  num_cliques = length(params.γdims)
  Yvars = Dict()
  γs = Vector{Any}()
  for k = 1:num_cliques
    γk = @variable(model, [1:params.γdims[k]])
    @constraint(model, γk[1:params.γdims[k]] .>= 0)
    push!(γs, γk)
    Yvars[Symbol("γ" * string(k))] = γk
  end
  γ = vcat(γs...)

  # Now construct each Zk with a cache
  for k = 1:num_cliques
    ωk = H(k, opts.β, params.γdims) * γ
    zk = cache.Js[k] * ωk + cache.zaffs[k]
    Zkdim = Int(round(sqrt(length(zk))))
    Zk = reshape(zk, (Zkdim, Zkdim))
    @SDconstraint(model, Zk <= 0)
  end

  # Artificial constraint to force strong convexity
  γnorm = @variable(model)
  @constraint(model, [γnorm; γ] in SecondOrderCone())
  @objective(model, Min, γnorm)
  setup_time = round(time() - setup_start_time, digits=3)
  return model, Yvars, setup_time, γ
end

function _runWithAdmmCache(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  total_start_time = time()

  # Set up the instance
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-9))

  _, vars, setup_time, γ = _setupSafetyViaCache(model, params, cache, opts)

  # Run the solve! function equivalent manually
  optimize!(model)
  summary = solution_summary(model)
  solve_time = round(summary.solve_time, digits=3)
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end

  # Set up the thing to return
  total_time = round(time() - total_start_time, digits=3)
  return SolutionOutput(
      objective_value = objective_value(model),
      values = values,
      summary = summary,
      termination_status = summary.termination_status,
      total_time = total_time,
      setup_time = setup_time,
      solve_time = solve_time),
    value.(γ)
end

# The split safety instance, but with an artificial norm constraint injected
function _runSplitCustom(inst :: SafetyInstance, opts :: SplitSdpOptions)
  total_start_time = time()

  # Set up the instance
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-9))

  _, vars, setup_time = SplitSdp.setupSafety!(model, inst, opts)
  γ = Vector{Any}()
  for k in 1:vars.count
    γk = vars[Symbol("γ" * string(k))]
    push!(γ, γk)
  end
  γ = vcat(γ...)

  # Artificial constraint to force strong convexity
  γnorm = @variable(model)
  @constraint(model, [γnorm; γ] in SecondOrderCone())
  @objective(model, Min, γnorm)

  # Solve
  summary, values, solve_time = solve!(model, vars, opts)
  total_time = round(time() - total_start_time, digits=3)
  return SolutionOutput(
      objective_value = objective_value(model),
      values = values,
      summary = summary,
      termination_status = summary.termination_status,
      total_time = total_time,
      setup_time = setup_time,
      solve_time = solve_time),
    value.(γ)
end

# test that the cache is well-formed
function testAdmmCache(verbose=true)
  # Don't touch!!
  Random.seed!(12345)
  xdims = [2; 6; 8; 10; 8; 6; 2]
  ffnet = randomNetwork(xdims, σ=0.5)

  xcenter = ones(ffnet.xdims[1])
  ε = 0.01
  input = BoxInput(x1min=(xcenter .- ε), x1max=(xcenter .+ ε))
  runAndPlotRandomTrajectories(10000, ffnet, x1min=input.x1min, x1max=input.x1max)
  x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

  safety = safetyNormBound(10, xdims)
  safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)

  # The split stuff
  split_opts = SplitSdpOptions(β=2, verbose=verbose, x_intvs=x_intvs, slope_intvs=slope_intvs)
  split_soln, split_γ = _runSplitCustom(safety_inst, split_opts)

  # The admm stuff
  admm_opts = AdmmSdpOptions(β=2, verbose=verbose, x_intvs=x_intvs, slope_intvs=slope_intvs)
  admm_params = initParams(safety_inst, admm_opts)
  admm_cache = precomputeCache(admm_params, safety_inst, admm_opts)
  admm_soln, admm_γ = _runWithAdmmCache(admm_params, admm_cache, admm_opts)

  # Tests differences
  maxdiff = maximum(abs.(split_γ - admm_γ))
  if verbose; println("maxdiff: " * string(maxdiff)) end

  # Return stuff
  return split_soln, admm_soln, admm_cache, split_γ, admm_γ
end

end # End module

