# Use during the evaluation
module Methods

using ..Header
using ..Common
using ..Intervals
using ..DeepSdp
using ..SplitSdp
using ..NNetParser
using ..Utils

using LinearAlgebra
using Printf

# Safety
function solveSafetyL2gain(ffnet :: FeedForwardNetwork, input :: BoxInput, opts, L2gain :: Float64; verbose :: Bool = false)
  @assert (opts isa DeepSdpOptions) || (opts isa SplitSdpOptions)
  @assert L2gain > 1e-4 # Not a trivial gain
  safety = L2gainSafety(L2gain, ffnet.xdims)
  safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
  if opts isa DeepSdpOptions
    soln = DeepSdp.run(safety_inst, opts)
  else
    soln = SplitSdp.run(safety_inst, opts)
  end
  return soln
end

# Reachability
function solveReach(ffnet :: FeedForwardNetwork, input :: BoxInput, opts, normal :: VecF64)
  @assert (opts isa DeepSdpOptions) || (opts isa SplitSdpOptions)
  @assert length(normal) == ffnet.xdims[end] == 2 # 2D visualization
  hplane = HyperplaneSet(normal=normal)
  reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane)
  if opts isa DeepSdpOptions
    soln = DeepSdp.run(reach_inst, opts)
  else
    soln = SplitSdp.run(reach_inst, opts)
  end
  return soln
end

# Solve for a polytope
function solveReachPolytope(ffnet :: FeedForwardNetwork, input :: BoxInput, opts, num_hplanes :: Int; verbose :: Bool = false)
  @assert (opts isa DeepSdpOptions) || (opts isa SplitSdpOptions)
  start_time = time()
  hplanes = Vector{Tuple{VecF64, Float64}}()
  for i in 1:num_hplanes
    # Set up the hyperplane normal
    θ = ((i-1) / num_hplanes) * 2 * π
    normal = [cos(θ); sin(θ)]
    if verbose; @printf("\tpoly [%d/%d] with normal θ=%.3f\n", i, num_hplanes, θ) end

    # Run the query
    soln = solveReach(ffnet, input, opts, normal)
    if verbose
      @printf("\t\tsolve time: %.3f, \t objval: %.4f (%s)\n",
              soln.solve_time, soln.objective_value, soln.termination_status)
    end
    push!(hplanes, (normal, soln.objective_value))
  end

  poly_time = time() - start_time
  if verbose; @printf("\ttotal time: %.3f\n", poly_time) end
  return hplanes
end

# Warm up stuff
function warmup(;verbose=false)
  warmup_start_time = time()
  xdims = [2;3;3;3;3;3;3;2]
  ffnet = randomNetwork(xdims)
  x1min = ones(2) .- 1e-2
  x1max = ones(2) .+ 1e-2
  input = BoxInput(x1min=x1min, x1max=x1max)
  L2gain = 100.0
  normal = [1.0; 1.0]

  x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
  deep_opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
  split_opts = SplitSdpOptions(β=1, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)

  # Warm up safety first
  deep_safety_soln = solveSafetyL2gain(ffnet, input, deep_opts, L2gain, verbose=false)
  split_safety_soln = solveSafetyL2gain(ffnet, input, split_opts, L2gain, verbose=false)

  # Then warmup reachability
  deep_reach_soln = solveReach(ffnet, input, deep_opts, normal)
  split_reach_soln = solveReach(ffnet, input, split_opts, normal)

  warmup_time = time() - warmup_start_time
  if verbose; @printf("warmup time: %.3f\n", warmup_time) end
end

# Load up a P1 instance
function loadP1(nnet_filepath :: String, input :: BoxInput; verbose=false, tband = nothing)
  nnet = NNetParser.NNet(nnet_filepath)
  ffnet = Utils.NNet2FeedForwardNetwork(nnet)
  x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
  opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=verbose)
  return ffnet, opts
end

# Load a P2 instance
function loadP2(nnet_filepath :: String, input :: BoxInput, β :: Int; verbose=false, tband_func = nothing)
  nnet = NNetParser.NNet(nnet_filepath)
  ffnet = Utils.NNet2FeedForwardNetwork(nnet)
  x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
  if tband_func isa Nothing
    tband_func = (k, qxdim) -> qxdim
  end
  opts = SplitSdpOptions(
          β = β,
          x_intvs = x_intvs,
          slope_intvs = slope_intvs,
          tband_func = tband_func,
          verbose = verbose)
  return ffnet, opts
end

#
export solveSafetyL2gain
export solveReach, solveReachPolytope
export warmup
export loadP1, loadP2

end

