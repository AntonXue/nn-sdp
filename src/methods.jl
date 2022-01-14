# Use during the evaluation
module Methods

using ..Header
using ..Common
using ..Intervals
using ..Partitions
using ..DeepSdp
using ..SplitSdp
using ..AdmmSdp
using ..NNetParser
using ..Utils

using LinearAlgebra

# Reachability
function solveReach(ffnet :: FeedForwardNetwork, input :: BoxInput, opts, normal :: Vector{Float64})
  @assert (opts isa DeepSdpOptions) || (opts isa SplitSdpOptions)
  @assert length(normal) == ffnet.xdims[end] == 2
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
function solveReachPolytope(ffnet :: FeedForwardNetwork, input :: BoxInput, opts, num_hplanes :: Int, saveto :: String; verbose :: Bool = true)
  @assert (opts isa DeepSdpOptions) || (opts isa SplitSdpOptions)
  start_time = time()
  hplanes = Vector{Tuple{Vector{Float64}, Float64}}()
  for i in 1:num_hplanes
    if verbose; println("\tsetting p " * string(i) * "/" * string(num_hplanes) * "") end

    # Set up the hyperplane normal
    θ = ((i-1) / num_hplanes) * 2 * π
    normal = [cos(θ); sin(θ)]
    if verbose; println("\t\tnormal: " * string(normal)) end

    # Run the query
    soln = solveReach(ffnet, input, opts, normal)
    if verbose; println("\t\tobjval: " * string(soln.objective_value)) end
    if verbose; println("\t\tstatus: " * string(soln.termination_status)) end
    if verbose; println("\t\tsolv t: " * string(soln.solve_time)) end

    push!(hplanes, (normal, soln.objective_value))
  end

  poly_time = time() - start_time
  if verbose; println("\tdone t: " * string(round(poly_time, digits=2))) end
  return hplanes, poly_time
end

# Safety
function solveSafetyNorm2(ffnet :: FeedForwardNetwork, input :: BoxInput, opts, norm2 :: Float64; verbose :: Bool = true)
  @assert (opts isa DeepSdpOptions) || (opts isa SplitSdpOptions) || (opts isa AdmmSdpOptions)
  @assert norm2 > 1e-4
  safety = outputSafetyNorm2(1.0, 1.0, norm2, ffnet.xdims)
  safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
  if opts isa DeepSdpOptions
    soln = DeepSdp.run(safety_inst, opts)
  elseif opts isa SplitSdpOptions
    soln = SplitSdp.run(safety_inst, opts)
  else
    soln = AdmmSdp.run(safety_inst, opts)
    println("\ttimes: " * string(round.((soln.setup_time, soln.solve_time, soln.total_time), digits=2)))
    println("\tstatus: " * string(soln.termination_status))
  end
  return soln
end

# Load up a P1 instance
function loadP1(nnet_filepath :: String, input :: BoxInput)
  nnet = NNetParser.NNet(nnet_filepath)
  ffnet = Utils.NNet2FeedForwardNetwork(nnet)
  x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
  opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
  return ffnet, opts
end

# Load a P2 instance
function loadP2(nnet_filepath :: String, input :: BoxInput, β :: Int)
  nnet = NNetParser.NNet(nnet_filepath)
  ffnet = Utils.NNet2FeedForwardNetwork(nnet)
  x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
  opts = SplitSdpOptions(β=β, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
  return ffnet, opts
end

# Load a P3 instance
function loadP3(nnet_filepath :: String, input :: BoxInput, β :: Int)
  nnet = NNetParser.NNet(nnet_filepath)
  ffnet = Utils.NNet2FeedForwardNetwork(nnet)
  x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
  # x_intvs, _, slope_intvs = randomizedPropagation(input.x1min, input.x1max, ffnet, 100000)
  opts = AdmmSdpOptions(β=β, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
  return ffnet, opts
end

#
export solveReach, solveReachPolytope
export solveSafetyNorm2
export loadP1, loadP2, loadP3

end
