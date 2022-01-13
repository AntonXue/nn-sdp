#
start_time = time()

#
include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/deep-sdp.jl"); using .DeepSdp

include("parsers/nnet-parser.jl"); using .NNetParser
include("parsers/vnnlib-parser.jl"); using .VnnlibParser
include("utils.jl"); using .Utils

using LinearAlgebra
using ArgParse

println("Finished importing: " * string(round(time() - start_time, digits=2)))

#
argparse_settings = ArgParseSettings()
@add_arg_table argparse_settings begin
    "--benchdir"
        help = "the NNet file location"
        arg_type = String
end

args = parse_args(ARGS, argparse_settings)

# Make sure the relevant directories exist
@assert isdir(args["benchdir"])

nnet_dir = joinpath(args["benchdir"], "nnet")
@assert isdir(nnet_dir)

p1_dir = joinpath(args["benchdir"], "p1")
if !isdir(p1_dir)
  mkdir(p1_dir)
end



# LAYER_DIMS = [5; 10]
# NUM_LAYERS = [5; 10; 15]

LAYER_DIMS = [5; 10; 15]
NUM_LAYERS = [5; 10]

# Solve each instance
function solveReach(ffnet :: FeedForwardNetwork, input :: BoxInput, opts :: DeepSdpOptions, normal :: Vector{Float64})
  @assert length(normal) == ffnet.xdims[end] == 2
  hplane = HyperplaneSet(normal=normal)
  reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane)
  soln = DeepSdp.run(reach_inst, opts)
  return soln
end

# Solve for a polytope
function solveReachPolytope(ffnet :: FeedForwardNetwork, input :: BoxInput, opts :: DeepSdpOptions, num_hplanes :: Int, imgfile :: String)
  start_time = time()
  bounds = Vector{Float64}()
  hplanes = Vector{Tuple{Vector{Float64}, Float64}}()
  for i in 1:num_hplanes
    println("\tsetting p " * string(i) * "/" * string(num_hplanes) * "")
    θ = ((i-1) / num_hplanes) * 2 * π
    normal = [cos(θ); sin(θ)]
    println("\t\tnormal: " * string(normal))
    soln = solveReach(ffnet, input, opts, normal)
    println("\t\tobjval: " * string(soln.objective_value))
    println("\t\tstatus: " * string(soln.termination_status))
    println("\t\tsolvet: " * string(soln.solve_time))
    push!(hplanes, (normal, soln.objective_value))
  end

  println("\tdone solving: " * string(round(time() - start_time, digits=2)))

  # Generate some random points and shove them through
  xfs = randomTrajectories(10000, ffnet, input.x1min, input.x1max)
  plotReachPolytope(xfs, hplanes, imgfile=imgfile)
  println("\tdone plotting: " * string(round(time() - start_time, digits=2)))
end

# Safety
function solveSafetyNorm2(ffnet :: FeedForwardNetwork, input :: BoxInput, opts :: DeepSdpOptions, norm2 :: Float64)
  @assert norm2 > 1e-4
  safety = outputSafetyNorm2(1.0, 1.0, norm2, ffnet.xdims)
  safety_inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
  soln = DeepSdp.run(safety_inst, opts)
  return soln
end

#
solns = Vector{Any}()

for layer_dim in LAYER_DIMS
  for num_layers in NUM_LAYERS
    nnet_filename = "rand-in2-out2-ldim" * string(layer_dim) * "-numl" * string(num_layers) * ".nnet"
    nnet_filepath = joinpath(nnet_dir, nnet_filename)
    @assert isfile(nnet_filepath)
    nnet = NNetParser.NNet(nnet_filepath)
    ffnet = Utils.NNet2FeedForwardNetwork(nnet)
    println("processing NNet: " * nnet_filepath)

    x1min = ones(nnet.inputSize) .- 1e-2
    x1max = ones(nnet.inputSize) .+ 1e-2
    input = BoxInput(x1min=x1min, x1max=x1max)
    x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
    ymin, ymax = x_intvs[end]
    println("\ty1: " * string((ymin[1], ymax[1])))
    println("\ty2: " * string((ymin[2], ymax[2])))

    # Safety stuff
    #=
    opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
    rand_x_intvs, _, _ = randomizedPropagation(input.x1min, input.x1max, ffnet, 10000)
    rymin, rymax = rand_x_intvs[end]
    println("\try1: " * string((rymin[1], rymax[1])))
    println("\try2: " * string((rymin[2], rymax[2])))

    norm2 = 2 + 2 * norm([rymin; rymax])^2
    println("\ttgt norm2: " * string(norm2))

    soln = solveSafetyNorm2(ffnet, input, opts, norm2)
    println("\tobjval: " * string(soln.objective_value))
    println("\tstatus: " * string(soln.termination_status))
    println("\tsolve time: " * string(round(soln.solve_time, digits=2)))
    println("")
    push!(solns, soln)
    =#

    # Uncomment for hplane reachability
    opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)

    img_filepath = joinpath(p1_dir, nnet_filename * ".png")
    solveReachPolytope(ffnet, input, opts, 6, img_filepath)
    println("")
  end
end

println("end time: " * string(round(time() - start_time, digits=2)))



