#
start_time = time()
include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/deep-sdp.jl"); using .DeepSdp
include("core/split-sdp.jl"); using .SplitSdp
include("core/admm-sdp.jl"); using .AdmmSdp
include("parsers/nnet-parser.jl"); using .NNetParser
include("utils.jl"); using .Utils

include("methods.jl"); using .Methods

using LinearAlgebra
using ArgParse

println("Imports done: " * string(round(time() - start_time, digits=2)))

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

REACH_WIDTH_DEPTHS =
  [
   (5, 3); (5, 4); (5, 5); (5, 6); (5, 7); (5, 8); (5, 9); (5, 10); (5, 15); (5, 20);
   (10, 3); (10, 4); (10, 5); (10, 6); (10, 7); (10, 8); (10, 9); (10, 10);
  ]

function runReach()
  results = Vector{Any}()
  for (layer_dim, num_layers) in REACH_WIDTH_DEPTHS
    nnet_filename = "rand-in2-out2-ldim" * string(layer_dim) * "-numl" * string(num_layers) * ".nnet"
    nnet_filepath = joinpath(nnet_dir, nnet_filename)
    println("processing NNet: " * nnet_filepath)
    @assert isfile(nnet_filepath)

    # Load the thing
    x1min = ones(2) .- 1e-2
    x1max = ones(2) .+ 1e-2
    input = BoxInput(x1min=x1min, x1max=x1max)
    ffnet, opts = loadP1(nnet_filepath, input)

    # Print the interval propagation bounds, for comparison
    ymin, ymax = opts.x_intvs[end]
    println("\ty1: " * string((ymin[1], ymax[1])))
    println("\ty2: " * string((ymin[2], ymax[2])))

    # Safety stuff
    image_filepath = joinpath(p1_dir, nnet_filename * ".png")
    hplanes, poly_time = solveReachPolytope(ffnet, input, opts, 6, image_filepath)
    xfs = randomTrajectories(10000, ffnet, input.x1min, input.x1max)
    plotReachPolytope(xfs, hplanes, saveto=image_filepath)

    # TODO: write to a text file, probably

    push!(results, (layer_dim, num_layers, poly_time))
    println("")
  end

  return results
end

function runSafety()
  results = Vector{Any}()
  
  for (layer_dim, num_layers) in REACH_WIDTH_DEPTHS
    nnet_filename = "rand-in2-out2-ldim" * string(layer_dim) * "-numl" * string(num_layers) * ".nnet"
    nnet_filepath = joinpath(nnet_dir, nnet_filename)
    println("processing NNet: " * nnet_filepath)
    @assert isfile(nnet_filepath)

    # Load the thing
    x1min = ones(2) .- 1e-2
    x1max = ones(2) .+ 1e-2
    input = BoxInput(x1min=x1min, x1max=x1max)
    ffnet, opts = loadP1(nnet_filepath, input)

    # Print the interval propagation bounds, for comparison
    ymin, ymax = opts.x_intvs[end]
    println("\ty1: " * string((ymin[1], ymax[1])))
    println("\ty2: " * string((ymin[2], ymax[2])))

    # Safety stuff
    image_filepath = joinpath(p1_dir, nnet_filename * ".png")
    norm2 = 10.0
    soln = solveSafetyNorm2(ffnet, input, opts, norm2)
    soln_time = round(soln.total_time, digits=3)
    push!(results, (layer_dim, num_layers, soln_time, string(soln.termination_status)))
  end
  return results
end

println("end time: " * string(round(time() - start_time, digits=2)))

# reach_res = runReach()
safety_res = runSafety()



