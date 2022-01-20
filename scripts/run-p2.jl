#
start_time = time()
include("../src/core/header.jl"); using .Header
include("../src/core/common.jl"); using .Common
include("../src/core/intervals.jl"); using .Intervals
include("../src/core/deep-sdp.jl"); using .DeepSdp
include("../src/core/split-sdp.jl"); using .SplitSdp
include("../src/core/admm-sdp.jl"); using .AdmmSdp
include("../src/parsers/nnet-parser.jl"); using .NNetParser
include("../src/utils.jl"); using .Utils
include("../src/methods.jl"); using .Methods

using LinearAlgebra
using ArgParse
using Printf

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

p2_dir = joinpath(args["benchdir"], "p2")
if !isdir(p2_dir)
  mkdir(p2_dir)
end

WIDTHS = [5, 10, 15, 20]
DEPTHS = [5, 10, 15, 20, 25, 30, 35, 40]

REACH_WIDTH_DEPTHS = [(w, d) for w in WIDTHS for d in DEPTHS]

function runReach(β :: Int)
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
    ffnet, opts = loadP2(nnet_filepath, input, β)

    # Print the interval propagation bounds, for comparison
    ymin, ymax = opts.x_intvs[end]
    # println("\ty1: " * string((ymin[1], ymax[1])))
    # println("\ty2: " * string((ymin[2], ymax[2])))

    # Safety stuff
    aug_nnet_filename = "β" * string(β) * "_" * nnet_filename
    image_filepath = joinpath(p2_dir, aug_nnet_filename * ".png")
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
    @printf("processing NNet: %s\n", nnet_filepath)
    @assert isfile(nnet_filepath)

    # Load the thing
    x1min = ones(2) .- 1e-2
    x1max = ones(2) .+ 1e-2
    input = BoxInput(x1min=x1min, x1max=x1max)

    # num_layers is K
    β = min(1, num_layers - 2)
    ffnet, opts = loadP2(nnet_filepath, input, β)

    # Print the interval propagation bounds, for comparison
    ymin, ymax = opts.x_intvs[end]
    # println("\ty1: " * string((ymin[1], ymax[1])))
    # println("\ty2: " * string((ymin[2], ymax[2])))

    # Safety stuff
    image_filepath = joinpath(p2_dir, nnet_filename * ".png")
    norm2 = 1e6
    # norm2 = 1e6
    soln = solveSafetyNorm2(ffnet, input, opts, norm2)
    soln_time = round(soln.total_time, digits=3)
    @printf("\tstatus: %s \t (%d,%d) \ttime: %.3f\n", soln.termination_status, layer_dim, num_layers, soln.total_time)
    push!(results, (layer_dim, num_layers, soln_time, string(soln.termination_status)))
  end
  return results
end


println("end time: " * string(round(time() - start_time, digits=2)))

# reach_res1 = runReach(1)
# reach_res2 = runReach(2)
safety_res = runSafety()



