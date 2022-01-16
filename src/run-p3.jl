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

p2_dir = joinpath(args["benchdir"], "p2")
if !isdir(p2_dir)
  mkdir(p2_dir)
end

REACH_WIDTH_DEPTHS =
  [
  (5, 4); (5, 5); (5, 6); (5, 7); (5, 8); (5, 9); (5, 10); (5, 15); (5, 20);
  (10, 4); (10, 5); (10, 6); (10, 7); (10, 8); (10, 9); (10, 10);
  # (15; 3); (15, 4);
  ]


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

    # num_layers is K
    β = min(2, num_layers - 2)
    ffnet, opts = loadP3(nnet_filepath, input, β)

    # Print the interval propagation bounds, for comparison
    ymin, ymax = opts.x_intvs[end]
    println("\ty1: " * string((ymin[1], ymax[1])))
    println("\ty2: " * string((ymin[2], ymax[2])))

    # Safety stuff
    image_filepath = joinpath(p2_dir, nnet_filename * ".png")
    norm2 = 5
    soln = solveSafetyNorm2(ffnet, input, opts, norm2)
    soln_time = round.((soln.setup_time, soln.solve_time, soln.total_time), digits=2)
    push!(results, (layer_dim, num_layers, soln_time, string(soln.termination_status)))
  end
  return results
end


println("end time: " * string(round(time() - start_time, digits=2)))

# reach_res = runReach()
safety_res = runSafety()

#=
nnet_filepath = joinpath(nnet_dir, "rand-in2-out2-ldim5-numl7.nnet")
x1min = ones(2) .- 1e-2
x1max = ones(2) .+ 1e-2
input = BoxInput(x1min=x1min, x1max=x1max)
ffnet, opts = loadP3(nnet_filepath, input, 1)
safety = outputSafetyNorm2(1.0, 1.0, 500, ffnet.xdims)
inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
params = AdmmSdp.initParams(inst, opts)
cache = AdmmSdp.precompute(inst, params, opts)

res = AdmmSdp.run(inst, opts)
=#

