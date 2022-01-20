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

@printf("Imports done: %.3f\n", (time() - start_time))

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
    ffnet, opts = loadP3(nnet_filepath, input, β)

    # Print the interval propagation bounds, for comparison
    ymin, ymax = opts.x_intvs[end]
    # @printf("\ty1: (%.3f, %.3f)\n", ymin[1], ymax[1])
    # @printf("\ty2: (%.3f, %.3f)\n", ymin[2], ymax[2])

    # Safety stuff
    image_filepath = joinpath(p2_dir, nnet_filename * ".png")
    norm2 = 1e6
    soln = solveSafetyNorm2(ffnet, input, opts, norm2)
    time_str = @sprintf("(%.3f, %.3f, %.3f)", soln.setup_time, soln.solve_time, soln.total_time)
    @printf("\tstatus: %s \t (%d, %d) \ttimes: %s\n", soln.termination_status, layer_dim, num_layers, time_str)
    soln_time = round.((soln.setup_time, soln.solve_time, soln.total_time), digits=2)
    push!(results, (layer_dim, num_layers, soln_time, string(soln.termination_status)))
  end
  return results
end

@printf("here time: %.3f\n", (time() - start_time))
@printf("\n")

# reach_res = runReach()
safety_res = runSafety()

#=
nnet_filepath = joinpath(nnet_dir, "rand-in2-out2-ldim5-numl5.nnet")
x1min = ones(2) .- 1e-2
x1max = ones(2) .+ 1e-2
input = BoxInput(x1min=x1min, x1max=x1max)
ffnet, opts = loadP3(nnet_filepath, input, 1; verbose=true)
zdims = ffnet.zdims
safety = outputSafetyNorm2(1.0, 1.0, 500, ffnet.xdims)
inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
start_params = AdmmSdp.initParams(inst, opts)

cache0, cache0_time = AdmmSdp.precompute(inst, start_params, opts)
@printf("cache0 time: %.3f\n", cache0_time)
@printf("\n")

# cache1, cache1_time = AdmmSdp.precompute(inst, start_params, opts)
# @printf("cache1 time: %.3f\n", cache1_time)

# final_params, summary = AdmmSdp.admm(start_params, cache, opts)
# res = AdmmSdp.run(inst, opts)

ds = [ones(size(cache0.Hs[k])[1])' * cache0.Hs[k] for k in 1:start_params.num_cliques]
D = sum(cache0.Hs[k]' * cache0.Hs[k] for k in 1:start_params.num_cliques)
D = Diagonal(diag(D))
DJJt = D + cache0.J * cache0.J'
=#


