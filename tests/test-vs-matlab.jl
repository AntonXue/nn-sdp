#
main_start_time = time()

#
include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/deep-sdp.jl"); using .DeepSdp

include("parsers/nnet-parser.jl"); using .NNetParser
include("parsers/vnnlib-parser.jl"); using .VnnlibParser
include("utils.jl"); using .Utils

using ArgParse

println("Finished importing: " * string(time() - main_start_time))

# Solve each instance
function solve_for_h(ffnet :: FeedForwardNetwork, input :: BoxInput, opts :: DeepSdpOptions, normal :: Vector{Float64})
  @assert length(normal) == ffnet.xdims[1] == ffnet.xdims[end] == 2
  
  hplane = HyperplaneSet(normal=normal)
  reach_inst = ReachabilityInstance(ffnet=ffnet, input=input, reach_set=hplane)
  soln = DeepSdp.run(reach_inst, opts)
  return soln
end


function output_polytope(ffnet :: FeedForwardNetwork, input :: BoxInput, opts :: DeepSdpOptions, num_hplanes :: Int)
  bounds = Vector{Float64}()
  for i in 1:num_hplanes
    θ = (i-1) / (num_hplanes * 2 * π)
    normal = [cos(θ); sin(θ)]
    res = solve_for_h(ffnet, input, opts, normal)
    println("res: " * string(res))
    push!(bounds, res.objective_value)
    println("\n\n")
  end
  return bounds
end

#
argparse_settings = ArgParseSettings()
@add_arg_table argparse_settings begin
    "--nnet"
        help = "the NNet file location"
        arg_type = String
end

# Begin stuff
args = parse_args(ARGS, argparse_settings)

nnet = NNetParser.NNet(args["nnet"])
ffnet = Utils.NNet2FeedForwardNetwork(nnet)

input = BoxInput(x1min=[0.99; 0.99]; x1max=[1.01, 1.01])
x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband=0)


