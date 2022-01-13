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

#
argparse_settings = ArgParseSettings()
@add_arg_table argparse_settings begin
    "--nnet"
        help = "the NNet file location"
        arg_type = String
    "--prop"
        help = "the VNNLIB file location"
        arg_type = String
    "--indim"
        arg_type = Int
        default = 0
    "--outdim"
        arg_type = Int
        default = 0
end

args = parse_args(ARGS, argparse_settings)

@assert args["indim"] > 0
@assert args["outdim"] > 0

nnet = NNetParser.NNet(args["nnet"])
parsed_prop = VnnlibParser.read_vnnlib_simple(args["prop"], args["indim"], args["outdim"])

ffnet = Utils.NNet2FeedForwardNetwork(nnet)
disjs = vnnlib2constraints(parsed_prop, ffnet)

inst_opt_pairs = Vector{Any}()

for conjs in disjs
  for c1 in conjs
    input, safety = c1
    inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
    x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
    opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband=0)
    push!(inst_opt_pairs, (inst, opts))
  end
end

println("Done generating instances: " * string(time() - main_start_time))

if length(disjs) == 1 && length(disjs[1]) == 1
  @assert length(inst_opt_pairs) == 1
  inst, opt = inst_opt_pairs[1]
end


