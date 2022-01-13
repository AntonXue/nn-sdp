#
main_start_time = time()

#
include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/admm-sdp.jl"); using .AdmmSdp
include("parsers/nnet-parser.jl"); using .NNetParser
include("parsers/vnnlib-parser.jl"); using .VnnlibParser
include("utils.jl"); using .Utils
using LinearAlgebra
using JuMP
using Random

# Seed is fixed, but all rand calls should also happen in the same expected sequence


test_prop_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/random/prop/test_prop.vnnlib"
parsed_prop = VnnlibParser.read_vnnlib_simple(test_prop_filepath, 2, 2)
nnet = NNetParser.NNet("/home/taro/stuff/test/nv-tests/benchmarks/random/nnet/rand-in2-d10-out2-K10__5-5.nnet")

ffnet = Utils.NNet2FeedForwardNetwork(nnet)
constrs1 = vnnlib2constraints(parsed_prop, ffnet)

split_pairs = Vector{Any}()

for c1s in constrs1
  for c1 in c1s
    input, safety = c1
    inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
    x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
    opt = AdmmSdpOptions(β=1, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband_func=(x, y) -> 2)
    push!(split_pairs, (inst, opt))
  end
end

println("here! " * string(time() - main_start_time))

safety_inst = split_pairs[1][1]
admm_opts = split_pairs[1][2]

admm_params = initParams(safety_inst, admm_opts)
admm_cache, _ = precomputeCache(admm_params, safety_inst, admm_opts)
iter_params, _ = AdmmSdp.admm(admm_params, admm_cache, admm_opts)


