#
main_start_time = time()

#
include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/split-sdp.jl"); using .SplitSdp
include("parsers/nnet-parser.jl"); using .NNetParser
include("parsers/vnnlib-parser.jl"); using .VnnlibParser
include("utils.jl"); using .Utils
using LinearAlgebra
using JuMP
using Random

#

#=
prop1_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_1.vnnlib"
parsed1 = VnnlibParser.read_vnnlib_simple(prop1_filepath, 5, 5)
ACAS_1_1 = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
nnet = NNetParser.NNet(ACAS_1_1)
=#


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
    opt = SplitSdpOptions(β=1, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=true, tband_func=(x, y) -> 2)
    push!(split_pairs, (inst, opt))
  end
end

println("here! " * string(time() - main_start_time))

inst = split_pairs[1][1]
opts = split_pairs[1][2]
# res = SplitSdp.run(inst, opts)



