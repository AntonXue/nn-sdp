#

include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/deep-sdp.jl"); using .DeepSdp

include("parsers/nnet-parser.jl"); using .NNetParser
include("parsers/vnnlib-parser.jl"); using .VnnlibParser
include("utils.jl"); using .Utils

prop1_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_1.vnnlib"
prop2_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_2.vnnlib"
prop3_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_3.vnnlib"
prop4_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_4.vnnlib"
prop5_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_5.vnnlib"
prop6_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_6.vnnlib"
prop7_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_7.vnnlib"
prop8_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_8.vnnlib"
prop9_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_9.vnnlib"
prop10_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_10.vnnlib"
prophello_filepath = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/hello.vnnlib"

parsed1 = VnnlibParser.read_vnnlib_simple(prop1_filepath, 5, 5)
parsed2 = VnnlibParser.read_vnnlib_simple(prop2_filepath, 5, 5)
parsed3 = VnnlibParser.read_vnnlib_simple(prop3_filepath, 5, 5)
parsed4 = VnnlibParser.read_vnnlib_simple(prop4_filepath, 5, 5)
parsed5 = VnnlibParser.read_vnnlib_simple(prop5_filepath, 5, 5)
parsed6 = VnnlibParser.read_vnnlib_simple(prop6_filepath, 5, 5)
parsed7 = VnnlibParser.read_vnnlib_simple(prop7_filepath, 5, 5)
parsed8 = VnnlibParser.read_vnnlib_simple(prop8_filepath, 5, 5)
parsed9 = VnnlibParser.read_vnnlib_simple(prop9_filepath, 5, 5)
parsed10 = VnnlibParser.read_vnnlib_simple(prop10_filepath, 5, 5)

ACAS_1_1 = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
nnet = NNetParser.NNet(ACAS_1_1)
ffnet = Utils.NNet2FeedForwardNetwork(nnet)
constrs1 = vnnlib2constraints(parsed1, ffnet)
constrs2 = vnnlib2constraints(parsed2, ffnet)
constrs3 = vnnlib2constraints(parsed3, ffnet)
constrs4 = vnnlib2constraints(parsed4, ffnet)
constrs5 = vnnlib2constraints(parsed5, ffnet)
constrs6 = vnnlib2constraints(parsed6, ffnet)
constrs7 = vnnlib2constraints(parsed7, ffnet)
constrs8 = vnnlib2constraints(parsed8, ffnet)
constrs9 = vnnlib2constraints(parsed9, ffnet)
constrs10 = vnnlib2constraints(parsed10, ffnet)

# Some interval propagation
# x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation()

# Deep Sdp options

deep_pairs = Vector{Any}()

for c1s in constrs1
  for c1 in c1s
    input, safety = c1
    inst = SafetyInstance(ffnet=ffnet, input=input, safety=safety)
    x_intvs, ϕin_intvs, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)
    opt = DeepSdpOptions(x_intervals=x_intvs, slope_intervals=slope_intvs, verbose=true)
    push!(deep_pairs, (inst, opt))
  end
end

println("here!")

inst = deep_pairs[1][1]
opts = deep_pairs[1][2]

res = DeepSdp.run(inst, opts)



