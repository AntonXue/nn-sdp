#

include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/intervals.jl"); using .Intervals
include("core/partitions.jl"); using .Partitions
include("core/deep-sdp.jl"); using .DeepSdp

include("parsers/nnet-parser.jl"); using .NNetParser
include("parsers/vnnlib-parser.jl"); using .VnnlibParser
include("utils.jl"); using .Utils


#=
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

hello = VnnlibParser.read_vnnlib_simple(prophello_filepath, 5, 5)
=#

ACAS_1_1 = "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
nnet = NNetParser.NNet(ACAS_1_1)

