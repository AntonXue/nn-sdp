# Run this file to do the nnenum evaluations
from nnenum_bridge import *

acasxu_pairs = [
    ("/home/taro/stuff/test/nv-tests/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx",
      "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_1.vnnlib"),

    ("/home/taro/stuff/test/nv-tests/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx",
      "/home/taro/stuff/test/nv-tests/benchmarks/acasxu/prop/prop_2.vnnlib"),
  ]

nnenum_opts = NnenumOptions(verbose=True)
res = []

for (onnx_filepath, vnnlib_filepath) in acasxu_pairs:
  r = run_nnenum(onnx_filepath, vnnlib_filepath, nnenum_opts)
  res.append(r)



