# Convert all the nnet files in a directory to onnx files in a target directory
import sys
import os
import pathlib
import argparse

# Point Python to the NNet package
NNET_PATH = "/home/taro/stuff/nv-repos"
sys.path.append(NNET_PATH)
import NNet.converters.nnet2onnx as n2o
import NNet.converters.onnx2nnet as o2n

# Parser configuration
def parser():
  parser = argparse.ArgumentParser("Convert between NNet and ONNX files")
  parser.add_argument("--nnet2onnx", action="store_true")
  parser.add_argument("--onnx2nnet", action="store_true")
  parser.add_argument("-idir", type=str)
  parser.add_argument("-odir", type=str)
  return parser

# Parse input arguments
args = parser().parse_args()

# Exactly one of nnet2onnx or onnx2nnet can be true
assert(args.nnet2onnx is not args.onnx2nnet)
src_ext = "nnet" if args.nnet2onnx else "onnx"
tgt_ext = "onnx" if args.nnet2onnx else "nnet"

src_paths = list(pathlib.Path(args.idir).glob(f"*.{src_ext}"))

for src_path in src_paths:
  src_name = os.path.basename(src_path)
  pre, _ = os.path.splitext(src_name)
  tgt_path = os.path.join(args.odir, pre + f".{tgt_ext}")

  if args.nnet2onnx:
    n2o.nnet2onnx(src_path, onnxFile=tgt_path)
  else:
    o2n.onnx2nnet(src_path, nnetFile=tgt_path)
    
print(f"Total converted files: {len(src_paths)}")

