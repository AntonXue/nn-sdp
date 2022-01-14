import sys
import os
import pathlib
import argparse

import math
import numpy as np

NNET_PATH = "/home/taro/stuff/nv-repos"
try:
  sys.path.index(NNET_PATH)
except:
  sys.path.append(NNET_PATH)

import NNet.utils.writeNNet as wn

# Important!
np.random.seed(1234)
INPUT_DIM = 2
OUTPUT_DIM = 2
LAYER_DIMS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
NUM_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]

# Generate a random network given the number of layers, input, output, and layer dimensions
def random_params(input_dim, output_dim, layer_dim, num_layers, sigma):
  xdims = [input_dim] + ([layer_dim] * (num_layers - 1)) + [output_dim]
  Ws = [np.random.standard_normal(size=(xdims[k+1], xdims[k])) * sigma for k in range(0, len(xdims)-1)]
  bs = [np.random.standard_normal(size=xdims[k+1]) * sigma for k in range(0, len(xdims)-1)]
  return xdims, Ws, bs

# Go through all the combinations 
def enumerate_random_params():
  for layer_dim in LAYER_DIMS:
    for num_layers in NUM_LAYERS:
      sigma = 2 / math.sqrt(layer_dim + num_layers)
      input_dim = INPUT_DIM
      output_dim = OUTPUT_DIM
      xdims, Ws, bs = random_params(input_dim, output_dim, layer_dim, num_layers, sigma)
      yield input_dim, output_dim, layer_dim, num_layers, xdims, Ws, bs

#
def write_NNet(input_dim, Ws, bs, filepath):
  input_mins = -10000 * np.ones(input_dim)
  input_maxes = 10000 * np.ones(input_dim)
  means = np.zeros(input_dim + 1)
  ranges = np.ones(input_dim + 1)
  wn.writeNNet(Ws, bs, input_mins, input_maxes, means, ranges, filepath)

# Parser setup
def parser():
  parser = argparse.ArgumentParser("Generate NNet files")
  parser.add_argument("--nnetdir", type=str)
  return parser

args = parser().parse_args()

# Check that nnetdir is not None and in fact exists
assert args.nnetdir is not None
assert os.path.isdir(args.nnetdir)

# Generate each
for input_dim, output_dim, layer_dim, num_layers, _, Ws, bs in enumerate_random_params():
  idim = str(input_dim)
  odim = str(output_dim)
  ldim = str(layer_dim)
  numl = str(num_layers)

  filename = f"rand-in{idim}-out{odim}-ldim{ldim}-numl{numl}"
  filename = filename + ".nnet"
  filepath = os.path.join(args.nnetdir, filename)

  print(f"writing: " + filepath)
  write_NNet(input_dim, Ws, bs, filepath)


