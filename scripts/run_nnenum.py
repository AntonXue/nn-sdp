# Run this file to do the nnenum evaluations
from nnenum_bridge import *

DEFAULT_NNENUM_OPTS = NnenumOptions(verbose=True)

def run_one(onnx_filepath, vnnlib_filepath, opts=DEFAULT_NNENUM_OPTS):
  res = run_nnenum(onnx_filepath, vnnlib_filepath, opts)
  return res

#
def parser():
  parser = argparse.ArgumentParser("Run nnenum")
  parser.add_argument("--one", action="store_true")
  parser.add_argument("--onnx", type=str, default=None)
  parser.add_argument("--prop", type=str, default=None)
  return parser


# Parse arguments
args = parser().parse_args()

# Check if we're in run one mode
if args.one and args.onnx and args.prop:
  res = run_one(args.onnx, args.prop)

