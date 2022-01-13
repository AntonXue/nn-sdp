import argparse
import sys
import os
import numpy

# Set up the path to nnenum
NNENUM_PATH = "/home/taro/stuff/nv-repos/nnenum/src"
try:
  sys.path.index(NNENUM_PATH)
except:
  sys.path.append(NNENUM_PATH)

from nnenum import nnenum

#
class NnenumOptions():
  def __init__(
          self, 
          settings_str = "auto",
          timeout = 600,
          verbose = False
  ):
    self.settings_str = settings_str
    self.timeout = timeout
    self.verbose = verbose

# Much of this is adapted from
# https://github.com/stanleybak/nnenum/blob/2ad47c186f8113e75f16d45db9e314b1a0b72827/src/nnenum/nnenum.py
def run_nnenum(onnx_filepath : str,
               vnnlib_filepath : str,
               opts : NnenumOptions):

    # Load the network and generate the specs
    network = nnenum.load_onnx_network(onnx_filepath)
    spec_list, input_dtype = nnenum.make_spec(vnnlib_filepath, onnx_filepath)
    

    # Depending on the settings_str, change the mode accordingly to some heuristics
    if opts.settings_str == "auto":
      num_inputs = len(spec_list[0][0])
      if num_inputs < 700:
        nnenum.set_control_settings()
      else:
        nnenum.set_image_settings()
    elif opts.settings_str == "control":
      nnenum.set_control_settings()
    elif opts.settings_str == "image":
      nnenum.set_image_settings()
    else:
      assert opts.settings_str == "exact"
      nnenum.set_exact_settings()

    # Custom settings
    nnenum.Settings.PRINT_PROGRESS = opts.verbose
    nnenum.Settings.PRINT_OUTPUT = opts.verbose
    nnenum.Settings.CHECK_SINGLE_THREAD_BLAS = False # Fails if we don't do this

    # Assume that the spec list is simple; i.e. has only one element
    assert(len(spec_list) == 1)
    init_box, spec = spec_list[0]
    init_box = numpy.array(init_box, dtype=input_dtype)

    # Run stuff 
    nnenum.Settings.TIMEOUT = opts.timeout
    res = nnenum.enumerate_network(init_box, network, spec)

    return res



