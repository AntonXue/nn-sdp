import torch
import torch.nn as nn
import numpy as np
# pip install git+https://github.com/AntonXue/onnx2pytorch.git
import onnx2pytorch
import onnx
import auto_LiRPA
from auto_LiRPA.operators import *
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import re

# Adapted from:
# https://github.com/huanzhang12/alpha-beta-CROWN/blob/11ef887edc3394f64caf64d9e5685882f506e2a4/complete_verifier/utils.py#L167
def load_model_onnx(path, input_shape):
  onnx_model = onnx.load(path) # Load using onnx library
  onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
  onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
  input_shape = tuple(input_shape)
  pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

  new_modules = []
  modules = list(pytorch_model.modules())[1:]
  for m in modules:
    if isinstance(m, torch.nn.Linear):
      new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
      new_m.weight.data.copy_(m.weight.data)
      new_m.bias.data.copy_(m.bias)
      new_modules.append(new_m)
    elif isinstance(m, torch.nn.ReLU):
      new_modules.append(torch.nn.ReLU())
    elif isinstance(m, torch.nn.Tanh):
      new_modules.append(torch.nn.Tanh())
    elif isinstance(m, onnx2pytorch.operations.flatten.Flatten):
      new_modules.append(torch.nn.Flatten())
    elif (isinstance(m, onnx2pytorch.operations.base.OperatorWrapper) and
          m.forward.__name__ == "tanh"):
      new_modules.append(torch.nn.Tanh())
    else:
      raise NotImplementedError

  seq_model = nn.Sequential(*new_modules)
  return seq_model

# Call this function
def find_bounds(onnx_file, x1min, x1max, method="CROWN"):
  assert len(x1min) == len(x1max)
  # Convert the input to torch tensors
  x1min, x1max = torch.tensor(x1min).float(), torch.tensor(x1max).float()
  x1min, x1max = x1min.view(1, len(x1min)), x1max.view(1, len(x1max))
  xcenter = (x1max + x1min) / 2
  ptb = PerturbationLpNorm(x_L=x1min, x_U=x1max) # By default this is Linfty box
  my_input = BoundedTensor(xcenter, ptb)

  # Construct the model
  pytorch_model = load_model_onnx(onnx_file, x1min.size())
  model = BoundedModule(pytorch_model, xcenter)

  # Use this for its side effects of populating the lower and upper fields
  _, _ = model.compute_bounds(x=(my_input,), method=method)

  # Extract all the linear module while tracking what the next layer is (if present)
  modules = list(model.children())
  # Sort int-ified names, where we append "0" so it's always valid int
  modules.sort(key=lambda m : int(re.sub('[^0-9]','', m.name + "0")))
  linears = []
  linear_succs = {}
  for i, mod in enumerate(modules):
    if isinstance(mod, BoundLinear):
      linears.append(mod)
      next_mod = modules[i+1] if (i < len(modules) - 1) else None
      linear_succs[mod.name] = next_mod

  # Create x_intvs based on each linear layer, starting with the inputs
  x_intvs = [(x1min[0].tolist(), x1max[0].tolist())]
  for i, lin in enumerate(linears):
    acx_lb, acx_ub = lin.lower[0].detach().numpy(), lin.upper[0].detach().numpy()
    # Last element
    if i == len(linears) - 1:
      x_intvs.append((acx_lb.tolist(), acx_ub.tolist()))

    # Not last element: depending on the next node do different things
    else:
      next_mod = linear_succs[lin.name]
      if isinstance(next_mod, BoundRelu):
        print("RELU STUFF")
        acy1, acy2 = np.maximum(0, acx_lb), np.maximum(0, acx_ub)
      elif isinstance(next_mod, BoundTanh):
        acy1, acy2 = np.tanh(acx_lb), np.tanh(acx_ub)
      else:
        acy1, acy2 = acx_lb, acx_ub

      acy_lb, acy_ub = np.minimum(acy1, acy2), np.maximum(acy1, acy2)
      x_intvs.append((acy_lb.tolist(), acy_ub.tolist()))


  print("x_intvs is:")
  for lb, ub in x_intvs:
    print(f"lb: {lb}")
    print(f"ub: {ub}")

  return x_intvs, model

'''
# ONNX_FILE = "/home/antonxue/stuff/nn-sdp/bench/acas/ACASXU_run2a_1_1_batch_2000.onnx"
# ONNX_FILE = "/home/antonxue/dump/W5-D5.onnx"
ONNX_FILE = "/home/antonxue/dump/rand.onnx"

x_intvs, bounded = find_bounds(ONNX_FILE, torch.zeros(2), torch.ones(2))
mods = list(bounded.modules())[1:]

lins = list(filter(lambda x : isinstance(x, BoundLinear), mods))

m0 = mods[0]
m1 = mods[1]
m2 = mods[2]
m3 = mods[3]
'''

