import torch
import torch.nn as nn
import onnx2pytorch
import onnx
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Adapted from:
# https://github.com/huanzhang12/alpha-beta-CROWN/blob/11ef887edc3394f64caf64d9e5685882f506e2a4/complete_verifier/utils.py#L167
def load_model_onnx(path, input_shape):
  onnx_model = onnx.load(path) # Anton: load using onnx library
  onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
  onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
  input_shape = tuple(input_shape)
  pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

  new_modules = []
  modules = list(pytorch_model.modules())[1:]
  for mi, m in enumerate(modules):
    if isinstance(m, torch.nn.Linear):
      # Anton: onnx2pytorch seems to (incorrectly) swap the in_features and out_features
      in_features, out_features = m.out_features, m.in_features
      new_m = nn.Linear(in_features=in_features, out_features=out_features, bias=m.bias is not None)
      new_m.weight.data.copy_(m.weight.data.T) # Need to transpose due to swaps
      new_m.bias.data.copy_(m.bias)
      new_modules.append(new_m)
    elif isinstance(m, torch.nn.ReLU):
      new_modules.append(torch.nn.ReLU())
    elif isinstance(m, onnx2pytorch.operations.flatten.Flatten):
      new_modules.append(torch.nn.Flatten())
    else:
      raise NotImplementedError

  seq_model = nn.Sequential(*new_modules)
  return seq_model

# Call this function
def find_bounds(onnx_file, x1min, x1max):
  assert len(x1min) == len(x1max)
  # Convert the input to torch tensors
  x1min, x1max = torch.tensor(x1min).float(), torch.tensor(x1max).float()
  x1min, x1max = x1min.view(1, len(x1min)), x1max.view(1, len(x1max))
  xcenter = (x1max + x1min) / 2
  ptb = PerturbationLpNorm(x_L=x1min, x_U=x1max) # By default this is Linfty box
  my_input = BoundedTensor(xcenter, ptb)

  # Construct the model
  pytorch_model = load_model_onnx(onnx_file, (len(x1min),))
  model = BoundedModule(pytorch_model, xcenter)

  # Bounds calculation
  # lb, ub = model.compute_bounds(x=(my_input,), method="IBP")
  # return lb.tolist(), ub.tolist()

  lb, ub = model.compute_bounds(x=(my_input,), method="CROWN")
  return lb[0].tolist(), ub[0].tolist()

# ONNX_FILE = "/home/antonxue/dump/W5-D5.onnx"
ONNX_FILE = "/home/antonxue/dump/rand.onnx"

x1min = torch.tensor([0,0])
x1max = torch.tensor([1,1])
xcenter = (x1max + x1min) / 2
ptb = PerturbationLpNorm(x_L=x1min, x_U=x1max) # By default this is Linfty box
ptb_input = BoundedTensor(xcenter, ptb)

pytorch_model = load_model_onnx(ONNX_FILE, (len(x1min),))
# pytorch_model = load_model_onnx(ONNX_FILE, (len(x1min),), force_convert=False)
seqmods = list(pytorch_model.children())

bounded = BoundedModule(pytorch_model, xcenter)

nl1 = nn.Linear(in_features=3, out_features=5, bias=True)

