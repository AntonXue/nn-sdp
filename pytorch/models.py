import sys
import os
import torch
import torch.nn as nn
import copy

DT = 0.05

# Cartpoly dynamics
def old_cartpole_dynamics(z, u, m_cart=0.25, m_pole=0.1, l=0.4, g=9.81):
  assert z.ndim == 3
  bsize, L, _ = z.shape
  assert u.shape == torch.Size([bsize, L, 1])
  assert z.device == u.device
  dev = z.device

  x, dx, theta, dtheta = z[:,:,0:1], z[:,:,1:2], z[:,:,2:3], z[:,:,3:4]

  # For the simple, pure vector case, we'd like to have
  # A * [ddx; ddtheta] = [-gsin(theta); u - ml dtheta^2 sin(theta)] = c
  # ... but the below is this extended to the batched case
  _A11 = torch.cos(theta)
  _A12 = -1 * l * torch.ones(bsize, L, 1).to(dev)
  _A21 = (m_cart + m_pole) * torch.ones(bsize, L, 1).to(dev)
  _A22 = -1 * m_pole * l * torch.cos(theta)
  _A1 = torch.cat([_A11, _A12], dim=2)
  _A2 = torch.cat([_A21, _A22], dim=2)
  # A.shape == bsize * L * 2 * 2
  A = torch.stack([_A1, _A2], dim=2)

  # Now do the RHS stuff
  _c1 = -1 * g * torch.sin(theta)
  _c2 = u - m_pole * l * (dtheta**2) * torch.sin(theta)
  c = torch.cat([_c1, _c2], dim=2)
  B = torch.inverse(A)
  Bc = torch.matmul(B, c.reshape(bsize, L, 2, 1))

  # Accumulate stuff and return
  ddx = Bc[:,:,0]
  ddtheta = Bc[:,:,1]
  dz = torch.cat([dx, ddx, dtheta, ddtheta], dim=2)
  return dz

def cartpole_dynamics(z, u, m_cart=0.25, m_pole=0.1, l=0.4, g=9.81):
  assert z.ndim == 3
  bsize, L, _ = z.shape
  assert u.shape == torch.Size([bsize, L, 1])
  assert z.device == u.device
  dev = z.device

  x, dx, theta, dtheta = z[:,:,0:1], z[:,:,1:2], z[:,:,2:3], z[:,:,3:4]

  M = m_cart + m_pole

  ddtheta_top_1 = (-u - m_pole * l * dtheta**2 * torch.sin(theta)) / M
  ddtheta_top = g*torch.sin(theta) + torch.cos(theta) * ddtheta_top_1
  ddtheta_bot = l*((4/3) - (m_pole * torch.cos(theta)**2 / M))
  ddtheta = ddtheta_top / ddtheta_bot

  ddx_top = u + m_pole*l * (dtheta**2 * torch.sin(theta) - ddtheta*torch.cos(theta))
  ddx = ddx_top / M

  dz = torch.cat([dx, ddx, dtheta, ddtheta], dim=2)
  return dz

# One step of the dynamics
def dynamics_step(dyn, ctrl, z, dt=DT):
  assert z.ndim == 3
  u = ctrl(z)
  dz = dyn(z, u)
  return z + dz*dt

# Generate a trajectory
def dynamics_traj(dynamics, ctrl, x0, T):
  assert x0.ndim == 3
  assert x0.shape[1] == 1
  xt = x0
  xs = [xt]
  for t in range(T):
    ut = ctrl(xt)
    xt = dynamics_step(dynamics, ctrl, xt)
    xs.append(xt)
  xs = torch.cat(xs, dim=1)
  return xs

def closed_loop_traj(dynamics, x0, T):
  assert x0.ndim == 3
  assert x0.shape[1] == 1
  xt = x0
  xs = [xt]
  for t in range(T):
    xt = dynamics(xt)
    xs.append(xt)
  xs = torch.cat(xs, dim=1)
  return xs

# Neural controller
def make_neural_controller(nc=50):
  return nn.Sequential(
    nn.Linear(4, nc),
    nn.ReLU(),
    nn.Linear(nc,nc),
    nn.ReLU(),
    nn.Linear(nc, 1))

def make_zero_controller():
  lin = nn.Linear(4, 1, bias=True)
  lin.weight.data[:] = 0
  lin.bias.data[:] = 0
  return nn.Sequential(lin)

def make_closed_loop_cartpole(nc=40):
  return nn.Sequential(
    nn.Linear(4, nc),
    nn.ReLU(),
    nn.Linear(nc,nc),
    nn.ReLU(),
    nn.Linear(nc,nc),
    nn.ReLU(),
    nn.Linear(nc,nc),
    nn.ReLU(),
    nn.Linear(nc, 4))

def concat_sequentials(seq1, seq2):
  assert isinstance(seq1, nn.Sequential) and isinstance(seq2, nn.Sequential)
  seq1, seq2 = copy.deepcopy(seq1), copy.deepcopy(seq2)
  xs, ys = list(seq1.children()), list(seq2.children())
  xlast, yfirst = xs[-1], ys[0]
  assert isinstance(xlast, nn.Linear) and isinstance(yfirst, nn.Linear)
  assert xlast.out_features == yfirst.in_features

  A1, b1 = xlast.weight.data, xlast.bias.data
  A2, b2 = yfirst.weight.data, yfirst.bias.data
  Anew, bnew = A2@A1, b2 + A2@b1
  lin = nn.Linear(xlast.in_features, yfirst.out_features)
  lin.weight.data = Anew
  lin.bias.data = bnew
  mods = xs[:-1] + [lin] + ys[1:]
  seq = nn.Sequential(*mods)
  return seq


def unroll_and_save(model, models_dir, T=20, base_name="cartpole"):
  model1 = copy.deepcopy(model)
  torch.save(model1, os.path.join(models_dir, base_name + "1.pth"))

  # Now unroll
  this_model = model1
  for i in range(2,T+1):
    this_model = concat_sequentials(this_model, model1)
    torch.save(this_model, os.path.join(models_dir, base_name + f"{i}.pth"))

