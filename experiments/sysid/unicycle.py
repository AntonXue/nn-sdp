import math
import torch
from torch import nn
import matplotlib.pylab as plt

# Fix the seed
torch.manual_seed(1234)

# time step
DT = 0.1

''' Feedforward network
Input: x, y, theta, v, omega
Output: x1, y1, theta1
'''
class Unicycle(nn.Module):
  def __init__(self):
    super(Unicycle, self).__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(5, 100),
      nn.Tanh(),
      nn.Linear(100, 100),
      nn.Tanh(),
      nn.Linear(100, 100),
      nn.Tanh(),
      nn.Linear(100, 100),
      nn.Tanh(),
      nn.Linear(100, 100),
      nn.Tanh(),
      nn.Linear(100, 100),
      nn.Tanh(),
      nn.Linear(100, 100),
      nn.Tanh(),
      nn.Linear(100, 3)
    )

  # This is the function that gets called when you do model(z)
  # Assume z = [x; y; theta], and zu = [x, y, theta, v, omega]
  def forward(self, zu):
    z1 = self.linear_relu_stack(zu)
    return z1

''' Discrete time dynamics for a unicycle
  x1 = x + cos(theta) * v * dt
  y1 = y + sin(theta) * v * dt
  theta1 = theta + omega
'''
def step_unicycle(zu):
  x, y, theta, v, omega = zu[0], zu[1], zu[2], zu[3], zu[4]
  x1 = x + math.cos(theta) * v * DT
  y1 = y + math.sin(theta) * v * DT
  theta1 = theta + omega * DT
  z = torch.tensor([x1, y1, theta1])
  return z

''' Generate some trajectories
Input:
  z0: 3-tensor for [x0, y0, theta0]
  us: H x 3 tensor for the [v, omega] sequence
  model: a model
Output:
  zs_real
  zs_model
'''
def traj_unicycle(z0, us, model):
  assert(z0.size(dim=0) == 3) # State is 3-dim
  assert(us.size(dim=1) == 2) # Control is 2-dim
  # View functions cast a 3-vector into a 1x3 matrix to make pytorch happy
  zs_real = z0.view(1, 3) # Treat this as a 2D tensor since it's a history
  zs_model = z0.view(1, 3)
  for u in us:
    # History for the real system
    zu_real = torch.cat([zs_real[-1], u], dim=0)
    z1_real = step_unicycle(zu_real)
    zs_real = torch.cat([zs_real, z1_real.view(1, 3)], dim=0)

    # History for the learned model
    zu_model = torch.cat([zs_model[-1], u], dim=0)
    z1_model = model(zu_model) # This will call the forward function
    zs_model = torch.cat([zs_model, z1_model.view(1, 3)], dim=0)

  return zs_real, zs_model

# Make some training data
def train(model, num_data=40000, num_iters=15000):
  # A bunch of random points
  xys = 8 * (torch.rand(num_data, 2) - 0.5 * torch.ones(num_data, 2))
  thetas = -2 * math.pi * (torch.rand(num_data, 1) - 0.5 * torch.ones(num_data, 1))
  vs = 4  * (torch.rand(num_data, 1) - 0.5 * torch.ones(num_data, 1))
  omegas = 4  * (torch.rand(num_data, 1) - 0.5 * torch.ones(num_data, 1))
  zus = torch.cat([xys, thetas, vs, omegas], dim=1)
  # zus = 10 * (torch.rand(num_data, 5) - 0.5 * torch.ones(num_data, 5))

  # The true system outputs
  ys = torch.stack([step_unicycle(zu) for zu in zus])
  # Mean-squared-error loss
  loss_fn = torch.nn.MSELoss(reduction="sum")
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
  for t in range(num_iters):
    ys_pred = model(zus)
    loss = loss_fn(ys_pred, ys)
    if t % 100 == 99: print(t, loss.item())
    # Sequence of steps for applying a gradient step in the optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return model

# Do plots
def do_plots(z0, us, model, saveto="/home/antonxue/dump/unicycle.png"):
  zs_real, zs_model = traj_unicycle(z0, us, model)
  xs_real, ys_real = zs_real[:,0], zs_real[:,1]
  xs_model, ys_model = zs_model[:,0].detach(), zs_model[:,1].detach()

  # Real system traj in blue
  plt.plot(xs_real, ys_real, c="b")
  # Model traj in red
  plt.plot(xs_model, ys_model, c="r")
  plt.savefig(saveto)
  return plt


# Initialize stuff to do the circle plots
model = Unicycle()
known_z0 = torch.zeros(3)
H = 100
vs = torch.linspace(1, 2, H).view(H, 1)
known_us = torch.cat([vs, torch.ones(H,1)], dim=1)

# Uncomment these lines below to train and plot
# train(model)
# do_plots(known_z0, known_us, model)

