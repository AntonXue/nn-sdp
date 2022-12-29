import os
import sys

from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm

from models import *

torch.manual_seed(1234)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")

cpu = torch.device("cpu")
cuda = torch.device("cuda")

def make_x0(bsize):
  x = torch.clamp(torch.randn(bsize, 1, 1), 2.0, 2.2)
  dx = torch.clamp(torch.randn(bsize, 1, 1), 1.0, 1.2)
  theta = torch.clamp(torch.randn(bsize, 1, 1), -0.174, -0.104)
  dtheta = torch.clamp(torch.randn(bsize, 1, 1), -1.0, -0.8)
  z = torch.cat([x, dx, theta, dtheta], dim=2)
  return z

def train_controller(ctrl_model, dynamics, epochs=32, nbatches=100, bsize=64, L=40, saveto=None):
  if saveto is None:
    saveto = os.path.join(MODELS_DIR, "ctrl.pth")

  ctrl_model.train()
  ctrl_model.cuda()
  lr = 1e-4
  optimizer = torch.optim.Adam(ctrl_model.parameters(), lr=lr)
  print(f"Controller training start!")
  print(f"Epochs: {epochs} | num batches: {nbatches} | batch size: {bsize} | L: {L} | lr0: {lr}")
  loss_fn = nn.MSELoss()
  for ep in range(0, epochs):
    loss_sum = 0
    pbar = tqdm(range(0, nbatches))
    pbar.set_description(f"Epoch {ep+1}/{epochs}")
    for i in pbar:
      x0 = make_x0(bsize)
      x0 = x0.cuda()
      xs_pred = dynamics_traj(dynamics, ctrl_model, x0, L)
      xs_pred = xs_pred.cpu()
      thetas_pred = xs_pred[:,:,2]
      loss = loss_fn(thetas_pred, torch.zeros_like(thetas_pred))
      loss_sum += loss
      loss.backward()
      optimizer.step()
      avg_loss = loss_sum.item() / (i + 1.0)
      pbar.set_postfix(loss=str(round(avg_loss, 3)))

    if ((ep + 1) % 2) == 0:
      torch.save(ctrl_model, saveto)
      print(f"saved model to {saveto} | now is {datetime.now()}")

  torch.save(ctrl_model, saveto)

  return ctrl_model

def train_closed_loop_model(cart_model, dynamics, ctrl_model, epochs=100, nbatches=100, bsize=128, L=60, saveto=None):
  if saveto is None:
    saveto = os.path.join(MODELS_DIR, "cart.pth")

  cart_model.cuda()
  cart_model.train()
  ctrl_model.cuda()
  ctrl_model.eval()

  lr = 1e-4
  optimizer = torch.optim.Adam(cart_model.parameters(), lr=lr)
  print(f"Closed-loop model training start!")
  print(f"Epochs: {epochs} | num batches: {nbatches} | batch size: {bsize} | L: {L} | lr0: {lr}")
  loss_fn = nn.MSELoss()
  for ep in range(0, epochs):
    loss_sum = 0
    pbar = tqdm(range(0, nbatches))
    pbar.set_description(f"Epoch {ep+1}/{epochs}")
    for i in pbar:
      optimizer.zero_grad()
      x0 = make_x0(bsize)
      x0 = x0.cuda()
      xs_pred = closed_loop_traj(cart_model, x0, L)
      xs_true = dynamics_traj(dynamics, ctrl_model, x0, L)
      loss = loss_fn(xs_pred.cpu(), xs_true.cpu())
      loss_sum += loss
      loss.backward()
      optimizer.step()

      avg_loss = loss_sum.item() / (i + 1.0)
      pbar.set_postfix(loss=str(round(avg_loss, 3)))

    if ((ep + 1) % 2) == 0:
      torch.save(cart_model, saveto)
      print(f"saved model to {saveto} | now is {datetime.now()}")

  torch.save(cart_model, saveto)

  return cart_model
   
zero_ctrl = make_zero_controller()
cart40 = make_closed_loop_cartpole(40)
cart10 = make_closed_loop_cartpole(10)

try:
  trained_cart40 = torch.load(os.path.join(MODELS_DIR, "cartw40_step1.pth"))
  trained_cart10 = torch.load(os.path.join(MODELS_DIR, "cartw10_step1.pth"))
  unroll_and_save(trained_cart40, MODELS_DIR, T=30, base_name="cartw40")
  unroll_and_save(trained_cart10, MODELS_DIR, T=30, base_name="cartw10")
except:
  pass



