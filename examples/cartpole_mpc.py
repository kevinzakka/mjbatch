# SPDX-License-Identifier: Apache-2.0

"""Predictive-sampling MPC swinging up a cart-pole."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from mjbatch import Batch
from window import CHOSEN, GHOST, Window

MODEL = Path(__file__).parent / "assets" / "cartpole.xml"
K, H, SUB = 1024, 25, 4  # rollouts, horizon knots, substeps per knot
SIGMA, STEPS, NTRACE = 0.3, 150, 32  # control noise, headless control steps, rollouts drawn
W_CART, W_SPEED, W_SPIN, W_CTRL = 0.1, 0.01, 0.01, 0.2  # cost weights


# The state x is (cart position, pole angle, cart speed, pole rate); angle 0 is upright.
# A receding horizon never reaches a final knot, so the cost is the running one at every knot.


def cost(x, u):
  upright = 1.0 - np.cos(x[..., 1])
  quiet = W_SPEED * x[..., 2] ** 2 + W_SPIN * x[..., 3] ** 2
  return upright + W_CART * x[..., 0] ** 2 + quiet + W_CTRL * (u[..., 0] ** 2)


def wrap(angles):
  return (angles + np.pi) % (2 * np.pi) - np.pi


def tips(xs, height):
  """Return the pole's tip position along a trajectory, shape (len(xs), 3)."""
  cart, angle = xs[:, 0], xs[:, 1]
  return np.stack([cart + np.sin(angle), np.zeros_like(cart), height + np.cos(angle)], axis=-1)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--headless", action="store_true")
  args = parser.parse_args()
  model = mujoco.MjModel.from_xml_path(str(MODEL))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("hang").id)
  mujoco.mj_forward(model, data)
  planner = Batch(model, K)  # every sim restarts from the real system's state, one rollout each
  qpos, qvel, ctrl, now = (planner.bind(f) for f in ("qpos", "qvel", "ctrl", "time"))
  rng, plan, scale = np.random.default_rng(0), np.zeros((H, model.nu)), 1.0
  xs = np.empty((H, K, model.nq + model.nv))
  window = None if args.headless else Window(model, data, "side", "cart-pole")
  height = model.body("cart").pos[2]
  for _ in range(STEPS) if window is None else iter(window.open, False):
    qpos[:], qvel[:], now[:] = data.qpos, data.qvel, data.time
    us = np.clip(plan + scale * SIGMA * rng.normal(size=(K, H, model.nu)), -1.0, 1.0)
    us[0] = plan  # the nominal is always a candidate
    total = np.zeros(K)
    for h in range(H):
      ctrl[:] = us[:, h]
      planner.step(nstep=SUB)
      xs[h] = np.concatenate([qpos, qvel], axis=1)
      total += cost(xs[h], us[:, h])
    best = int(np.argmin(total))
    scale = np.clip(total[best] / H / 0.2, 0.1, 1.0)  # the noise shrinks with the cost, to a tenth
    plan = np.roll(us[best], -1, axis=0)  # shift, holding the last knot
    plan[-1] = plan[-2]
    data.ctrl[:] = us[best, 0]
    for _ in range(SUB):
      mujoco.mj_step(model, data)
      if window is not None:
        ghosts = [(tips(xs[:, k], height), GHOST, 0.004) for k in range(0, K, K // NTRACE)]
        window.draw(ghosts + [(tips(xs[:, best], height), CHOSEN, 0.012)])
        window.pace(model.opt.timestep)
  tilt = np.abs(wrap(data.qpos[1:]))
  print(f"done: tilt {np.round(tilt, 3)} rad, cart x {data.qpos[0]:+.3f}")


if __name__ == "__main__":
  main()
