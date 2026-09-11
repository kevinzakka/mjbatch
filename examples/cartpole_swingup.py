# SPDX-License-Identifier: Apache-2.0

"""iLQR swinging up a cart with two poles."""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from mjbatch import Batch
from window import CHOSEN, GHOST, Window, progress

MODEL, POLES = Path(__file__).parent / "assets" / "cartpoles.xml", 2
T, SUB, NX, NU = 100, 4, 2 * (POLES + 1), 1  # knots, substeps per knot, state and control dims
EPS, ALPHAS = 1e-6, 0.5 ** np.arange(9)  # finite-difference step, line search steps
ITERS, TOL = 300, 1e-4  # max iterations; stop when the relative cost drop is below TOL
W_CART, W_SPEED, W_SPIN, W_CTRL, TERMINAL = 0.1, 0.01, 0.01, 0.2, 100.0  # TERMINAL: last knot
FRAME = 1 / 30  # seconds per frame while solving


# The state x is (cart position, pole angles, cart speed, pole rates); angle 0 is upright.


def cost(t, x, u):
  poles = x.shape[-1] // 2 - 1
  cart, angles, rates = x[..., 0], x[..., 1 : poles + 1], x[..., poles + 1 :]
  upright = (1.0 - np.cos(angles)).sum(-1)
  quiet = W_SPEED * rates[..., 0] ** 2 + W_SPIN * (rates[..., 1:] ** 2).sum(-1)
  state = upright + W_CART * cart**2 + quiet
  return (TERMINAL if t == T else 1.0) * state + W_CTRL * (u[..., 0] ** 2)


def expand(xs, us):
  """Compute the cost's gradients and Hessians along a trajectory."""
  w = np.r_[2 * W_CART, np.zeros(POLES), 2 * W_SPEED, np.full(POLES, 2 * W_SPIN)]
  lx, diag, angles = w * xs, np.tile(w, (len(xs), 1)), xs[:, 1 : POLES + 1]
  lx[:, 1 : POLES + 1] = np.sin(angles)
  diag[:, 1 : POLES + 1] = np.cos(angles)  # negative while a pole hangs; mu regularizes it
  lxx = diag[:, :, None] * np.eye(NX)
  lx[-1], lxx[-1] = TERMINAL * lx[-1], TERMINAL * lxx[-1]
  return lx, lxx, 2 * W_CTRL * us, np.tile(2 * W_CTRL * np.eye(NU), (len(us), 1, 1))


class Planner:
  def __init__(self, model, T, sub, cost, expand):
    self.T, self.sub = T, sub
    self.cost, self.expand = cost, expand
    self.nq, self.nv, self.nu = model.nq, model.nv, model.nu
    self.nx = 2 * self.nv
    self.lo, self.hi = model.actuator_ctrlrange.T
    # Slide and hinge only, so the state is a plain vector: states subtract and steps add.
    assert (model.jnt_type > 1).all(), "free and ball joints need a quaternion-aware planner"
    self.batch = Batch(model, T * (1 + self.nx + self.nu))
    self.qpos, self.qvel = self.batch.bind("qpos"), self.batch.bind("qvel")
    self.ctrl, self.warm = self.batch.bind("ctrl"), self.batch.bind("qacc_warmstart")
    # The line search batch. forward=True keeps derived fields current for the cost.
    self.line = Batch(model, len(ALPHAS), forward=True)
    self.line_fields = [self.line.bind(f) for f in ("qpos", "qvel", "ctrl", "qacc_warmstart")]
    self.warning = self.line.bind("warning")  # counters; an unstable sim resets and bumps them

  def step(self, x, u):
    self.qpos[:], self.qvel[:] = x[:, : self.nq], x[:, self.nq :]
    self.ctrl[:], self.warm[:] = u, 0.0
    self.batch.step(nstep=self.sub)
    return np.concatenate([self.qpos, self.qvel], axis=1)

  def advance(self, x, u):
    qpos, qvel, ctrl, warm = self.line_fields
    qpos[:], qvel[:], ctrl[:], warm[:] = x[:, : self.nq], x[:, self.nq :], u, 0.0
    self.line.step(nstep=self.sub)
    return np.concatenate([qpos, qvel], axis=1)

  def linearize(self, xs, us):
    """Compute the Jacobians A_t = df/dx and B_t = df/du at every knot by forward differences."""
    T, nx, nu, nv, cols = self.T, self.nx, self.nu, self.nv, 1 + self.nx + self.nu
    x, u = np.repeat(xs[:-1], cols, axis=0), np.repeat(us, cols, axis=0)
    delta = EPS * np.tile(np.eye(cols)[:, 1:], (T, 1))  # column 0 is the unperturbed row
    delta[:, nx:] *= np.where(u + EPS > self.hi, -1.0, 1.0)  # MuJoCo clamps ctrl: step down at hi
    x[:, : self.nq] += delta[:, :nv]
    x[:, self.nq :] += delta[:, nv:nx]
    out = self.step(x, u + delta[:, nx:]).reshape(T, cols, -1)
    base = np.repeat(out[:, 0], cols - 1, axis=0)
    h = delta.reshape(T, cols, -1)[:, 1:].sum(axis=2).ravel()  # signed step of each row
    jac = (out[:, 1:].reshape(-1, out.shape[2]) - base) / h[:, None]
    jac = np.swapaxes(jac.reshape(T, nx + nu, nx), 1, 2)  # (T, nx, nx + nu)
    return jac[:, :, :nx], jac[:, :, nx:]

  def rollout(self, x0, us, gains=None):
    """Roll out us on every line search sim; with gains (xs, k, K), sim a plays u + a k + K dx."""
    n, T = len(ALPHAS), self.T
    x, warned = np.tile(x0, (n, 1)), self.warning.copy()
    new_xs = np.empty((T + 1, n, self.nq + self.nv))
    new_us, total = np.empty((T, n, self.nu)), np.zeros(n)
    new_xs[0] = x
    for t in range(T):
      u = np.tile(us[t], (n, 1))
      if gains is not None:
        xs, k, K = gains
        u += ALPHAS[:, None] * k[t] + (x - xs[t]) @ K[t].T
      u = np.clip(u, self.lo, self.hi)
      total += self.cost(t, x, u)
      x = self.advance(x, u)
      new_xs[t + 1], new_us[t] = x, u
    total += self.cost(T, x, np.zeros((n, self.nu)))
    total[(self.warning != warned).any((1, 2))] = np.inf  # a sim that reset is not a valid plan
    return new_xs, new_us, total


def boxqp(Q, q, lo, hi, k):
  """Minimize 1/2 k'Qk + q'k subject to lo <= k <= hi. Return k and the free set."""
  k, index, n = np.clip(k, lo, hi), np.empty(len(q), np.int32), len(q)
  free = np.zeros(n, bool)
  free[index[: mujoco.mju_boxQP(k, np.empty((n, n + 7)), index, Q, q, lo, hi)]] = True
  return k, free


def backward(A, B, lx, lxx, lu, luu, lo, hi, mu):
  """Run the backward pass with box-constrained controls (Tassa 2012, 2014)."""
  T, nu, nx = len(lu), lu.shape[1], lx.shape[1]
  vx, vxx = lx[-1], lxx[-1]
  k, K = np.zeros((T + 1, nu)), np.empty((T, nu, nx))
  for t in reversed(range(T)):
    qx, qu = lx[t] + A[t].T @ vx, lu[t] + B[t].T @ vx
    qxx = lxx[t] + A[t].T @ vxx @ A[t]
    quu, qux = luu[t] + B[t].T @ vxx @ B[t], B[t].T @ vxx @ A[t]
    reg = vxx + mu * np.eye(nx)
    quu_reg, qux_reg = luu[t] + B[t].T @ reg @ B[t], B[t].T @ reg @ A[t]
    if not (np.isfinite(quu_reg).all() and np.linalg.eigvalsh(quu_reg).min() > 0.0):
      return None
    k[t], free = boxqp(quu_reg, qu, lo[t], hi[t], k[t + 1])  # warm start from the next knot
    K[t] = 0.0  # no feedback on clamped controls
    K[t, free] = -np.linalg.solve(quu_reg[np.ix_(free, free)], qux_reg[free])
    vx = qx + K[t].T @ quu @ k[t] + K[t].T @ qu + qux.T @ k[t]
    vxx = qxx + K[t].T @ quu @ K[t] + K[t].T @ qux + qux.T @ K[t]
    vxx = 0.5 * (vxx + vxx.T)
  return k[:-1], K


def ilqr(planner, x0, us, watch=lambda xs: None):
  T, nu, nx = planner.T, planner.nu, planner.nx
  xs, _, total = planner.rollout(x0, us)  # without gains, every sim replays us
  xs, total = xs[:, 0], total[0]
  watch(xs)

  def derivatives():  # of the dynamics and the cost along xs, us, plus the control bounds
    return *planner.linearize(xs, us), *planner.expand(xs, us), planner.lo - us, planner.hi - us

  mu, K, d = 1.0, np.zeros((T, nu, nx)), derivatives()
  start, alpha, stop = time.perf_counter(), 0.0, False
  for it in range(ITERS):
    accepted = False
    if (sweep := backward(*d, mu=mu)) is not None:
      k, K = sweep
      new_xs, new_us, totals = planner.rollout(x0, us, (xs, k, K))
      best = int(np.argmin(totals))
      if accepted := bool(totals[best] < total):
        xs, us, drop = new_xs[:, best], new_us[:, best], total - totals[best]
        total, mu, alpha = totals[best], max(mu / 10.0, 1e-6), ALPHAS[best]
        stop = drop < TOL * total
    if not accepted:
      mu *= 10.0
      stop = mu > 1e6
    watch(xs)
    done, tail = (it + 1) / ITERS, f"cost {total:9.4f}  alpha {alpha:.4f}  mu {mu:.0e}"
    progress(f"iter {it + 1:3d}/{ITERS}", done, time.perf_counter() - start, tail, end=stop or done == 1)
    if stop:
      break
    if accepted:
      d = derivatives()
  return xs, us, K, total  # K is from the sweep that produced xs and us


def wrap(angles):
  return (angles + np.pi) % (2 * np.pi) - np.pi


def tips(xs, height):
  """Return the tip position of every pole along a trajectory, shape (len(xs), poles, 3)."""
  poles = xs.shape[1] // 2 - 1
  cart, angles = xs[:, 0], np.cumsum(xs[:, 1 : poles + 1], axis=1)
  x = cart[:, None] + np.cumsum(np.sin(angles), axis=1)
  z = height + np.cumsum(np.cos(angles), axis=1)
  return np.stack([x, np.zeros_like(x), z], axis=-1)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--headless", action="store_true")
  args = parser.parse_args()
  model = mujoco.MjModel.from_xml_path(str(MODEL))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("hang").id)
  mujoco.mj_forward(model, data)  # so the window can draw the pose while the solve runs
  x0 = np.concatenate([data.qpos, data.qvel])
  # Hanging still, all gradients vanish, so the initial plan is small random noise.
  us = 0.1 * np.random.default_rng(0).standard_normal((T, NU))
  planner = Planner(model, T, SUB, cost, expand)

  if args.headless:
    xs, us, K, _ = ilqr(planner, x0, us)
  else:
    window, traces = Window(model, data, "side", "cart-poles"), []
    height = model.body("cart").pos[2]

    def watch(xs):  # draw every accepted plan; older ones as ghosts
      traces.append((tips(xs, height)[:, -1], CHOSEN, 0.012))
      ghosts = [(points, GHOST, 0.004) for points, _, _ in traces[-16:-1]]
      window.draw(ghosts + traces[-1:])
      window.pace(FRAME)

    xs, us, K, _ = ilqr(planner, x0, us, watch=watch)
    # LQR gains to balance afterwards: the same backward pass over a plan resting upright.
    top = np.zeros((T + 1, NX)), np.zeros((T, NU))
    d = *planner.linearize(*top), *expand(*top), np.full((T, NU), -1.0), np.full((T, NU), 1.0)
    hold = backward(*d, mu=0.0)[1][0]
    t = 0
    while window.open():  # play the plan under its gains, then balance
      x = np.concatenate([data.qpos, data.qvel])
      if t < T:
        u = us[t] + K[t] @ (x - xs[t])
      else:
        x[1 : POLES + 1] = wrap(x[1 : POLES + 1])  # angle errors from upright
        u = hold @ x
      data.ctrl[:] = np.clip(u, planner.lo, planner.hi)
      ahead = [(tips(xs[t + 1 :], height)[:, -1], CHOSEN, 0.012)] if t < T else []
      for _ in range(SUB):  # draw every substep in real time
        mujoco.mj_step(model, data)
        window.draw(ahead)
        window.pace(model.opt.timestep)
      t += 1
  tilt = np.abs(wrap(xs[-1, 1 : POLES + 1]))
  print(f"done: tilt {np.round(tilt, 3)} rad, cart x {xs[-1, 0]:+.3f}, |u| {np.abs(us).max():.2f}")


if __name__ == "__main__":
  main()
