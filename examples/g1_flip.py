# SPDX-License-Identifier: Apache-2.0

"""Receding-horizon iLQR that tracks a mocap backflip on the Unitree G1."""

import argparse
import copy
import time
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np

from mjbatch import Batch
from window import CAPSULE, CHOSEN, GHOST, Window, progress

ASSETS = Path(__file__).parent / "assets"
G1 = ASSETS / "g1.xml"
CLIP = ASSETS / "flip.npz"  # a mocap flip retargeted to the G1
PLAN = ASSETS / "g1_flip_plan.npz"  # the solve saves here; --play loads it
SUB = 2  # substeps per knot: control at 50 Hz, the clip's frame rate
# Frames of the clip to track (the ankles roll after 200). LIFT raises the feet out of the floor.
START, END, LIFT = 0, 200, 0.001
# Cost weights per m^2, rad^2, (m/s)^2, (rad/s)^2 of body error and per (N m)^2 of torque.
W_POS, W_ROT, W_VEL, W_ANG, W_CTRL = 100.0, 100.0, 5.0, 1.0, 1e-2
W_ROOT = np.array([1.0, 1.0, 1000.0])  # pelvis position weights: loose in x and y, tight in z
HUBER = 0.05  # weighted position errors past sqrt(W_POS) * HUBER cost linearly, not quadratically
HORIZON, STEP, ITERS = 50, 10, 20  # knots planned per window, knots committed, iLQR iterations
SHADE = (0.62, 0.3, 0.2, 0.35)  # color of the reference robot
HEAD = 0.43  # height of the drawn head point above the torso frame, m
BODIES = (  # bodies the cost tracks
  "pelvis", "torso_link",
  "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "left_shoulder_roll_link", "left_elbow_link",
  "right_shoulder_roll_link", "right_elbow_link",
)  # fmt: skip
TORSO = BODIES.index("torso_link")
FEET = [BODIES.index(f"{s}_ankle_roll_link") for s in ("left", "right")]


# The iLQR solver. The batch holds every (knot, perturbed coordinate) pair, so one step()
# linearizes the whole trajectory; the line search runs its step sizes as sims too.

EPS, ALPHAS = 1e-6, 0.5 ** np.arange(9)  # finite-difference step, line search steps
TOL = 1e-4  # stop when the relative cost drop falls below this


# Quaternion helpers on (..., 4) arrays in MuJoCo's (w, x, y, z) order.


def quat_mul(a, b):
  w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdims=True)
  v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
  return np.concatenate([w, v], axis=-1)


def quat_exp(v):
  angle = np.linalg.norm(v, axis=-1, keepdims=True)
  axis = np.divide(v, angle, out=np.zeros_like(v), where=angle > 0)
  return np.concatenate([np.cos(angle / 2), np.sin(angle / 2) * axis], axis=-1)


def quat_log(q):
  norm = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
  axis = np.divide(q[..., 1:], norm, out=np.zeros_like(q[..., 1:]), where=norm > 0)
  angle = 2 * np.arctan2(norm, q[..., :1])
  return np.where(angle > np.pi, angle - 2 * np.pi, angle) * axis  # wrap to the shorter rotation


class Planner:
  """iLQR over a MuJoCo model. Subclasses supply cost(t, x, u) and expand(xs, us)."""

  def __init__(self, model, T, sub):
    self.T, self.sub = T, sub
    self.nq, self.nv, self.nu = model.nq, model.nv, model.nu
    self.nx = 2 * self.nv
    self.lo, self.hi = model.actuator_ctrlrange.T
    jt = model.jnt_type
    assert not (jt == 1).any(), "ball joints are not supported"
    self.free = list(zip(model.jnt_qposadr[jt == 0], model.jnt_dofadr[jt == 0], strict=True))
    quat = [a + i for a, _ in self.free for i in range(3, 7)]
    spin = [d + i for _, d in self.free for i in range(3, 6)]
    self.lin_q, self.lin_v = np.setdiff1d(range(self.nq), quat), np.setdiff1d(range(self.nv), spin)
    self.batch = Batch(model, T * (1 + self.nx + self.nu))
    self.qpos, self.qvel = self.batch.bind("qpos"), self.batch.bind("qvel")
    self.ctrl, self.warm = self.batch.bind("ctrl"), self.batch.bind("qacc_warmstart")
    # The line search batch. forward=True keeps derived fields current for the cost.
    self.line = Batch(model, len(ALPHAS), forward=True)
    self.line_fields = [self.line.bind(f) for f in ("qpos", "qvel", "ctrl", "qacc_warmstart")]
    self.warning = self.line.bind("warning")  # counters; an unstable sim resets and bumps them

  def integrate(self, qpos, dq):
    out = qpos.copy()
    out[:, self.lin_q] += dq[:, self.lin_v]
    for a, d in self.free:
      quat = quat_mul(qpos[:, a + 3 : a + 7], quat_exp(dq[:, d + 3 : d + 6]))
      out[:, a + 3 : a + 7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return out

  def difference(self, x, y):
    dq = np.empty((len(x), self.nv))
    dq[:, self.lin_v] = y[:, self.lin_q] - x[:, self.lin_q]
    for a, d in self.free:
      conj = x[:, a + 3 : a + 7] * [1, -1, -1, -1]
      dq[:, d + 3 : d + 6] = quat_log(quat_mul(conj, y[:, a + 3 : a + 7]))
    return np.concatenate([dq, y[:, self.nq :] - x[:, self.nq :]], axis=1)

  def step(self, x, u):
    n = len(x)
    self.qpos[:n], self.qvel[:n] = x[:, : self.nq], x[:, self.nq :]
    self.ctrl[:n], self.warm[:n] = u, 0.0
    ids = np.arange(n) if n < self.batch.num_sims else None  # a shorter horizon uses n sims
    self.batch.step(ids, nstep=1)
    self.probe(n)  # the sensors still read the input states here
    if self.sub > 1:
      self.batch.step(ids, nstep=self.sub - 1)
    return np.concatenate([self.qpos[:n], self.qvel[:n]], axis=1)

  def probe(self, n):
    pass

  def cost(self, t, x, u):
    raise NotImplementedError

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
    x[:, : self.nq] = self.integrate(x[:, : self.nq], delta[:, :nv])
    x[:, self.nq :] += delta[:, nv:nx]
    out = self.step(x, u + delta[:, nx:]).reshape(T, cols, -1)
    base = np.repeat(out[:, 0], cols - 1, axis=0)
    h = delta.reshape(T, cols, -1)[:, 1:].sum(axis=2).ravel()  # signed step of each row
    jac = self.difference(base, out[:, 1:].reshape(-1, out.shape[2])) / h[:, None]
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
        u += ALPHAS[:, None] * k[t] + self.difference(np.tile(xs[t], (n, 1)), x) @ K[t].T
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


def ilqr(planner, x0, us, xs, watch):
  T, nu, nx = planner.T, planner.nu, planner.nx
  total = np.inf  # accept the first step
  watch(xs)

  def derivatives():  # of the dynamics and the cost along xs, us, plus the control bounds
    return *planner.linearize(xs, us), *planner.expand(xs, us), planner.lo - us, planner.hi - us

  mu, K, d = 1.0, np.zeros((T, nu, nx)), derivatives()
  stop = False
  for _ in range(ITERS):
    accepted = False
    if (sweep := backward(*d, mu=mu)) is not None:
      k, K = sweep
      new_xs, new_us, totals = planner.rollout(x0, us, (xs, k, K))
      best = int(np.argmin(totals))
      if accepted := bool(totals[best] < total):
        xs, us, drop = new_xs[:, best], new_us[:, best], total - totals[best]
        total, mu = totals[best], max(mu / 10.0, 1e-6)
        stop = drop < TOL * total
    if not accepted:
      mu *= 10.0
      stop = mu > 1e6
    watch(xs)
    if stop:
      break
    if accepted:
      d = derivatives()
  return xs, us, K, total  # K is from the sweep that produced xs and us


def build_model():
  spec = mujoco.MjSpec.from_file(str(G1), assets=mm.get("unitree_g1").assets())
  obj = mujoco.mjtObj.mjOBJ_XBODY  # body frame, not the inertial frame
  for body in BODIES:  # 13 sensor values per body: pos, quat, linvel, angvel
    for name in ("pos", "quat", "linvel", "angvel"):
      kind = getattr(mujoco.mjtSensor, f"mjSENS_FRAME{name.upper()}")
      spec.add_sensor(name=f"{body}_{name}", type=kind, objtype=obj, objname=body)
  return spec.compile()


def rotate(q, v):
  v = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
  return quat_mul(quat_mul(q, v), q * [1, -1, -1, -1])[..., 1:]


class Clip:
  """Load a motion clip as MuJoCo states and as the tracked bodies' sensor readings."""

  def __init__(self, path, model):
    d = {k: v.astype(float) for k, v in np.load(path).items()}
    self.dt = 1.0 / d["fps"][0]
    pos, quat = d["body_pos_w"][:, 0] + [0, 0, LIFT], d["body_quat_w"][:, 0]  # pelvis
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    keep = model.jnt_bodyid[1:] - 2  # clip columns of the kept joints; every link has one joint
    spin = rotate(quat * [1, -1, -1, -1], d["body_ang_vel_w"][:, 0])  # world frame to body frame
    self.qpos = np.concatenate([pos, quat, d["joint_pos"][:, keep]], axis=1)
    self.qvel = np.concatenate([d["body_lin_vel_w"][:, 0], spin, d["joint_vel"][:, keep]], axis=1)
    ids = [model.body(name).id - 1 for name in BODIES]  # the clip's body index skips the world body
    pos, quat = d["body_pos_w"][:, ids] + [0, 0, LIFT], d["body_quat_w"][:, ids]
    quat = quat / np.linalg.norm(quat, axis=2, keepdims=True)
    lin, ang = d["body_lin_vel_w"][:, ids], d["body_ang_vel_w"][:, ids]
    self.features = np.concatenate([pos, quat, lin, ang], axis=2)  # same layout as the sensors


def warm_start(model, clip):
  """Solve for joint torques that reproduce the clip using inverse dynamics, with the contact
  forces from a nonnegative least squares instead of from MuJoCo. MuJoCo's inverse dynamics
  computes each contact force from the penetration depth, about 180 N per contact point per
  millimeter here, so the clip's few millimeters of foot penetration would demand tens of
  kilonewtons and joint torques ten times the actuator limits. The least squares instead picks
  the forces that explain the six unactuated root rows, which sum to the robot's weight."""
  data, limit = mujoco.MjData(model), model.actuator_ctrlrange[:, 1]
  qacc = np.gradient(clip.qvel, clip.dt, axis=0)
  torque = np.empty((len(clip.qpos), model.nu))
  for t in range(len(clip.qpos)):
    data.qpos[:], data.qvel[:], data.qacc[:] = clip.qpos[t], clip.qvel[t], qacc[t]
    mujoco.mj_inverse(model, data)
    rows = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
    Jc = data.efc_J.reshape(-1, model.nv)[: data.nefc][rows]
    b = data.qfrc_inverse + data.qfrc_constraint  # the generalized force without the contacts
    torque[t] = b[6:]
    if len(Jc):  # min |A f - r| over f >= 0: forces that explain the root's rows, small torques
      A = np.concatenate([Jc[:, :6].T, Jc[:, 6:].T / limit[:, None], 1e-3 * np.eye(len(Jc))])
      r = np.concatenate([b[:6], b[6:] / limit, np.zeros(len(Jc))])
      lo = np.zeros(len(Jc))
      f, _ = boxqp(A.T @ A, -A.T @ r, lo, np.full_like(lo, np.inf), lo)
      torque[t] -= Jc[:, 6:].T @ f
  return np.clip(torque, -limit, limit)


class Tracker(Planner):
  """Planner whose cost is the tracking error of the clip's bodies, read from frame sensors.
  Compute the cost's gradient and Hessian from the Jacobian of the error vector, measured by
  finite differences on the batch (Gauss-Newton)."""

  def __init__(self, model, clip):
    super().__init__(model, HORIZON, SUB)
    self.clip, self.start = clip, START
    self.sensor, self.line_sensor = self.batch.bind("sensordata"), self.line.bind("sensordata")

  def window(self, start, length):
    self.start, self.T = START + start, length

  def features(self, x):
    """Read the sensors at each state through a forward on the batch: (n, bodies, 13)."""
    n = len(x)
    self.qpos[:n], self.qvel[:n] = x[:, : self.nq], x[:, self.nq :]
    self.batch.forward(np.arange(n))
    return self.sensor[:n].reshape(n, len(BODIES), 13)

  def probe(self, n):  # called by Planner.step while the sensors read the perturbed input states
    cols = 1 + self.nx + self.nu
    self.feat = self.sensor[:n].reshape(-1, cols, len(BODIES), 13)[:, : 1 + self.nx].copy()

  def residual(self, t, feat):
    """Compute weighted errors against clip knot t and the pseudo-Huber slope of each position."""
    ref = self.clip.features[np.minimum(self.start + t, END)]  # hold the last frame past the end
    pos, quat, vel = feat[..., :3], feat[..., 3:7], feat[..., 7:]
    rot = quat_log(quat_mul(quat, ref[..., 3:7] * [1, -1, -1, -1]))
    rel = (pos - pos[..., :1, :]) - (ref[..., :3] - ref[..., :1, :3])
    root = pos[..., :1, :] - ref[..., :1, :3]
    lin, ang = vel[..., :3] - ref[..., 7:10], vel[..., 3:] - ref[..., 10:]
    weighted = (rel, W_POS), (root, W_ROOT), (rot, W_ROT), (lin, W_VEL), (ang, W_ANG)
    errors = [e * np.sqrt(w) for e, w in weighted]
    robust = np.concatenate(errors[:2], axis=-2)  # the position errors get the robust loss
    slope = 1 / np.sqrt(1 + (robust**2).sum(-1) / (W_POS * HUBER**2))
    return np.concatenate([e.reshape(*e.shape[:-2], -1) for e in errors], axis=-1), slope

  def cost(self, t, x, u):
    feat = self.features(x) if t == 0 else self.line_sensor.reshape(len(x), len(BODIES), 13)
    r, slope = self.residual(t, feat)  # after t = 0, the line batch has just advanced to x
    n = 3 * slope.shape[-1]  # number of robust entries
    robust = 2 * W_POS * HUBER**2 * (1 / slope - 1)  # pseudo-Huber: s^2 when small, then linear
    return robust.sum(-1) + (r[..., n:] ** 2).sum(-1) + W_CTRL * (u**2).sum(-1)

  def expand(self, xs, us):
    """Compute the cost's gradient and Hessian by Gauss-Newton, with the error Jacobian from
    forward differences. Reweight the robust rows by their loss slope (IRLS). The perturbed
    sensor readings of the first T knots come from probe(); the last knot's need a forward."""
    T, nx, nv, nq = self.T, self.nx, self.nv, self.nq
    last = np.repeat(xs[-1:], 1 + nx, axis=0)
    delta = EPS * np.eye(1 + nx)[:, 1:]
    last[:, :nq] = self.integrate(last[:, :nq], delta[:, :nv])
    last[:, nq:] += delta[:, nv:]
    feat = np.concatenate([self.feat, self.features(last)[None]])  # (T + 1, 1 + nx, bodies, 13)
    r, slope = self.residual(np.arange(T + 1)[:, None], feat)
    J = (r[:, 1:] - r[:, :1]) / EPS  # (T + 1, nx, residuals)
    w = np.ones(r.shape[::2])
    w[:, : 3 * slope.shape[-1]] = np.repeat(slope[:, 0], 3, axis=-1)
    lx = 2 * np.einsum("tkr,tr->tk", J, w * r[:, 0])
    lxx = 2 * np.einsum("tkr,tr,tlr->tkl", J, w, J, optimize=True)
    return lx, lxx, 2 * W_CTRL * us, 2 * W_CTRL * np.tile(np.eye(self.nu), (T, 1, 1))


def solve(tracker, x0, feedforward, watch=lambda xs, commit=0: None):
  """Plan HORIZON knots at a time and commit STEP of them. Grow the first window from STEP to
  HORIZON, because a full-length plan from a cold start falls over and iLQR cannot recover
  from a fall."""
  T, nu = END - START, tracker.nu
  stages = [(0, end) for end in range(STEP, HORIZON, STEP)]
  stages += [(at, HORIZON) for at in range(0, T, STEP)]  # past the end, track the last frame
  held = np.concatenate([feedforward, np.tile(feedforward[-1:], (HORIZON, 1))])
  xs_all, us_all, K_all, us, committed = [x0], [], [], np.zeros((0, nu)), False
  start = time.perf_counter()
  for i, (at, length) in enumerate(stages):
    tracker.window(at, length)
    shift = STEP if committed else 0  # drop the committed knots from the previous plan
    us = np.concatenate([us[shift:], held[at + len(us) - shift : at + length]])
    xs = tracker.rollout(x0, us)[0][:, 0]
    xs, us, K, cost = ilqr(tracker, x0, us, xs, watch)
    committed = length == HORIZON or at > 0
    n = min(STEP, length) if committed else 0
    watch(xs, commit=n)
    xs_all, us_all, K_all = xs_all + list(xs[1 : n + 1]), us_all + list(us[:n]), K_all + list(K[:n])
    x0 = xs[n]
    done, tail = (i + 1) / len(stages), f"cost {cost:8.1f}  height {x0[2]:.2f}"
    progress(f"knot {len(us_all):3d}/{T}", done, time.perf_counter() - start, tail, end=done == 1)
  return np.array(xs_all), np.array(us_all), np.array(K_all)


def ghost(model):
  """Copy the model for drawing the reference: translucent visual geoms, hidden colliders."""
  shade = copy.deepcopy(model)
  visual = (shade.geom_contype == 0) & (shade.geom_conaffinity == 0)
  shade.geom_rgba[visual], shade.geom_rgba[~visual, 3] = SHADE, 0.0
  return shade, mujoco.MjData(shade)


def paths(tracker, xs):
  """Return the head's and the feet's positions along a trajectory."""
  feat = tracker.features(xs)
  torso, feet = feat[:, TORSO], feat[:, FEET, :3]
  head = torso[:, :3] + rotate(torso[:, 3:7], np.array([0.0, 0.0, HEAD]))
  return head, feet[:, 0], feet[:, 1]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--clip", default=CLIP, type=Path)
  parser.add_argument("--headless", action="store_true", help="solve, save, no window")
  parser.add_argument("--play", action="store_true", help="the saved plan, no solve")
  args = parser.parse_args()
  mujoco.set_mju_user_warning(lambda _: None)  # unstable candidates are discarded silently
  model = build_model()
  clip, data = Clip(args.clip, model), mujoco.MjData(model)
  tracker = Tracker(model, clip)
  x0 = np.concatenate([clip.qpos[START], clip.qvel[START]])
  if args.headless:
    xs, us, K = solve(tracker, x0, warm_start(model, clip)[START:END])
    np.savez(PLAN, xs=xs, us=us, K=K)
    return
  window, dt = Window(model, data, "side", "G1"), model.opt.timestep
  shade = ghost(model)
  plans, knot = [], 0

  def watch(xs, commit=0):
    """Draw the robot, the reference and the plan's paths. When a window is planned, play its
    committed knots in real time."""
    nonlocal knot
    now = paths(tracker, xs)
    plans[:] = [] if commit else plans + [now]
    for t in range(commit + 1):
      data.qpos[:] = xs[t, : model.nq]
      mujoco.mj_forward(model, data)  # kinematics, and the camera
      shade[1].qpos[:] = clip.qpos[min(START + knot + t, END)]
      mujoco.mj_kinematics(*shade)
      old = [(p, GHOST, 0.004, CAPSULE) for plan in plans[-8:-1] for p in plan]
      window.draw(old + [(p[t:], CHOSEN, 0.008, CAPSULE) for p in now], [shade])
      if commit:
        window.pace(SUB * dt)
    knot += commit

  if args.play:
    xs, us, K = (np.load(PLAN)[k] for k in ("xs", "us", "K"))
  else:
    xs, us, K = solve(tracker, x0, warm_start(model, clip)[START:END], watch)
    np.savez(PLAN, xs=xs, us=us, K=K)
  while window.open():  # replay the plan under its feedback gains, with the clip as a ghost
    data.qpos[:], data.qvel[:] = x0[: model.nq], x0[model.nq :]
    for t in range(len(us)):
      if not window.open():
        break
      x = np.concatenate([data.qpos, data.qvel])
      u = us[t] + K[t] @ tracker.difference(xs[t : t + 1], x[None])[0]
      data.ctrl[:] = np.clip(u, tracker.lo, tracker.hi)
      shade[1].qpos[:] = clip.qpos[START + t]
      mujoco.mj_kinematics(*shade)
      for _ in range(SUB):
        mujoco.mj_step(model, data)
        window.draw([], [shade])
        window.pace(dt)
    time.sleep(1.0)  # pause before the replay repeats


if __name__ == "__main__":
  main()
