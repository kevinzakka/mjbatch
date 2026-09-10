# SPDX-License-Identifier: Apache-2.0

"""Identifying the link inertias of a Flexiv Rizon from one recorded motion."""

import argparse
import copy
import time
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np

from mjbatch import Batch
from window import CAPSULE, CHOSEN, Window, polyline, progress

ASSETS = Path(__file__).parent / "assets"
DT, SUB, T, WINDOW = 0.01, 5, 1000, 50  # s per command, substeps, commands recorded, per window
DIM, EPS, ITERS, TOL = 70, 1e-6, 25, 1e-5  # 7 links x 10 coordinates, difference step, stop
NOISE, PRIOR = 2e-4, 0.35  # rad of encoder noise; dimensionless CAD-coordinate prior std
ALPHAS = 0.5 ** np.arange(5)  # line search steps
HOME = np.array([0.0, -0.45, 0.15, 1.6, -0.15, 0.6, 0.0])  # rad
POSES = np.array([
  HOME, [-0.65, -0.8, -0.45, 1.15, 0.55, 0.7, -0.7], [0.65, -0.5, 0.6, 1.3, -0.7, 1.0, 0.85],
  [0.25, -0.15, -0.45, 2.1, 0.8, 0.25, -0.65], [-0.45, -0.65, 0.5, 1.5, -0.5, 0.8, 0.65], HOME,
])  # fmt: skip
AMPLITUDE = np.array([0.08, 0.08, 0.16, 0.10, 0.23, 0.18, 0.3])  # rad of multisine per joint
FREQ = np.array([0.37, 0.43, 0.59, 0.53, 0.83, 0.71, 1.07])  # Hz
KP = np.array([180, 240, 120, 140, 50, 45, 25])  # N m / rad
KD = np.array([45, 55, 25, 25, 6, 6, 3])  # N m s / rad
CENTER, EXTENT = np.array([0.3, 0.0, 0.48]), 1.05  # m, what the free camera frames
INITIAL = np.random.default_rng(12).uniform(-0.3, 0.3, DIM)  # the guess the fit starts from
HIDDEN = np.random.default_rng(4).normal(0, 0.18, DIM)  # the inertias the recording is made with
FIELDS = ("body_mass", "body_ipos", "body_inertia", "body_iquat")
ESTIMATE, ACTUAL = (0.76, 0.55, 0.43, 0.35), (0.60, 0.67, 0.75, 0.28)  # muted clay / blue-gray
SHELL, PIN = (0.72, 0.69, 0.64, 0.30), (0.98, 0.92, 0.76, 1.0)  # see-through link, centre of mass
MORPH, TRAIL, FPS = 25, 120, 50  # frames per fit shown, samples of tip trail, frames per second
COLLECT = ("recording", f"{T * DT:.0f} s at {1 / DT:.0f} Hz")


def build_model():
  spec = mujoco.MjSpec.from_file(str(ASSETS / "theme.xml"))
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  robot = mm.get("flexiv_rizon4").spec("flexiv_rizon4")
  for body in robot.bodies:
    body.gravcomp = 0
  for actuator in robot.actuators:
    actuator.set_to_motor()
    actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
  for light in list(robot.lights):
    robot.delete(light)
  robot.body("link7").add_site(name="tip", pos=[0, 0, 0.08], size=[0.012, 0, 0], rgba=ESTIMATE)
  spec.attach(robot, frame=spec.worldbody.add_frame(), prefix="")
  spec.option.timestep = DT / SUB
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  spec.visual.global_.azimuth, spec.visual.global_.elevation = 135, -18
  spec.visual.global_.fovy = 38
  model = spec.compile()
  model.stat.center[:], model.stat.extent = CENTER, EXTENT
  return model


class Inertias:
  """Each link's inertia as ten dimensionless coordinates, zero at CAD, valid by construction.

  The 4x4 pseudo-inertia P factors as L L.T; the coordinates fill a lower triangular U with a
  positive diagonal, and the link is L_CAD U.
  """

  def __init__(self, model):
    self.ids = np.array([model.body(f"link{i}").id for i in range(1, 8)])
    mass, c, moments = (getattr(model, f)[self.ids] for f in FIELDS[:3])
    rot = np.empty((7, 3, 3))
    for r, q in zip(rot, model.body_iquat[self.ids], strict=True):
      mujoco.mju_quat2Mat(r.ravel(), q)
    inertia = (rot * moments[:, None]) @ rot.swapaxes(-1, -2)
    central = 0.5 * np.trace(inertia, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) - inertia
    p = np.empty((7, 4, 4))
    p[:, :3, :3] = central + mass[:, None, None] * c[:, :, None] * c[:, None, :]
    p[:, :3, 3] = p[:, 3, :3] = mass[:, None] * c
    p[:, 3, 3] = mass
    self.base = np.linalg.cholesky(p)

  def decode(self, theta):
    """Return each link's mass, centre of mass, principal moments and inertial frame."""
    x = np.asarray(theta).reshape(-1, 7, 10)
    u = np.zeros((*x.shape[:2], 4, 4))
    rows, cols = np.tril_indices(4, -1)
    u[..., np.arange(4), np.arange(4)] = np.exp(x[..., :4])
    u[..., rows, cols] = x[..., 4:]
    factor = self.base @ u
    p = factor @ factor.swapaxes(-1, -2)
    mass = p[..., 3, 3]
    c = p[..., :3, 3] / mass[..., None]
    central = p[..., :3, :3] - mass[..., None, None] * c[..., :, None] * c[..., None, :]
    inertia = np.trace(central, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) - central
    moments, rot = np.linalg.eigh(inertia)
    rot[..., :, 0] *= np.linalg.det(rot)[..., None]  # eigh may hand back a reflection
    quat = np.empty((*mass.shape, 4))
    for q, r in zip(quat.reshape(-1, 4), rot.reshape(-1, 9), strict=True):
      mujoco.mju_mat2Quat(q, r)
    return mass, c, moments, quat

  def apply(self, target, theta):
    """Write one candidate per sim of a Batch, or the first candidate into a plain model."""
    batched = isinstance(target, Batch)
    for name, value in zip(FIELDS, self.decode(theta), strict=True):
      if batched:
        field = target.expand(name)
        field.reshape(len(value), -1, *field.shape[1:])[:, :, self.ids] = value[:, None]
      else:
        getattr(target, name)[self.ids] = value[0]
    if batched:
      target.set_const()
    else:
      mujoco.mj_setConst(target, mujoco.MjData(target))
      target.stat.center[:], target.stat.extent = CENTER, EXTENT  # mj_setConst rewrote both


def excitation(t, phase=0.0):
  """Return the reference joint angles at time t: eased poses plus a per-joint multisine."""
  t = np.asarray(t)
  segment = np.clip(t / (T * DT) * 5, 0, 5 - 1e-9)
  i, a = segment.astype(int), segment % 1
  a = np.clip((a - 0.12) / 0.76, 0, 1)  # dwell for 12% at each end
  blend = a**3 * (10 - 15 * a + 6 * a * a)
  pose = POSES[i] + blend[..., None] * (POSES[i + 1] - POSES[i])
  wave = np.sin(2 * np.pi * FREQ * t[..., None] + phase + np.arange(7) * 0.7)
  return pose + np.sin(np.pi * t / (T * DT))[..., None] ** 2 * AMPLITUDE * wave


def record(model, phase=0.0):
  """Track the excitation with PD over a bias-force feedforward, commanded as joint torque.

  Return every state and every torque; qfrc_bias carries Coriolis and centrifugal, not just gravity.
  """
  data = mujoco.MjData(model)
  data.qpos[:] = HOME
  mujoco.mj_forward(model, data)
  limits = model.jnt_actfrcrange[:, 1]
  times = np.arange(T) * DT
  desired = excitation(times, phase)
  velocity = (excitation(times + 1e-4, phase) - excitation(times - 1e-4, phase)) / 2e-4
  states, torques = [np.r_[data.qpos, data.qvel]], []
  for t in range(T):
    mujoco.mj_forward(model, data)  # mj_step leaves qfrc_bias one step stale
    error, rate = desired[t] - data.qpos, velocity[t] - data.qvel
    torque = np.clip(data.qfrc_bias + KP * error + KD * rate, -limits, limits)
    data.ctrl[:] = torque
    mujoco.mj_step(model, data, nstep=SUB)
    states.append(np.r_[data.qpos, data.qvel])
    torques.append(torque)
  return np.array(states), np.array(torques)


def rollout(batch, inertia, theta, x0, torque):
  """Return the angles of every window, shape (time, candidate, window, joint). The starts and the
  torques broadcast over candidates; only the inertias differ between them."""
  inertia.apply(batch, theta)
  batch.reset()
  q, v, ctrl = (batch.bind(f).reshape(len(theta), -1, 7) for f in ("qpos", "qvel", "ctrl"))
  q[:], v[:] = x0[:, :7], x0[:, 7:]
  batch.forward()
  out = np.empty((len(torque) + 1, *q.shape))
  out[0] = q
  for t, action in enumerate(torque):
    ctrl[:] = action
    batch.step(nstep=SUB)
    out[t + 1] = q
  return out


def identify(model, inertia, observed, torque):
  """Fit the 70 coordinates by Levenberg-Marquardt over twenty short windows of the recording,
  each started from a recorded state. Return every accepted (theta, data RMS)."""
  starts = np.arange(0, T, WINDOW)
  noisy = observed[:, :7] + np.random.default_rng(10).normal(0, NOISE, (T + 1, 7))
  target = np.stack([noisy[s + 1 : s + WINDOW + 1] for s in starts], axis=1)
  controls = np.stack([torque[s : s + WINDOW] for s in starts], axis=1)
  delta = np.r_[np.zeros((1, DIM)), EPS * np.eye(DIM)]  # row 0 is the unperturbed candidate
  probe = Batch(model, len(starts) * len(delta))
  line = Batch(model, len(starts) * len(ALPHAS))

  def residual(batch, params):
    predicted = rollout(batch, inertia, params, observed[starts], controls)[1:]
    return (predicted - target[:, None]).transpose(0, 2, 3, 1).reshape(-1, len(params))

  theta, mu = INITIAL.copy(), 1e-3
  r = residual(probe, theta + delta)
  # MAP: mean((angle error / NOISE)^2) + mean((theta / PRIOR)^2), rescaled into angle units.
  weight = (NOISE / PRIOR) ** 2 / len(r)
  mse = np.mean(r[:, 0] ** 2)
  cost, stop = mse + weight * (theta @ theta), False
  fits, start = [(theta.copy(), np.sqrt(mse))], time.perf_counter()
  for it in range(ITERS + 1):
    last, rms = stop or it == ITERS, fits[-1][1]
    progress(f"fit {len(fits) - 1:2d}/{ITERS}", 1.0 if last else (len(fits) - 1) / ITERS,
             time.perf_counter() - start, f"data RMS {1000 * rms:.3f} mrad", end=last)  # fmt: skip
    if last:
      break
    jac = (r[:, 1:] - r[:, :1]) / EPS
    h = jac.T @ jac / len(r) + weight * np.eye(DIM)
    g = jac.T @ r[:, 0] / len(r) + weight * theta
    step = np.linalg.solve(h + mu * np.diag(np.maximum(np.diag(h), 1e-12)), -g)
    params = np.clip(theta + ALPHAS[:, None] * step, -0.8, 0.8)
    errors = residual(line, params)
    costs = np.mean(errors**2, axis=0) + weight * np.sum(params**2, axis=1)
    best = int(np.argmin(costs))
    if costs[best] >= cost:
      mu *= 10
      stop = mu > 1e5
      continue
    drop, rms = cost - costs[best], np.sqrt(np.mean(errors[:, best] ** 2))
    theta, cost, mu = params[best], costs[best], max(mu / 10, 1e-12)
    fits.append((theta.copy(), rms))
    stop = rms <= 1.01 * NOISE or drop < TOL * cost  # stop at the noise floor, not below it
    if not stop:
      r = residual(probe, theta + delta)
  return fits


def tip_path(model, states):
  """Return the tip site's world position at every state, shape (len(states), 3)."""
  batch = Batch(model, len(states))
  batch.bind("qpos")[:] = states[:, :7]
  tip = batch.site("tip").xpos
  batch.forward()
  return tip.copy()


def shapes(model, data, ids):
  """Return the uniform ellipsoid matching each body's inertia: semi-axes, centre, frame."""
  mass, moment = model.body_mass[ids], model.body_inertia[ids]
  size = np.sqrt(5 * np.maximum(0.5 * moment.sum(-1)[:, None] - moment, 0) / mass[:, None])
  return list(zip(size, data.xipos[ids], data.ximat[ids], strict=True))


def rings(ellipsoids):
  """Return the three great circles of every ellipsoid, 33 points each."""
  angle, out = np.linspace(0, 2 * np.pi, 33)[:, None], []
  for size, center, mat in ellipsoids:
    axes = mat.reshape(3, 3) * size
    for a, b in ((0, 1), (0, 2), (1, 2)):
      out.append(center + np.cos(angle) * axes[:, a] + np.sin(angle) * axes[:, b])
  return out


def add_geom(scene, kind, size, pos, mat, color):
  if scene.ngeom == scene.maxgeom:
    return
  geom = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(geom, kind, size, pos, mat, color)
  geom.category = mujoco.mjtCatBit.mjCAT_DECOR
  geom.shininess, geom.specular = 0.6, 0.4
  scene.ngeom += 1


def show(model, inertia, truth, observed, fits, held_out):
  """Play the recording, then loop either the fitted inertias or the held-out prediction."""
  model, held = copy.copy(model), INITIAL.copy()  # the caller keeps CAD; this copy is morphed
  data = mujoco.MjData(model)
  moving = np.isin(model.geom_bodyid, inertia.ids)
  solid = model.geom_matid.copy(), model.geom_rgba.copy()
  clear = np.where(moving, -1, solid[0]), np.where(moving[:, None], SHELL, solid[1])
  shade = copy.copy(model)
  shade.geom_matid[moving], shade.geom_rgba[moving] = -1, ACTUAL
  shade.site_rgba[:, 3] = 0
  ghost = (shade, mujoco.MjData(shade))
  reference = mujoco.MjData(truth)
  reference.qpos[:] = HOME
  mujoco.mj_forward(truth, reference)
  actual = shapes(truth, reference, inertia.ids)
  circles = rings(actual)
  window = Window(model, data, title="Rizon / identify inertia")
  mujoco.mjv_defaultFreeCamera(model, window.camera)

  def ellipsoids(scene):
    estimate = shapes(model, data, inertia.ids)
    for group, color in ((actual, ACTUAL[:3] + (0.20,)), (estimate, ESTIMATE)):
      for size, pos, mat in group:
        add_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, size, pos, mat, color)
    for circle in circles:
      polyline(scene, circle, ACTUAL[:3] + (0.55,), 0.0015, CAPSULE)
    for _, pos, _ in estimate:
      add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.007), pos, np.eye(3).ravel(), PIN)

  def morph(theta, rms, i):
    """Ease the links from the coordinates last shown to theta, both ellipsoid sets drawn."""
    nonlocal held
    model.geom_matid[:], model.geom_rgba[:] = clear
    data.qpos[:] = HOME
    previous = held if i else theta  # a repeat restarts at the bad initial guess
    measured = "not fitted" if rms is None else f"{1000 * rms:.3f} mrad"
    for a in np.linspace(0, 1, MORPH):
      if not window.open():
        return
      # Interpolate Cholesky coordinates, not ambiguous principal-axis rotations.
      inertia.apply(model, previous + a * a * (3 - 2 * a) * (theta - previous))
      mujoco.mj_forward(model, data)
      window.draw(legend=("data RMS", measured), decor=ellipsoids)
      window.pace(1 / FPS)
    held = theta.copy()

  def replay(states, other=None, legend=None):
    """Play a joint trajectory, with a second one behind it as a translucent ghost."""
    model.geom_matid[:], model.geom_rgba[:] = solid
    trail = tip_path(model, states)
    behind = None if other is None else tip_path(model, other)
    for t in range(0, len(states), 2):
      if not window.open():
        return
      data.qpos[:] = states[t, :7]
      mujoco.mj_forward(model, data)
      traces, ghosts = [(trail[max(0, t - TRAIL) : t + 1], CHOSEN, 0.003)], []
      if other is not None:
        ghost[1].qpos[:] = other[t, :7]
        mujoco.mj_forward(*ghost)
        traces.append((behind[max(0, t - TRAIL) : t + 1], ACTUAL[:3] + (0.65,), 0.002))
        ghosts = [ghost]
      window.draw(traces, ghosts, legend=legend)
      window.pace(2 * DT)

  try:
    replay(observed, legend=COLLECT)
    while window.open():
      if held_out is not None:
        replay(*held_out)
      else:
        for i, (theta, rms) in enumerate(fits):
          morph(theta, rms, i)
  finally:
    window.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--predict", action="store_true", help="loop the held-out prediction")
  parser.add_argument("--preview", action="store_true", help="the excitation and the guess only")
  args = parser.parse_args()
  model = build_model()
  inertia = Inertias(model)
  truth = copy.copy(model)
  inertia.apply(truth, HIDDEN)
  observed, torque = record(truth)
  fits, held_out = [(INITIAL, None)], None
  if not args.preview:
    fits = identify(model, inertia, observed, torque)
    # Two active seconds of a new recording: one initial state, then torque replay, no resets.
    unseen, controls = record(truth, phase=1.4)
    unseen, controls = unseen[200:401], controls[200:400]
    guesses = np.stack([fits[0][0], fits[-1][0]])
    predicted = rollout(Batch(model, 2), inertia, guesses, unseen[:1], controls[:, None])[:, :, 0]
    rms = np.sqrt(np.mean((predicted - unseen[:, None, :7]) ** 2, axis=(0, 2)))
    print(f"prediction: initial {rms[0]:.3e}, fitted {rms[1]:.3e} rad "
          f"({rms[0] / rms[1]:.0f}x improvement)")  # fmt: skip
    held_out = (predicted[:, 1], unseen) if args.predict else None
  if not args.headless:
    show(model, inertia, truth, observed, fits, held_out)


if __name__ == "__main__":
  main()
