# SPDX-License-Identifier: Apache-2.0

"""CEM co-designing a throwing arm's proportions, gears and torques."""

import argparse
import time
from functools import partial
from pathlib import Path

import mujoco
import numpy as np

from mjbatch import Batch
from window import CAPSULE, CHOSEN, GHOST, Window, polyline, progress

ASSETS = Path(__file__).parent / "assets"
DT, G, SOLVER = 0.002, 9.81, (40, 1e-10)  # s, m/s^2, solver iterations and tolerance
REACH, RADIUS = 1.0, 0.045  # m
DEADLINE, KNOTS = 0.65, 4  # s, knots per motor
STALL, FREE, ROTOR = np.array([6.0, 3.0]), 24.0, 0.0006  # N m, rad/s, kg m^2 at the motor
DENSITY, ELBOW_MOTOR, ARMATURE = 0.6, 0.15, 0.001  # kg/m, kg, kg m^2
ARM, BALL = slice(0, 2), slice(2, 5)  # the joints, then the ball: same columns in qpos and qvel
FRAC, GEARS, RELEASE, SIGNALS = 0, slice(1, 3), 3, slice(4, None)
LOW = np.r_[0.22, 0.5, 0.5, 0.25, np.full(2 * KNOTS, -1.0)]
HIGH = np.r_[0.78, 2.5, 2.5, DEADLINE, np.ones(2 * KNOTS)]
STOCK = np.array([0.5, 1.0, 1.0])  # equal links, direct drive
POP, GENERATIONS, ELITE, SEED, MISS = 512, 30, 0.1, 0, -10.0
COOL, TICK, LANES = (0.45, 0.54, 0.58, 1.0), 0.13, (0.48, -0.48)
HOLD, SLOW, FPS, LOOP, RECORD = 1.0, 0.55, 50, 7.0, 2.5  # s, rate, fps, s per loop, s recorded


def build_spec():
  spec = mujoco.MjSpec.from_file(str(ASSETS / "arm_throw.xml"))
  spec.option.timestep, spec.option.integrator = DT, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.iterations, spec.option.tolerance = SOLVER
  spec.geom("ball").size[:] = [RADIUS, 0, 0]
  spec.geom("floor").contype = spec.geom("floor").conaffinity = 1  # only ball and floor collide
  return spec


def shape(target, designs, prefix=""):
  """Write one design per sim: link lengths, inertia and transmissions."""
  batched = isinstance(target, Batch)
  model = target.model if batched else target

  def field(name):
    return target.expand(name) if batched else getattr(model, name)[None]

  designs = np.atleast_2d(designs)
  lengths = REACH * np.column_stack((designs[:, FRAC], 1.0 - designs[:, FRAC]))
  beam = model.geom_size[model.geom(prefix + "upper_beam").id]  # half length, width, thickness
  hub = model.geom_size[model.geom(prefix + "upper_hub").id]  # radius, half length
  width, thick, radius, length_hub = 2 * beam[1], 2 * beam[2], hub[0], 2 * hub[1]
  for j, name in enumerate(("upper", "lower")):
    length = lengths[:, j]
    rod, motor = DENSITY * length, ELBOW_MOTOR * j  # the motor sits at the elbow, on the forearm
    mass = rod + motor
    centre = rod * length / (2 * mass)
    ix = rod * (width**2 + thick**2) / 12 + motor * (3 * radius**2 + length_hub**2) / 12
    iy = rod * (length**2 + thick**2) / 12 + motor * radius**2 / 2
    iz = rod * (length**2 + width**2) / 12 + motor * (3 * radius**2 + length_hub**2) / 12
    shift = rod * (length / 2 - centre) ** 2 + motor * centre**2
    body = model.body(prefix + name).id
    field("body_mass")[:, body] = mass
    field("body_ipos")[:, body, 0] = centre
    field("body_inertia")[:, body] = np.column_stack((ix, iy + shift, iz + shift))
    for suffix, inset in (("beam", 0.0), ("trim", 0.05)):
      geom = model.geom(f"{prefix}{name}_{suffix}").id
      field("geom_pos")[:, geom, 0] = length / 2
      field("geom_size")[:, geom, 0] = length / 2 - inset
  field("body_pos")[:, model.body(prefix + "lower").id, 0] = lengths[:, 0]
  field("body_pos")[:, model.body(prefix + "hand").id, 0] = lengths[:, 1]
  motors = [model.actuator(prefix + k + "_motor").id for k in ("shoulder", "elbow")]
  dofs = [model.joint(prefix + k).dofadr[0] for k in ("shoulder", "elbow")]
  field("actuator_gear")[:, motors, 0] = designs[:, GEARS]
  field("dof_armature")[:, dofs] = ARMATURE + ROTOR * designs[:, GEARS] ** 2
  if batched:
    target.set_const()
  else:
    mujoco.mj_setConst(model, mujoco.MjData(model))


def commands(designs, t):
  phase = np.clip(t / designs[:, RELEASE], 0, 1) * (KNOTS - 1)
  left = np.minimum(phase.astype(int), KNOTS - 2)
  values, rows = designs[:, SIGNALS].reshape(-1, KNOTS, 2), np.arange(len(designs))
  weight = (phase - left)[:, None]
  return (1 - weight) * values[rows, left] + weight * values[rows, left + 1]


def landing(pos, vel):
  flight = (vel[:, 2] + np.sqrt(vel[:, 2] ** 2 + 2 * G * np.maximum(pos[:, 2] - RADIUS, 0))) / G
  return pos[:, 0] + flight * vel[:, 0], flight


class Throws:
  def __init__(self, n):
    self.batch = Batch(build_spec().compile(), n)
    self.q, self.v, self.u = (self.batch.bind(k) for k in ("qpos", "qvel", "ctrl"))
    self.grasp, self.warning = self.batch.bind("eq_active"), self.batch.bind("warning")
    self.grip = self.batch.site("grip").xpos

  def reset(self, designs):
    shape(self.batch, designs)
    self.batch.reset()
    self.q[:, ARM], self.v[:], self.u[:], self.grasp[:] = [-np.pi / 2, 0.0], 0, 0, 1
    self.batch.forward()
    self.q[:, BALL] = self.grip
    self.batch.forward()

  def run(self, designs, record=False):
    """Step every arm to its release and score the throw from the ball's state there."""
    self.reset(designs)
    n, path = len(designs), []
    released, at = np.zeros(n, bool), np.ceil(designs[:, RELEASE] / DT).astype(int)
    pos, vel = np.zeros((n, 3)), np.zeros((n, 3))
    for k in range(round((DEADLINE + RECORD * record) / DT) + 1):
      leaving = (k >= at) & ~released
      pos[leaving], vel[leaving] = self.q[leaving, BALL], self.v[leaving, BALL]
      self.grasp[leaving], released = 0, released | leaving
      if released.all() and not record:
        break
      signal = commands(designs, k * DT)
      signal[released] = -np.tanh(self.v[released, ARM]) if record else 0.0
      speed = self.v[:, ARM] * designs[:, GEARS]
      self.u[:] = STALL * signal * np.clip(1 - np.sign(signal) * speed / FREE, 0, 1)
      if record:
        path.append(self.q.copy())
      self.batch.step(None if record else ~released)  # a released arm cannot change its throw
    distance, flight = landing(pos, vel)
    thrown = (vel[:, 0] > 0) & (self.warning.reshape(n, -1).sum(1) == 0)
    return dict(score=np.where(thrown, distance, MISS), distance=distance, pos=pos, vel=vel,
                flight=flight, path=np.asarray(path))  # fmt: skip


def search(pop, rng, label, fixed=False):
  """Run CEM: draw a generation, keep the best tenth, move the distribution onto them."""
  mean, std = (LOW + HIGH) / 2, (HIGH - LOW) / 3
  best, top, history = None, -np.inf, []
  start = time.perf_counter()
  for it in range(GENERATIONS):
    designs = np.clip(rng.normal(mean, std, (POP, len(LOW))), LOW, HIGH)
    if fixed:
      designs[:, :3] = STOCK  # the baseline: same search, only the throw is free
    if best is not None:
      designs[0] = best
    score = pop.run(designs)["score"]
    elite = designs[np.argsort(score)[-round(ELITE * POP) :]]
    mean, std = 0.2 * mean + 0.8 * elite.mean(0), 0.2 * std + 0.8 * elite.std(0)
    std = np.maximum(std, 0.015 * (HIGH - LOW))  # never collapse onto the incumbent
    if score.max() > top:
      best, top = designs[score.argmax()].copy(), score.max()
    history.append(best.copy())
    done = (it + 1) / GENERATIONS
    progress(f"{label} {it + 1:2d}/{GENERATIONS}", done, time.perf_counter() - start,
             f"range {top:5.2f} m", end=it + 1 == GENERATIONS)  # fmt: skip
  return best, np.asarray(history)


def stage(designs):
  spec = mujoco.MjSpec.from_file(str(ASSETS / "theme.xml"))
  spec.option.timestep, spec.option.integrator = DT, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.iterations, spec.option.tolerance = SOLVER  # match the arm, so attach has no conflict
  for i, color in enumerate((COOL, CHOSEN)):
    arm = build_spec()
    arm.delete(arm.geom("floor"))
    for light in list(arm.lights):
      arm.delete(light)
    for material in ("clay", "ball"):
      arm.material(material).rgba[:] = color
    spec.attach(arm, prefix=f"d{i}_", frame=spec.worldbody.add_frame(pos=[0, LANES[i], 0]))
  model = spec.compile()
  for i, design in enumerate(designs):
    shape(model, design, f"d{i}_")
  return model, mujoco.MjData(model)


def tried(history):
  kept = history[np.r_[True, np.any(np.diff(history, axis=0) != 0, axis=1)]]
  kept = kept[np.linspace(0, len(kept) - 1, min(8, len(kept))).astype(int)]
  earlier = Throws(len(kept)).run(kept)
  arcs = []
  for pos, vel, span in zip(earlier["pos"], earlier["vel"], earlier["flight"], strict=True):
    t = np.linspace(0, span, 80)[:, None]
    arcs.append(pos + vel * t - [0, 0, G / 2] * t**2 + [0, LANES[1], 0])
  return arcs


def rule(scene, extent, distance, landed):
  for i, color in enumerate((COOL, CHOSEN)):
    y, end = LANES[i], int(np.ceil(extent))
    polyline(scene, np.array([[0, y, 0.008], [end, y, 0.008]]), (*color[:3], 0.35), 0.007)
    for x in range(1, end + 1):
      polyline(
        scene, np.array([[x, y - TICK, 0.009], [x, y + TICK, 0.009]]), (*color[:3], 0.8), 0.01
      )
    if landed[i]:
      x, angles = distance[i], np.linspace(0, 2 * np.pi, 49)
      polyline(scene, np.array([[x, y - 0.20, 0.012], [x, y + 0.20, 0.012]]), color, 0.022)
      circle = (x + 0.11 * np.cos(angles), y + 0.11 * np.sin(angles), np.full(49, 0.012))
      polyline(scene, np.column_stack(circle), color, 0.012)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--headless", action="store_true")
  args = parser.parse_args()
  pop = Throws(POP)
  stock, _ = search(pop, np.random.default_rng(SEED), "throw only", fixed=True)
  best, history = search(pop, np.random.default_rng(SEED), "arm + throw")
  designs = np.array([stock, best])
  results = Throws(2).run(designs, record=True)
  path, distance, flight = results["path"], results["distance"], results["flight"]
  shoulder, elbow = best[GEARS]
  print(f"upper / lower {best[FRAC]:.2f} / {1 - best[FRAC]:.2f} m, gears {shoulder:.2f} at the "
        f"shoulder and {elbow:.2f} at the elbow, release {best[RELEASE]:.3f} s")  # fmt: skip
  print(f"stock {distance[0]:.2f} m, co-design {distance[1]:.2f} m: "
        f"+{100 * (distance[1] / distance[0] - 1):.0f}%")  # fmt: skip
  if args.headless:
    return

  model, data = stage(designs)
  window = Window(model, data, title="throwing / co-design")
  window.camera.azimuth, window.camera.elevation = 94, -16
  window.option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False
  arcs, down = tried(history), designs[:, RELEASE] + flight
  extent = max(3.0, float(distance.max()))
  legend = ("stock\nco-design", "{:.2f} m\n{:.2f} m   (+{:.0f}%)".format(
    *distance, 100 * (distance[1] / distance[0] - 1)))  # fmt: skip
  frame = 0
  try:
    while window.open():
      sim = max(0.0, (frame / FPS) % LOOP - HOLD) * SLOW
      index = min(round(sim / DT), len(path) - 1)
      for i in range(2):  # freeze on landing; the ball rolls on
        data.qpos[9 * i : 9 * (i + 1)] = path[min(index, round(down[i] / DT)), i]
        data.qpos[9 * i + 3] += LANES[i]
      mujoco.mj_forward(model, data)

      zoom = min(1.0, float(path[: index + 1, :, BALL][..., 0].max()) / extent)
      window.camera.distance = 4.4 + (max(5.0, 0.85 * extent + 1.0) - 4.4) * zoom
      window.camera.lookat[:] = [0.25 + (0.47 * extent - 0.25) * zoom, 0, 0.85 + 0.45 * zoom]
      traces = [(arc, GHOST, 0.003) for arc in arcs] if zoom > 0.5 else []
      for i, color in enumerate((COOL, CHOSEN)):
        points = path[: min(index, round(down[i] / DT)) + 1 : 5, i, BALL]
        if len(points) > 1:
          traces.append((points + [0, LANES[i], 0], color, 0.006 + 0.006 * zoom, CAPSULE))
      marks = partial(rule, extent=extent, distance=distance, landed=sim >= down)
      window.draw(traces, legend=legend, decor=marks)
      window.pace(1 / FPS)
      frame += 1
  finally:
    window.close()


if __name__ == "__main__":
  main()
