# SPDX-License-Identifier: Apache-2.0

"""Batched release and swing-timing sweeps with native ball/pitch/bat/stump contacts.

uv run examples/cricket.py --task batting --headless
uv run examples/cricket.py --task bowling
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from mjbatch import Batch

MODEL = Path(__file__).parent / "assets" / "cricket.xml"
DURATION = 1.2
TIMESTEP = 0.00025
REST = -1.2


def build_model(task, timestep=TIMESTEP):
  model = mujoco.MjModel.from_xml_path(str(MODEL))
  model.opt.timestep = timestep
  if task == "bowling":
    blade = model.geom("blade").id
    model.geom_contype[blade] = model.geom_conaffinity[blade] = 0
  return model


def candidates(task, count, seed):
  rng = np.random.default_rng(seed)
  # Columns: initial ball velocity (m/s), swing start (s), swing duration (s).
  params = np.tile([24.0, 0.0, -3.0, 0.48, 0.16], (count, 1))
  if task == "batting":
    params[:, 3] = rng.uniform(0.35, 0.65, count)
    params[:, 4] = rng.uniform(0.08, 0.25, count)
  else:
    params[:, 1] = rng.uniform(-0.7, 0.7, count)
    params[:, 2] = rng.uniform(-4.5, -1.5, count)
  return params


def command(task, params, t):
  if task == "bowling":
    return np.full(len(params), REST)
  phase = np.clip((t - params[:, 3]) / params[:, 4], 0.0, 1.0)
  return REST + 2.4 * phase**2 * (3.0 - 2.0 * phase)


class Deliveries:
  def __init__(self, task, count, threads=0, timestep=TIMESTEP):
    self.task = task
    self.model = build_model(task, timestep)
    self.batch = Batch(self.model, count, num_threads=threads)
    self.qpos, self.qvel, self.ctrl = (self.batch.bind(f) for f in ("qpos", "qvel", "ctrl"))
    self.warning = self.batch.bind("warning")
    self.touch = self.batch.bind("sensordata")
    self.flight = self.batch.joint("flight")
    self.swing = self.batch.joint("swing")

  def reset(self, params):
    self.batch.reset()
    self.swing.qpos[:] = REST
    self.flight.qvel[:, :3] = params[:, :3]
    self.ctrl[:, 0] = REST
    self.batch.forward()

  def run(self, params, record=False):
    self.reset(params)
    steps = round(DURATION / self.model.opt.timestep)
    contacts = np.zeros((len(params), 3), dtype=bool)
    valid = np.ones(len(params), dtype=bool)
    min_distance = np.full(len(params), np.inf)
    outgoing = np.zeros(len(params))
    history = np.empty((steps + 1, len(params), self.model.nq)) if record else None
    if history is not None:
      history[0] = self.qpos
    for k in range(steps):
      self.ctrl[:, 0] = command(self.task, params, k * self.model.opt.timestep)
      self.batch.step()
      valid &= np.isfinite(self.qpos).all(1) & np.isfinite(self.qvel).all(1)
      # Touch forces belong to this step's input state; sample every substep, not just video frames.
      contacts |= self.touch > 0
      ball = self.flight.qpos[:, :3]
      min_distance = np.minimum(min_distance, np.linalg.norm(ball - [0.0, 0.0, 0.35], axis=1))
      outgoing = np.maximum(outgoing, np.where(contacts[:, 1], -self.flight.qvel[:, 0], 0.0))
      if history is not None:
        history[k + 1] = self.qpos
    valid &= (self.warning[:, :, 1] == 0).all(1)
    if self.task == "batting":
      score = np.where(contacts[:, 1] & ~contacts[:, 2], outgoing, -1.0)
    else:
      score = contacts[:, 2].astype(float) - min_distance
    score = np.where(valid, score, -np.inf)
    return score, contacts, valid, history


def replay(task, params, timestep=TIMESTEP):
  from window import CHOSEN, Window

  deliveries = Deliveries(task, 1, threads=1, timestep=timestep)
  _, _, _, history = deliveries.run(params[None], record=True)
  assert history is not None
  history = history[:, 0]
  model, data = deliveries.model, mujoco.MjData(deliveries.model)
  flight = model.joint("flight").qposadr[0]
  window = Window(model, data, title=f"Cricket: {task}")
  window.camera.lookat[:] = [-1.0, 0.0, 0.5] if task == "batting" else [-6.5, 0.0, 0.4]
  window.camera.distance = 4.5 if task == "batting" else 17.0
  window.camera.azimuth, window.camera.elevation = 115.0, -22.0
  frame, stride = 0, round(0.008 / timestep)
  while window.open():
    data.qpos[:] = history[frame]
    mujoco.mj_forward(model, data)
    window.draw(
      traces=[(history[: frame + 1, flight : flight + 3], CHOSEN, 2.0)],
      legend=("task\nplayback", f"{task}\n0.25x (selected candidate)"),
    )
    frame = (frame + stride) % len(history)
    window.pace(stride * model.opt.timestep / 0.25)
  window.close()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", choices=("batting", "bowling"), default="batting")
  p.add_argument("--num-sims", type=int, default=256)
  p.add_argument("--threads", type=int, default=0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--timestep", type=float, choices=(0.001, 0.0005, 0.00025, 0.000125), default=TIMESTEP)
  p.add_argument("--headless", action="store_true")
  a = p.parse_args()
  if a.num_sims < 1 or a.threads < 0:
    p.error("--num-sims must be positive and --threads nonnegative")
  params = candidates(a.task, a.num_sims, a.seed)
  deliveries = Deliveries(a.task, a.num_sims, a.threads, a.timestep)
  start = time.perf_counter()
  score, contacts, valid, _ = deliveries.run(params)
  elapsed = time.perf_counter() - start
  print(f"{a.task}: {a.num_sims} candidates, seed={a.seed}, dt={a.timestep}s")
  print(f"{elapsed:.3f}s (reset + control + physics + scoring)")
  print(f"valid={valid.sum()}/{a.num_sims}; pitch/bat/stump contacts={contacts.sum(0).tolist()}")
  if not valid.all():
    raise RuntimeError("nonfinite state or MuJoCo warning; do not use this sweep")
  best = int(score.argmax())
  print(f"best row={best}, score={score[best]:.6f}, parameters={params[best].tolist()}")
  if not a.headless:
    replay(a.task, params[best], a.timestep)


if __name__ == "__main__":
  main()
