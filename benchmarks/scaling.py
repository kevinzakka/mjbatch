# SPDX-License-Identifier: Apache-2.0

"""Physics throughput versus thread count for a model, one and ten substeps per call.

uv run python benchmarks/scaling.py path/to/scene.xml --threads 1,8,24

The model may instead be a Menagerie name, which loads that model's default scene, or a
path whose meshes Menagerie holds:

uv run python benchmarks/scaling.py unitree_g1
uv run python benchmarks/scaling.py examples/assets/g1.xml --assets unitree_g1
"""

import argparse
import os
import time

import mujoco
import numpy as np

from mjbatch import Batch


def serial(model: mujoco.MjModel, ctrl: np.ndarray, keyframe: int, substeps: int) -> float:
  """The baseline the speedups are against: one mjData stepped in a plain Python loop."""
  data = mujoco.MjData(model)
  if keyframe >= 0:
    mujoco.mj_resetDataKeyframe(model, data, keyframe)
  data.ctrl[:] = ctrl[0]
  for _ in range(20):
    mujoco.mj_step(model, data)
  t0 = time.perf_counter()
  for _ in range(substeps):
    mujoco.mj_step(model, data)
  return substeps / (time.perf_counter() - t0)


def rate(batch: Batch, ctrl: np.ndarray, nstep: int, substeps: int) -> float:
  batch.bind("ctrl")[:] = ctrl
  for _ in range(20):
    batch.step()
  t0 = time.perf_counter()
  for _ in range(substeps // nstep):
    batch.step(nstep=nstep)
  return substeps * batch.num_sims / (time.perf_counter() - t0)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("model")
  p.add_argument("--num-sims", type=int, default=256)
  p.add_argument("--threads", default="1,2,4,8")
  p.add_argument("--keyframe", type=int, default=-1)
  p.add_argument("--assets", help="a Menagerie model whose assets the XML refers to")
  a = p.parse_args()
  if a.assets or not os.path.exists(a.model):
    import mujoco_menagerie

    if a.assets:
      spec = mujoco.MjSpec.from_file(a.model, assets=mujoco_menagerie.get(a.assets).assets())
    else:
      spec = mujoco_menagerie.get(a.model).spec("scene")
    model = spec.compile()
  else:
    model = mujoco.MjModel.from_xml_path(a.model)
  ctrl = np.random.default_rng(0).standard_normal((a.num_sims, model.nu)) * 0.3
  print(f"{os.path.basename(a.model)}  num_sims={a.num_sims}  cores={os.cpu_count()}")
  base = serial(model, ctrl, a.keyframe, 200)
  print(f"  serial mj_step loop {base:10,.0f} sim-substeps/s")
  for threads in map(int, a.threads.split(",")):
    batch = Batch(model, a.num_sims, num_threads=threads)
    batch.reset(keyframe=a.keyframe)
    one, ten = rate(batch, ctrl, 1, 200), rate(batch, ctrl, 10, 200)
    print(
      f"  threads={threads:3d}  1 substep/call {one:10,.0f} ({one / base:4.1f}x)"
      f"  10/call {ten:10,.0f} sim-substeps/s"
    )


if __name__ == "__main__":
  main()
