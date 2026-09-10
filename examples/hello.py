# SPDX-License-Identifier: Apache-2.0

"""4096 pendulums released from 4096 different angles and stepped together."""

import time

import mujoco
import numpy as np

from mjbatch import Batch

XML = """
<mujoco>
  <worldbody>
    <body>
      <joint axis="0 1 0" damping=".05"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.5" size=".02"/>
    </body>
  </worldbody>
</mujoco>
"""

N, STEPS = 4096, 1000
batch = Batch(mujoco.MjModel.from_xml_string(XML), N)
qpos = batch.bind("qpos")
qpos[:, 0] = np.linspace(0.1, 3.0, N)

t0 = time.perf_counter()
batch.step(nstep=STEPS)  # one call: all 1000 steps run in C, with the GIL released
elapsed = time.perf_counter() - t0

print(
  f"{N * STEPS / elapsed:.3g} sim-steps/s on {batch.num_threads} threads, "
  f"mean final angle {qpos[:, 0].mean():.4f} rad"
)
