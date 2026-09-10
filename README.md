# mjbatch

[![Build](https://img.shields.io/github/actions/workflow/status/kevinzakka/mjbatch/ci.yml?branch=main)](https://github.com/kevinzakka/mjbatch/actions)
[![PyPI version](https://img.shields.io/pypi/v/mjbatch)](https://pypi.org/project/mjbatch/)

`mjbatch` is a Python library for running thousands of MuJoCo simulations in parallel on CPU.

Features include:

* C++ thread pool execution, with the GIL released;
* Live array access to simulation state and controls across the batch;
* Per-simulation model parameters, with `set_const` to recompute derived constants.

For example:

```python
import mujoco, numpy as np
from mjbatch import Batch

model = mujoco.MjModel.from_xml_path("scene.xml")
batch = Batch(model, num_sims=4096)  # threads default to every logical CPU
qpos, ctrl = batch.bind("qpos"), batch.bind("ctrl")
batch.expand("geom_friction")[:, :, 0] = np.random.uniform(0.4, 1.2, (4096, 1))
for _ in range(1000):
  ctrl[:] = policy(qpos)             # your controller, all 4096 at once
  batch.step()                       # step them in parallel; qpos updates in place
```

## Examples

We showcase a range of applications built using `mjbatch`: RL, MPC, SysID, and hardware
co-design. Each example is a self-contained, performant implementation. For instance, the Go1
RL controller learns to walk in under a minute on a five-year-old M1 laptop.

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/cartpole_swingup.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/cartpole_swingup.gif" alt="cart-pole swing-up"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/cartpole_mpc.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/cartpole_mpc.gif" alt="cart-pole MPC"></a>
    </td>
  </tr>
  <tr>
    <td align="center">A two-pole cart swung upright with <a href="https://ieeexplore.ieee.org/document/6386025">iLQR</a></td>
    <td align="center">A cart-pole swing-up controller using <a href="https://arxiv.org/abs/2212.00541">predictive sampling</a></td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/g1_flip.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/g1_flip.gif" alt="G1 backflip"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/go1_joystick.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/go1_joystick.gif" alt="Go1 joystick"></a>
    </td>
  </tr>
  <tr>
    <td align="center">A G1 humanoid tracking a reference backflip with receding-horizon iLQR</td>
    <td align="center">A Go1 quadruped joystick controller trained with PPO</td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/arm_throw.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/arm_throw.gif" alt="throwing arm co-design"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/rizon_inertia.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/rizon_inertia.gif" alt="Rizon inertia identification"></a>
    </td>
  </tr>
  <tr>
    <td align="center">CEM jointly optimizes a robot arm's proportions, gears, and controls</td>
    <td align="center">Damped Gauss–Newton fits a Rizon arm's inertial parameters to synthetic motion data</td>
  </tr>
</table>

Run with `uv run examples/<file>.py`; some need `uv sync --group examples`. The ones that open
a window need a display; `--headless` runs the solver without one.

## License

Apache-2.0.
