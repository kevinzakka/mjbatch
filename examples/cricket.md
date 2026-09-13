# Cricket Contact Sweeps

[`cricket.py`](cricket.py) runs independent deliveries through `Batch`'s CPU thread
pool. The batting task sweeps the start time and duration of an actuated bat swing;
the bowling task sweeps lateral and downward release velocity with the bat disabled.
Both use the same free ball, gravity, pitch and static stump collision geometry.
No additional dependencies, downloaded assets or checkpoints are needed.

```sh
uv run examples/cricket.py --task batting --headless --num-sims 256 --threads 2 --seed 0
uv run examples/cricket.py --task bowling --headless --num-sims 256 --threads 2 --seed 0
```

Omit `--headless` for a 0.25x replay of the highest-scoring candidate with its ball
trajectory. Replay reruns that candidate with the same timestep and plays recorded
MuJoCo states, rather than interpolating a synthetic hit. The printed contact counts
cover **every candidate**, including misses and stump contacts; the selected replay
is an illustration, not an evaluation sample. Contact categories can overlap.

## Model and Objectives

* The 0.156 kg ball has a 0.036 m radius. It starts 14 m before the stumps at
  1.8 m height with 24 m/s forward velocity. Velocity is initialized once at reset;
  subsequent bounce, deflection and spin come from MuJoCo contacts.
* The bat is a gravity-compensated, one-hinge rigid tool, not a human arm. A cubic
  position command drives its swing with torque limited to 60 N m. Its blade can
  contact only the ball, not the pitch or stumps. The handle is noncolliding.
* Touch sensors for the pitch, blade and stumps are sampled after **each physics
  step**. With `Batch`'s default `forward=False`, these are the forces computed by
  that step, not a later `forward()` evaluation of the resulting position.
* Batting maximizes observed return speed toward the bowler after blade contact,
  in m/s. Any stump contact, or no blade contact, gives score -1.
* Bowling uses a screening heuristic: a stump-contact indicator minus the minimum
  sampled distance in metres to `(0, 0, 0.35)`. This is not a cricket scoring rule.
* Nonfinite states or MuJoCo warnings invalidate a row; the command fails if any
  row is invalid instead of silently ranking only surviving simulations.

This is a reduced contact/control example, **not RL training**, a full bowler run-up,
human biomechanics, regulation cricket, or a calibrated ball/pitch material model.
It has no aerodynamic swing, spin-induced lift, deformable bat, bails or fielders.
The elapsed time includes reset, control, physics and scoring, excludes construction
and replay, and is not a speedup claim against serial MuJoCo.

## Numerical Checks

```sh
uv run pytest tests/test_cricket.py
uv run examples/cricket.py --task batting --headless --threads 2 --timestep 0.000125
uv run examples/cricket.py --task bowling --headless --threads 2 --timestep 0.000125
```

The tests compare every qpos, qvel and touch-force sample against independent serial
MuJoCo runs, with one and three worker threads. They independently verify sensor
events against native geom pairs with positive contact force, plus free-flight
integration, full reset replay, seed/thread invariance, joint/actuator units, and
viewer replay timing. A fixed 16-delivery regression checks contact classification
under timestep halving; this does not establish convergence for all impacts.

For the **entire 256-candidate, seed-0 sweep** on MuJoCo 3.11.0:

| Timestep | Batting: blade / stump contacts | Bowling: stump contacts |
| --- | ---: | ---: |
| 0.5 ms | 163 / 132 | 92 |
| 0.25 ms (default) | 159 / 131 | 93 |
| 0.125 ms | 154 / 134 | 93 |

All 256 candidates in each sweep contacted the pitch, with no invalid rows. Halving
the default timestep changed contact classification for 8/256 batting rows and
0/256 bowling rows; best-candidate identities also changed. Grazing impacts and
objective rankings remain timestep-sensitive. Use the finer-step option to check
candidates; these contact counts are not calibrated physical accuracy claims.
