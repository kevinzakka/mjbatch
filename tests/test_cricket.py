# SPDX-License-Identifier: Apache-2.0

import mujoco
import numpy as np
import pytest

import cricket


@pytest.mark.parametrize("task", ["batting", "bowling"])
@pytest.mark.parametrize("threads", [1, 3])
def test_serial_contact_trajectories(task, threads):
  params = cricket.candidates(task, 8, seed=7)
  deliveries = cricket.Deliveries(task, len(params), threads)
  deliveries.reset(params)
  model = deliveries.model
  data = [mujoco.MjData(model) for _ in params]
  qadr, vadr = model.joint("flight").qposadr[0], model.joint("flight").dofadr[0]
  swing = model.joint("swing").qposadr[0]
  for d, p in zip(data, params, strict=True):
    d.qpos[swing], d.qvel[vadr : vadr + 3], d.ctrl[0] = cricket.REST, p[:3], cricket.REST
    mujoco.mj_forward(model, d)
  seen = np.zeros((len(params), 3), dtype=bool)
  pairs = ("pitch", "blade", "stump_")
  for k in range(round(cricket.DURATION / model.opt.timestep)):
    u = cricket.command(task, params, k * model.opt.timestep)
    deliveries.ctrl[:, 0] = u
    deliveries.batch.step()
    for i, d in enumerate(data):
      d.ctrl[0] = u[i]
      mujoco.mj_step(model, d)
      np.testing.assert_array_equal(deliveries.qpos[i], d.qpos)
      np.testing.assert_array_equal(deliveries.qvel[i], d.qvel)
      np.testing.assert_array_equal(deliveries.touch[i], d.sensordata)
      # Check touch events against native geom pairs and positive normal force.
      actual = np.zeros(3, dtype=bool)
      for j in range(d.ncon):
        contact = d.contact[j]
        names = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
        force = np.zeros(6)
        mujoco.mj_contactForce(model, d, j, force)
        if "ball" in names and force[0] > 0:
          for s, name in enumerate(pairs):
            actual[s] |= any(n.startswith(name) for n in names)
      np.testing.assert_array_equal(d.sensordata > 0, actual)
      seen[i] |= actual
  assert seen[:, 0].all()
  assert seen[:, 1].any() if task == "batting" else not seen[:, 1].any()
  assert seen[:, 2].any()
  assert (deliveries.warning[:, :, 1] == 0).all()
  assert np.isfinite(deliveries.qpos[:, qadr : qadr + 3]).all()


@pytest.mark.parametrize("task", ["batting", "bowling"])
def test_reset_and_recording_reproduce_complete_sweep(task):
  params = cricket.candidates(task, 16, seed=0)
  deliveries = cricket.Deliveries(task, len(params), threads=2)
  first = deliveries.run(params, record=True)
  deliveries.run(cricket.candidates(task, 16, seed=99))
  again = deliveries.run(params, record=True)
  for a, b in zip(first, again, strict=True):
    np.testing.assert_array_equal(a, b)
  score, contacts, valid, history = first
  assert valid.all()
  assert history is not None
  assert history.shape == (round(cricket.DURATION / cricket.TIMESTEP) + 1, 16, deliveries.model.nq)
  assert np.isfinite(score).all()
  assert contacts.shape == (16, 3)
  if task == "batting":
    np.testing.assert_array_equal(score == -1, ~contacts[:, 1] | contacts[:, 2])
  else:
    np.testing.assert_array_equal(score > 0, contacts[:, 2])
  swing = deliveries.model.joint("swing").qposadr[0]
  np.testing.assert_allclose(history[0, :, swing], cricket.REST)
  if task == "batting":
    assert np.ptp(history[:, :, swing]) > 2.0
    assert history[:, :, swing].min() >= -1.41
    assert history[:, :, swing].max() <= 1.41


def test_free_flight_before_contact():
  deliveries = cricket.Deliveries("bowling", 3, threads=2)
  params = cricket.candidates("bowling", 3, seed=1)
  deliveries.reset(params)
  initial = deliveries.flight.qpos[:, :3].copy()
  n, dt = 100, deliveries.model.opt.timestep
  deliveries.batch.step(nstep=n)
  expected = initial + params[:, :3] * (n * dt)
  # implicitfast uses the updated velocity for position integration.
  expected[:, 2] -= 9.81 * dt**2 * n * (n + 1) / 2
  np.testing.assert_allclose(deliveries.flight.qpos[:, :3], expected, atol=1e-12, rtol=0)
  np.testing.assert_allclose(deliveries.flight.qvel[:, :3], params[:, :3] + [0, 0, -9.81 * n * dt])
  assert not deliveries.touch.any()


@pytest.mark.parametrize("task", ["batting", "bowling"])
def test_seed_and_thread_invariance(task):
  params = cricket.candidates(task, 32, seed=0)
  np.testing.assert_array_equal(params, cricket.candidates(task, 32, seed=0))
  assert not np.array_equal(params, cricket.candidates(task, 32, seed=1))
  single = cricket.Deliveries(task, 32, threads=1).run(params)
  threaded = cricket.Deliveries(task, 32, threads=3).run(params)
  for a, b in zip(single[:3], threaded[:3], strict=True):
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("task", ["batting", "bowling"])
def test_contact_classification_under_timestep_refinement(task):
  params = cricket.candidates(task, 16, seed=0)
  coarse = cricket.Deliveries(task, 16, threads=2).run(params)
  fine = cricket.Deliveries(task, 16, threads=2, timestep=cricket.TIMESTEP / 2).run(params)
  np.testing.assert_array_equal(coarse[1], fine[1])
  assert coarse[2].all() and fine[2].all()
  # Stable regression cases, not a claim that grazing impacts are timestep-independent.
  assert coarse[1][:, 2].any() and not coarse[1][:, 2].all()
  if task == "batting":
    assert coarse[0].max() > 10.0 and fine[0].max() > 10.0


def test_actuator_limits_and_no_ball_forcing():
  model = cricket.build_model("batting")
  assert model.nu == 1
  assert model.actuator_trnid[0, 0] == model.joint("swing").id
  np.testing.assert_array_equal(model.jnt_range[model.joint("swing").id], [-1.4, 1.4])
  np.testing.assert_array_equal(model.actuator_forcerange, [[-60.0, 60.0]])
  assert model.neq == 0
  params = cricket.candidates("batting", 4, seed=4)
  deliveries = cricket.Deliveries("batting", 4, threads=1)
  _, _, valid, _ = deliveries.run(params)
  assert valid.all()
  assert not deliveries.batch.bind("qfrc_applied").any()
  assert not deliveries.batch.bind("xfrc_applied").any()


def test_warning_invalidates_row_even_after_mujoco_recovers(monkeypatch, tmp_path):
  monkeypatch.chdir(tmp_path)
  reset = cricket.Deliveries.reset

  def bad_reset(self, params):
    reset(self, params)
    self.flight.qpos[0, 0] = np.nan

  monkeypatch.setattr(cricket.Deliveries, "reset", bad_reset)
  params = cricket.candidates("batting", 4, seed=0)
  deliveries = cricket.Deliveries("batting", 4, threads=2)
  score, _, valid, _ = deliveries.run(params)
  assert np.isfinite(deliveries.qpos).all()
  np.testing.assert_array_equal(valid, [False, True, True, True])
  assert score[0] == -np.inf
  assert deliveries.warning[0, :, 1].sum() > 0


def test_replay_uses_selected_timestep_and_native_history(monkeypatch):
  import window

  frames = []

  class Window:
    def __init__(self, model, data, title):
      assert model.opt.timestep == 0.0005
      self.data = data
      self.camera = mujoco.MjvCamera()

    def open(self):
      return len(frames) < 3

    def draw(self, traces, legend):
      assert traces[0][0].shape[1] == 3
      assert np.isfinite(traces[0][0]).all()
      frames.append(self.data.qpos.copy())

    def pace(self, dt):
      assert dt == pytest.approx(0.032)

    def close(self):
      pass

  monkeypatch.setattr(window, "Window", Window)
  cricket.replay("batting", cricket.candidates("batting", 1, seed=0)[0], timestep=0.0005)
  assert len(frames) == 3
  assert not np.array_equal(frames[0], frames[-1])
