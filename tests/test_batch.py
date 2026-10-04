# SPDX-License-Identifier: Apache-2.0

import copy
import subprocess
import sys
import threading

import mujoco
import numpy as np
import pytest

from mjbatch import Batch

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="cam" pos="1 1 1"/>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <site name="base"/>
      <body name="pole" pos="0 0 .05" gravcomp="0">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
        <site name="tip" pos="0 0 .5"/>
      </body>
    </body>
    <body name="puck" pos="0 1 .5">
      <joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>
    </body>
    <body name="mocap" mocap="true" pos="1 0 1">
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <tendon><spatial name="t"><site site="base"/><site site="tip"/></spatial></tendon>
  <equality><weld body1="mocap" body2="cart" active="false"/></equality>
  <actuator><motor joint="slide" gear="10"/><position joint="hinge" kp="1"/></actuator>
  <sensor><jointpos joint="hinge"/><framepos objtype="site" objname="tip"/></sensor>
  <keyframe><key qpos="0.5 0.2"/></keyframe>
</mujoco>
"""
N = 8

# Activation dynamics, a mocap weld, a keyframe, and sensors that set mjData's
# lazy-evaluation flags (accelerometer, subtreelinvel).
LOCKSTEP_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <body name="pole" pos="0 0 .05">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
        <site name="tip" pos="0 0 .5"/>
      </body>
    </body>
    <body name="ball" pos="0 1 .5">
      <freejoint/>
      <geom type="sphere" size=".05" mass=".2"/>
    </body>
    <body name="mocap" mocap="true" pos="0 1 .5">
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality><weld body1="mocap" body2="ball"/></equality>
  <actuator>
    <motor joint="slide" gear="5"/>
    <general joint="hinge" dyntype="filter" dynprm="0.02" gainprm="5"
             biastype="affine" biasprm="0 -5 0"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <accelerometer site="tip"/>
    <subtreelinvel body="cart"/>
  </sensor>
  <keyframe>
    <key qpos="0.3 0.5 0 1 .5 1 0 0 0" ctrl="0.1 0.2" act="0.2" mpos="0 1 .5"/>
  </keyframe>
</mujoco>
"""
COMPARED = ("qpos", "qvel", "act", "time", "sensordata", "site_xpos")


@pytest.fixture
def model():
  return mujoco.MjModel.from_xml_string(XML)


def lockstep(nstep, num_threads, heavy=()):
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  models = [model] * N
  batch = Batch(model, N, num_threads=num_threads)
  if heavy:
    batch.expand("body_mass")[list(heavy), 1] *= 3.0
    batch.set_const(np.array(heavy))
    for i in heavy:
      models[i] = copy.copy(model)
      models[i].body_mass[1] *= 3.0
      mujoco.mj_setConst(models[i], mujoco.MjData(models[i]))
  datas = [mujoco.MjData(m) for m in models]
  pending = [[] for _ in range(N)]
  bound = {f: batch.bind(f) for f in COMPARED}
  rng = np.random.default_rng(0)

  def write(field, ids, index, value):
    batch.bind(field)[(ids, *index)] = value
    for j, i in enumerate(ids):
      pending[i].append((field, index, value[j]))

  def apply(i):
    for field, index, value in pending[i]:
      getattr(datas[i], field)[index] = value
    pending[i].clear()

  every = list(range(N))
  for call in range(50):
    write("ctrl", every, (slice(None),), rng.uniform(-1, 1, (N, model.nu)))
    if call == 10:
      write("qpos", [1, 4, 6], (slice(0, 2),), rng.uniform(-0.3, 0.3, (3, 2)))
    if call == 15:
      write("mocap_pos", every, (0,), rng.uniform(-0.2, 0.2, (N, 3)) + [0, 1, 0.5])
    if call == 20:
      write("xfrc_applied", [2, 3], (3, slice(0, 3)), rng.uniform(-1, 1, (2, 3)))
    if call == 30:
      write("eq_active", [0, 2, 4, 6], (0,), np.zeros(4, np.uint8))
    ids = [0, 2, 3, 7] if call % 4 == 3 else every
    if call == 25:
      ids = [1, 5]
      batch.reset(np.array(ids), keyframe=0)
      for i in ids:
        mujoco.mj_resetDataKeyframe(models[i], datas[i], 0)
        apply(i)
        mujoco.mj_forward(models[i], datas[i])
    else:
      batch.step(None if len(ids) == N else np.array(ids), nstep=nstep)
      for i in ids:
        apply(i)
        for _ in range(nstep):
          mujoco.mj_step(models[i], datas[i])
    for field, arr in bound.items():
      np.testing.assert_array_equal(arr, [getattr(d, field) for d in datas], field)


@pytest.mark.parametrize("nstep", [1, 3])
@pytest.mark.parametrize("num_threads", [1, 3, 7])
def test_lockstep_with_mj_step(nstep, num_threads):
  lockstep(nstep, num_threads)


@pytest.mark.parametrize("nstep", [1, 3])
def test_lockstep_with_expanded_mass(nstep):
  lockstep(nstep, num_threads=4, heavy=(1, 2, 5))


def test_sleep_is_rejected():
  xml = LOCKSTEP_XML.replace("<option", '<option><flag sleep="enable"/></option><option')
  with pytest.raises(ValueError, match="sleep"):
    Batch(mujoco.MjModel.from_xml_string(xml), N)


def test_per_sim_gravity_matches_separate_models():
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch = Batch(model, N, num_threads=3)
  gravity = batch.expand("gravity")
  gravity[:, 2] = np.linspace(-9.81, -1.0, N)
  batch.bind("ctrl")[:, 0] = 0.5
  qpos, qvel = batch.bind("qpos"), batch.bind("qvel")
  batch.step(nstep=50)
  for i in range(N):
    m = copy.copy(model)
    m.opt.gravity[2] = gravity[i, 2]
    d = mujoco.MjData(m)
    d.ctrl[0] = 0.5
    for _ in range(50):
      mujoco.mj_step(m, d)
    np.testing.assert_array_equal(qpos[i], d.qpos)
    np.testing.assert_array_equal(qvel[i], d.qvel)


def test_per_sim_timestep(model):
  batch = Batch(model, N, num_threads=2)
  timestep = batch.expand("timestep")
  timestep[:] = 0.001 * (1 + np.arange(N))
  assert timestep.shape == (N,)
  time = batch.bind("time")
  batch.step(nstep=4)
  np.testing.assert_allclose(time, 4 * timestep)


def test_per_sim_integrator(model):
  batch = Batch(model, N)
  integrator = batch.expand("integrator")
  assert integrator.dtype == np.int32
  np.testing.assert_array_equal(integrator, model.opt.integrator)
  integrator[::2] = mujoco.mjtIntegrator.mjINT_RK4
  batch.bind("ctrl")[:, 0] = 1.0
  qvel = batch.bind("qvel")
  batch.step(nstep=20)
  np.testing.assert_array_equal(qvel[1], qvel[3])
  assert not np.array_equal(qvel[0], qvel[1])
  ref = copy.copy(model)
  ref.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
  d = mujoco.MjData(ref)
  d.ctrl[0] = 1.0
  for _ in range(20):
    mujoco.mj_step(ref, d)
  np.testing.assert_array_equal(qvel[0], d.qvel)


def test_expanded_option_seeds_from_the_template(model):
  batch, ref = Batch(model, N), Batch(model, N)
  gravity = batch.expand("gravity")
  np.testing.assert_array_equal(gravity, np.tile(model.opt.gravity, (N, 1)))
  np.testing.assert_array_equal(batch.expand("o_solref"), np.tile(model.opt.o_solref, (N, 1)))
  for b in (batch, ref):
    b.bind("ctrl")[:, 0] = 1.0
    b.step(nstep=20)
  np.testing.assert_array_equal(batch.bind("state"), ref.bind("state"))


def test_option_is_untouched_by_set_const(model):
  # mj_setConst never writes opt, so no opt field can be flagged as one of its
  # outputs and expanded behind the caller's back.
  ref = mujoco.MjModel.from_xml_string(XML)
  before = {f: np.array(getattr(ref.opt, f), copy=True) for f in dir(ref.opt) if f[0] != "_"}
  mujoco.mj_setConst(ref, mujoco.MjData(ref))
  for f, v in before.items():
    np.testing.assert_array_equal(getattr(ref.opt, f), v, f)
  batch = Batch(model, N)
  timestep = batch.expand("timestep")
  timestep[:] = 0.001 * (1 + np.arange(N))
  batch.expand("body_mass")[:, model.body("pole").id] = 0.5
  batch.set_const()
  np.testing.assert_array_equal(timestep, 0.001 * (1 + np.arange(N)))
  time = batch.bind("time")
  batch.step()
  np.testing.assert_allclose(time, timestep)


def test_per_sim_sleep_is_rejected(model):
  batch = Batch(model, N, num_threads=2)
  enableflags = batch.expand("enableflags")
  enableflags[3] |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  batch.step(np.array([0, 1]))  # sim 3 is not running
  for call in (batch.step, batch.forward, batch.set_const):
    with pytest.raises(ValueError, match="sim 3: sleep"):
      call()
  enableflags[3] = model.opt.enableflags
  batch.step()


def test_step_history_matches_a_loop(model):
  batch, ref = Batch(model, N, num_threads=3), Batch(model, N, num_threads=3)
  for b in (batch, ref):
    b.bind("ctrl")[:, 0] = np.linspace(-1, 1, N)
  history = np.empty((N, 7, batch.nstate))
  batch.step(nstep=7, history=history)
  np.testing.assert_array_equal(history[:, -1], batch.bind("state"))
  for k in range(7):
    ref.step()
    np.testing.assert_array_equal(history[:, k], ref.bind("state"))


def test_step_history_with_ids(model):
  batch = Batch(model, N)
  batch.bind("ctrl")[:, 0] = np.linspace(-1, 1, N)
  ids = np.array([1, 4, 6])
  history = np.empty((len(ids), 5, batch.nstate))
  batch.step(ids, nstep=5, history=history)
  state, time = batch.bind("state"), batch.bind("time")
  np.testing.assert_array_equal(history[:, -1], state[ids])
  np.testing.assert_allclose(history[:, :, 0], np.tile(np.arange(1, 6) * 0.002, (3, 1)))
  assert not np.any(time[[0, 2, 3, 5, 7]])


def test_step_history_validation(model):
  batch = Batch(model, N)
  nstate = batch.nstate
  for bad in (
    np.empty((N, 3)),
    np.empty((N, 2, nstate)),
    np.empty((N - 1, 3, nstate)),
    np.empty((N, 3, nstate), np.float32),
    np.empty((N, 3, nstate + 1))[:, :, :-1],
  ):
    with pytest.raises(ValueError, match="history"):
      batch.step(nstep=3, history=bad)
  with pytest.raises(ValueError, match="history"):
    batch.step(np.array([0, 1]), nstep=3, history=np.empty((N, 3, nstate)))
  assert not np.any(batch.bind("time"))


def test_step_history_accepts_degenerate_strides(model):
  # numpy gives a size-0 array all-zero strides and a new axis a zero stride.
  batch = Batch(model, N)
  batch.step(np.zeros(N, dtype=bool), nstep=3, history=np.empty((0, 3, batch.nstate)))
  history = np.empty((4, batch.nstate))
  batch.step(np.array([2]), nstep=4, history=history[None])
  np.testing.assert_array_equal(history[-1], batch.bind("state")[2])
  assert not np.any(np.delete(batch.bind("time"), 2))


def test_step_history_error_names_the_sim():
  batch = Batch(mujoco.MjModel.from_xml_string(LOCKSTEP_XML), N, num_threads=2)
  batch.expand("eq_type")[2] = 99
  history = np.zeros((N, 3, batch.nstate))
  with pytest.raises(RuntimeError, match="sim 2"):
    batch.step(nstep=3, history=history)
  assert np.any(history[0])  # the sims that ran still wrote their rows


@pytest.mark.skipif(sys.platform == "win32", reason="resource is Unix-only")
def test_memory_does_not_scale_with_num_sims():
  # A fresh process, so ru_maxrss growth is this batch's. One mjData per sim
  # grows it by 712 MB here; 4096 state vectors are a few MB.
  code = """
import resource, sys, mujoco
from mjbatch import Batch
xml = '<mujoco><size memory="256K"/><worldbody><body><joint/><geom size=".02"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
scale = 1 if sys.platform == "darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
batch = Batch(model, 4096)
batch.step()
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) * scale)
"""
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
  assert out.returncode == 0, out.stderr
  assert int(out.stdout) < 256 * 2**20


@pytest.mark.parametrize("num_threads", [1, 3])
def test_step_error_keeps_the_failing_sim(num_threads):
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch = Batch(model, N, num_threads=num_threads)
  eq_type = batch.expand("eq_type")
  eq_type[2] = 99
  ctrl, qpos = batch.bind("ctrl"), batch.bind("qpos")
  ctrl[:, 0] = np.linspace(-1, 1, N)
  with pytest.raises(RuntimeError, match="sim 2"):
    batch.step(nstep=3)
  # The others ran; sim 2 kept its state and its pending write, and the worker
  # it failed on serves later sims correctly.
  eq_type[2] = model.eq_type
  batch.step()
  one, four = reference(model, ctrl, 1), reference(model, ctrl, 4)
  for i in range(N):
    np.testing.assert_array_equal(qpos[i], (one if i == 2 else four)[i].qpos)


def reference(model, ctrl, nstep):
  datas = [mujoco.MjData(model) for _ in range(N)]
  for i, d in enumerate(datas):
    d.ctrl[:] = ctrl[i]
    for _ in range(nstep):
      mujoco.mj_step(model, d)
  return datas


@pytest.mark.parametrize("num_threads", [1, 4])
def test_step_matches_reference(model, num_threads):
  batch = Batch(model, N, num_threads=num_threads)
  ctrl = batch.bind("ctrl")
  ctrl[:, 0] = np.linspace(-1, 1, N)
  qpos, qvel, sensordata = (batch.bind(f) for f in ("qpos", "qvel", "sensordata"))
  for _ in range(25):
    batch.step()
  batch.step(nstep=25)
  for i, d in enumerate(reference(model, ctrl, 50)):
    np.testing.assert_array_equal(qpos[i], d.qpos)
    np.testing.assert_array_equal(qvel[i], d.qvel)
    np.testing.assert_array_equal(sensordata[i], d.sensordata)


def test_forward_after_step(model):
  """With forward=True the derived fields are current after a step; without, one
  substep behind, as with mj_step."""
  for forward in (False, True):
    batch = Batch(model, N, forward=forward)
    ctrl, xpos, sensordata = (batch.bind(f) for f in ("ctrl", "xpos", "sensordata"))
    ctrl[:, 0] = np.linspace(-1, 1, N)
    batch.step(nstep=10)
    behind = xpos.copy(), sensordata.copy()
    batch.forward()
    same = np.array_equal(behind[0], xpos) and np.array_equal(behind[1], sensordata)
    assert same == forward


def test_shapes_and_dtypes_match_mjdata(model):
  batch = Batch(model, N)
  d = mujoco.MjData(model)
  for name in dir(d):
    if name.startswith("_"):
      continue
    val = getattr(d, name)
    if not isinstance(val, np.ndarray) or name in ("contact", "efc_type"):
      continue
    try:
      arr = batch.bind(name)
    except ValueError:
      continue
    assert arr.shape == (N, *val.shape), name
    assert arr.dtype == val.dtype, name
  assert batch.bind("time").shape == (N,)
  batch = Batch(model, N)
  assert batch.bind("qpos", np.float32).dtype == np.float32
  with pytest.raises(ValueError):
    batch.bind("qpos")  # already bound as float32
  with pytest.raises(ValueError):
    batch.bind("eq_active", np.float32)
  with pytest.raises(ValueError):
    batch.bind("nope")
  with pytest.raises(ValueError, match=r'use expand\("geom_friction"\)'):
    batch.bind("geom_friction")
  with pytest.raises(ValueError, match=r'use bind\("qpos"\)'):
    batch.expand("qpos")
  with pytest.raises(ValueError):
    batch.expand("mesh_vert")
  with pytest.raises(ValueError):
    Batch(model, 0)


def test_float32_rows_only_written_when_changed(model):
  batch = Batch(model, N, num_threads=2)
  ctrl = batch.bind("ctrl", np.float32)
  qpos = batch.bind("qpos", np.float32)
  ctrl[:, 0] = 0.3
  for _ in range(200):
    batch.step()
  # Physics ran in mjtNum precision: untouched float32 rows never round-trip.
  ref_ctrl = np.zeros((N, model.nu))
  ref_ctrl[:, 0] = np.float32(0.3)
  d = reference(model, ref_ctrl, 200)[0]
  np.testing.assert_array_equal(qpos[0], d.qpos.astype(np.float32))
  # A written row takes effect on the next call; a derived field is read-only.
  site_xpos = batch.bind("site_xpos")
  before = site_xpos[1].copy()
  qpos[1, 1] = 0.7
  site_xpos[0] = 42.0
  batch.forward()
  assert not np.array_equal(site_xpos[1], before)
  assert not np.any(site_xpos[0] == 42.0)


def test_expand_and_set_const_are_complete(model):
  pole = model.body("pole").id
  results = []
  for num_threads in (1, 4):
    batch = Batch(model, N, num_threads=num_threads)
    mass = batch.expand("body_mass")
    mass[:, pole] = np.linspace(0.1, 1.0, N)
    batch.set_const()
    # Everything mj_setConst derived is now per sim, in native dtype.
    subtree = batch.expand("body_subtreemass")
    np.testing.assert_allclose(subtree[:, pole], mass[:, pole])
    assert batch.expand("dof_invweight0").shape == (N, model.nv)
    batch.bind("ctrl")[:, 0] = 1.0
    qvel = batch.bind("qvel")
    batch.step(nstep=20)
    results.append(qvel.copy())
  np.testing.assert_array_equal(results[0], results[1])
  ref = mujoco.MjModel.from_xml_string(XML)
  ref.body_mass[pole] = 1.0
  mujoco.mj_setConst(ref, mujoco.MjData(ref))
  d = mujoco.MjData(ref)
  d.ctrl[0] = 1.0
  for _ in range(20):
    mujoco.mj_step(ref, d)
  np.testing.assert_array_equal(results[0][-1], d.qvel)


def test_gravcomp_flag_is_per_sim(model):
  pole = model.body("pole").id
  results = []
  for num_threads in (1, 4):
    batch = Batch(model, N, num_threads=num_threads)
    gravcomp = batch.expand("body_gravcomp")
    gravcomp[::2, pole] = 1.0
    batch.set_const()
    batch.bind("qpos")[:, 1] = 0.3
    qvel = batch.bind("qvel")
    batch.step(nstep=5)
    results.append(qvel.copy())
  np.testing.assert_array_equal(results[0], results[1])
  assert np.all(results[0][::2, 1] != results[0][1::2, 1])


def test_reset_keyframe_and_expanded_qpos0(model):
  batch = Batch(model, N)
  qpos0 = batch.expand("qpos0")
  qpos0[:, 1] = np.arange(N) * 0.1
  qpos = batch.bind("qpos")
  batch.bind("ctrl")[:, 0] = 1.0
  batch.step()
  qpos[5, 0] = 0.25  # A write survives a reset of that sim and lands after it.
  xpos = batch.bind("xpos")
  batch.reset(np.array([2, 5]))
  np.testing.assert_array_equal(qpos[2], qpos0[2])
  assert qpos[5, 0] == 0.25
  np.testing.assert_array_equal(qpos[5, 1:], qpos0[5, 1:])
  assert qpos[0, 0] != 0.0
  assert xpos[2, 1, 2] != 0.0  # reset forwards
  mask = np.zeros(N, dtype=bool)
  mask[0] = True
  batch.reset(mask)
  assert qpos[0, 0] == 0.0
  batch.reset(keyframe=0)
  np.testing.assert_array_equal(qpos, np.tile(model.key_qpos[0], (N, 1)))
  for bad in (1, -2):
    with pytest.raises(ValueError):
      batch.reset(keyframe=bad)


def test_ids_validation_and_time(model):
  batch = Batch(model, N, num_threads=3)
  time = batch.bind("time")
  batch.step(np.array([1, 3]))
  np.testing.assert_array_equal(time, [0, 0.002, 0, 0.002, 0, 0, 0, 0])
  batch.step(np.array([1, 3], dtype=np.int32))
  np.testing.assert_array_equal(time, [0, 0.004, 0, 0.004, 0, 0, 0, 0])
  for bad in ([N], [1, 1], [3, 1], [1.0, 2.0]):
    with pytest.raises(ValueError):
      batch.step(np.array(bad))


def test_warning_counters(model):
  batch = Batch(model, N)
  warning = batch.bind("warning")
  batch.bind("qpos")[2] = np.nan
  batch.step()
  assert warning.shape == (N, mujoco.mjtWarning.mjNWARNING.value, 2)
  assert warning[2, mujoco.mjtWarning.mjWARN_BADQPOS, 1] == 1
  assert warning[:, :, 1].sum() == 1


@pytest.mark.parametrize("num_threads", [1, 2])
def test_mujoco_error_becomes_exception(model, num_threads):
  batch = Batch(model, N, num_threads=num_threads)
  mass = batch.expand("body_mass")
  mass[3, model.body("puck").id] = 0.0
  mass[5, model.body("pole").id] = 2.0
  with pytest.raises(RuntimeError, match="sim 3"):
    batch.set_const()
  # The other sims still ran; a later call works after fixing the input.
  assert batch.expand("body_subtreemass")[5, model.body("pole").id] == 2.0
  mass[3, model.body("puck").id] = 1.0
  batch.set_const()


def test_set_const_subset_keeps_every_sim_consistent(model):
  batch = Batch(model, N)
  pole = model.body("pole").id
  batch.expand("body_mass")[:, pole] = 5.0
  batch.set_const(np.array([0, 1]))
  np.testing.assert_array_equal(batch.expand("body_subtreemass")[:, pole], 5.0)


def test_concurrent_calls_are_serialized(model):
  batch = Batch(model, N, num_threads=4)
  time = batch.bind("time")

  other = Batch(model, N, num_threads=4)
  other_time = other.bind("time")

  def work(b, nstep):
    for _ in range(50):
      b.step(nstep=nstep)

  threads = [threading.Thread(target=work, args=a) for a in ((batch, 1), (batch, 5), (other, 2))]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  np.testing.assert_allclose(time, 300 * 0.002)
  np.testing.assert_allclose(other_time, 100 * 0.002)


def test_named_views(model):
  batch = Batch(model, N)
  hinge, tip = batch.joint("hinge"), batch.site("tip")
  assert hinge.qpos.shape == (N, 1) and hinge.qvel.shape == (N, 1)
  batch.bind("ctrl")[:, 0] = 1.0
  batch.step(nstep=10)
  np.testing.assert_array_equal(hinge.qpos[:, 0], batch.bind("qpos")[:, 1])
  np.testing.assert_array_equal(tip.xpos, batch.bind("site_xpos")[:, model.site("tip").id])
  assert batch.body("pole").xquat.shape == (N, 4)
  assert batch.body("pole").xpos.shape == (N, 3)


def test_state_rows_copy_restore_and_compose(model):
  batch = Batch(model, N)
  state, qpos, xpos = batch.bind("state"), batch.bind("qpos"), batch.bind("xpos")
  assert batch.nstate == mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
  assert state.shape == (N, batch.nstate) and state.dtype == np.float64
  batch.bind("ctrl")[:] = np.arange(N)[:, None] * 0.1
  batch.step(nstep=5)
  # Copying a row copies the physics: sim 1 becomes sim 0 in every field.
  assert not np.array_equal(qpos[0], qpos[1])
  state[1] = state[0]
  batch.forward(np.array([0, 1]))  # sim 0's derived fields lag by a substep
  np.testing.assert_array_equal(qpos[1], qpos[0])
  np.testing.assert_array_equal(xpos[1], xpos[0])
  # A state write and a field write compose, the field write winning on its overlap,
  # and the step matches mj_setState + mj_step on the same data.
  data = mujoco.MjData(model)
  saved = state[3].copy()
  batch.step(nstep=3)
  state[3] = saved
  qpos[3, 0] = 0.7
  mujoco.mj_setState(model, data, saved, mujoco.mjtState.mjSTATE_INTEGRATION)
  data.qpos[0], data.ctrl[:] = 0.7, batch.bind("ctrl")[3]
  batch.step(np.array([3]))
  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(qpos[3], data.qpos)
  np.testing.assert_array_equal(state[3, 1 : 1 + model.nq], data.qpos)
  # Round trip: restoring the rows replays the same steps bit for bit.
  before = state.copy()
  batch.step(nstep=2)
  once = state.copy()
  state[:] = before
  batch.step(nstep=2)
  np.testing.assert_array_equal(state, once)
  # reset applies a pending state write element by element, like a field write;
  # float32 is refused; the view is the same.
  state[2] = once[5]
  merged = mujoco.MjData(model)
  reset = np.empty(batch.nstate)
  mujoco.mj_getState(model, merged, reset, mujoco.mjtState.mjSTATE_INTEGRATION)
  written = np.where(once[5] != once[2], once[5], reset)
  mujoco.mj_setState(model, merged, written, mujoco.mjtState.mjSTATE_INTEGRATION)
  batch.reset(np.array([2]))
  np.testing.assert_array_equal(qpos[2], merged.qpos)
  np.testing.assert_array_equal(state[2], written)
  with pytest.raises(ValueError):
    batch.bind("state", np.float32)
  assert np.shares_memory(batch.bind("state"), state)


SLIDER_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body><joint name="j" type="slide"/><geom size=".1" mass="1"/></body>
    <body name="mocap" mocap="true"><geom size=".02" contype="0" conaffinity="0"/></body>
  </worldbody>
  <actuator><motor joint="j"/></actuator>
</mujoco>
"""


def slider(num_sims):
  model = mujoco.MjModel.from_xml_string(SLIDER_XML)
  s0 = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
  mujoco.mj_getState(model, mujoco.MjData(model), s0, mujoco.mjtState.mjSTATE_INTEGRATION)
  return model, Batch(model, num_sims), s0


def test_field_write_after_state_write_reaches_the_sim():
  _, batch, s0 = slider(1)
  ctrl, state, qvel = batch.bind("ctrl"), batch.bind("state"), batch.bind("qvel")
  ctrl[:] = 1.0
  batch.step()
  assert qvel[0, 0] == 0.002
  state[:] = s0  # ctrl 0, and the ctrl view says so at once: it is the state row
  assert ctrl[0, 0] == 0.0 and np.shares_memory(ctrl, state)
  ctrl[:] = 1.0  # the value the view held before the state write
  batch.step()
  assert qvel[0, 0] == 0.002


def test_mocap_write_after_state_write_reaches_the_sim():
  model, batch, s0 = slider(1)
  mocap_pos, state, xpos = batch.bind("mocap_pos"), batch.bind("state"), batch.bind("xpos")
  body = model.body("mocap").id
  mocap_pos[0, 0] = [0.3, 0.2, 0.1]
  batch.forward()
  np.testing.assert_array_equal(xpos[0, body], [0.3, 0.2, 0.1])
  state[:] = s0
  mocap_pos[0, 0] = [0.3, 0.2, 0.1]
  batch.forward()
  np.testing.assert_array_equal(xpos[0, body], [0.3, 0.2, 0.1])


def test_field_write_after_state_write_with_ids():
  _, batch, s0 = slider(3)
  ctrl, state, qvel = batch.bind("ctrl"), batch.bind("state"), batch.bind("qvel")
  ctrl[:] = 1.0
  batch.step()
  state[:] = s0
  ctrl[:] = 1.0
  batch.step(np.array([0, 2]))
  np.testing.assert_array_equal(qvel[:, 0], [0.002, 0.0, 0.002])
  batch.step(np.array([1]))  # sim 1's writes waited for its next call
  np.testing.assert_array_equal(qvel[:, 0], [0.002, 0.002, 0.002])


def test_state_write_wins_over_unwritten_fields():
  model, batch, s0 = slider(2)
  ctrl, mocap_pos, state = batch.bind("ctrl"), batch.bind("mocap_pos"), batch.bind("state")
  qvel, xpos = batch.bind("qvel"), batch.bind("xpos")
  ctrl[:] = 1.0
  mocap_pos[:, 0] = [0.3, 0.2, 0.1]
  batch.step()
  state[1] = s0
  batch.step()
  np.testing.assert_array_equal(qvel[:, 0], [0.004, 0.0])
  np.testing.assert_array_equal(ctrl[:, 0], [1.0, 0.0])
  np.testing.assert_array_equal(mocap_pos[1, 0], 0.0)
  np.testing.assert_array_equal(xpos[1, model.body("mocap").id], 0.0)


RAY_XML = """
<mujoco>
  <asset>
    <mesh name="wedge" vertex="0 0 0  .4 0 0  0 .4 0  0 0 .3"/>
    <hfield name="hills" nrow="4" ncol="4" size="1 1 .3 .1"
            elevation="0 .2 .4 .1  .3 1 .6 .2  .1 .5 .9 .3  0 .2 .3 .1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <geom name="hills" type="hfield" hfield="hills" pos="3 0 0"/>
    <geom name="wedge" type="mesh" mesh="wedge" pos="-2 0 0"/>
    <body name="box" pos="0 0 .5">
      <freejoint/>
      <geom name="box" type="box" size=".3 .2 .1"/>
      <geom name="ball" type="sphere" size=".15" pos="0 0 .3" group="1"/>
    </body>
    <body name="target" mocap="true" pos="1 1 1">
      <geom name="target" type="capsule" size=".1 .2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ray_setup(dtype):
  """A batch with each sim's box and mocap body somewhere else, written but not stepped."""
  model = mujoco.MjModel.from_xml_string(RAY_XML)
  batch = Batch(model, N, 3)
  rng = np.random.default_rng(0)
  qpos = batch.bind("qpos")
  qpos[:, :3] += rng.uniform(-0.3, 0.3, (N, 3))
  quat = rng.normal(size=(N, 4))
  qpos[:, 3:] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
  batch.bind("mocap_pos")[:] += rng.uniform(-0.5, 0.5, (N, 1, 3))
  nray = 64
  pnt = rng.uniform(-4, 4, (N, nray, 3)).astype(dtype)
  pnt[..., 2] = rng.uniform(1.5, 3, (N, nray))
  vec = rng.normal(size=(N, nray, 3))
  vec[..., 2] = -np.abs(vec[..., 2]) - 1
  vec = (vec / np.linalg.norm(vec, axis=-1, keepdims=True)).astype(dtype)
  # The first two rays drop straight onto the mocap body, for the exclusion checks.
  pnt[:, :2] = batch.bind("mocap_pos") + [0, 0, 1]
  vec[:, :2] = [0, 0, -1]
  return model, batch, pnt, vec


def _reference_rays(model, batch, pnt, vec, geomgroup, flg_static, bodyexclude):
  data = mujoco.MjData(model)
  dist = np.empty(pnt.shape[:2])
  geomid = np.empty(pnt.shape[:2], np.int32)
  normal = np.zeros(pnt.shape)
  hit, n = np.zeros(1, np.int32), np.zeros(3)
  for i in range(N):
    data.qpos[:] = batch.bind("qpos")[i]
    data.mocap_pos[:] = batch.bind("mocap_pos")[i]
    mujoco.mj_forward(model, data)
    for k in range(pnt.shape[1]):
      exclude = -1 if bodyexclude is None else int(bodyexclude[k])
      dist[i, k] = mujoco.mj_ray(
        model,
        data,
        pnt[i, k].astype(np.float64),
        vec[i, k].astype(np.float64),
        geomgroup,
        flg_static,
        exclude,
        hit,
        n,
      )
      geomid[i, k] = hit[0]
      if dist[i, k] >= 0:
        normal[i, k] = n
  return dist, geomid, normal


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("filtered", [False, True])
def test_rays_match_mj_ray(dtype, filtered):
  model, batch, pnt, vec = _ray_setup(dtype)
  nray = pnt.shape[1]
  geomgroup = np.array([1, 0, 0, 0, 0, 0], np.uint8) if filtered else None
  bodyexclude = None
  if filtered:
    bodyexclude = np.where(np.arange(nray) % 2, model.body("target").id, -1)
    bodyexclude = bodyexclude.astype(np.int32)
  flg_static = not filtered
  dist = np.empty((N, nray), dtype)
  geomid = np.empty((N, nray), np.int32)
  normal = np.empty((N, nray, 3), dtype)
  batch.rays(pnt, vec, dist, geomid, normal, geomgroup, flg_static, bodyexclude)

  want = _reference_rays(model, batch, pnt, vec, geomgroup, flg_static, bodyexclude)
  tol = 1e-12 if dtype == np.float64 else 1e-5
  np.testing.assert_array_equal(geomid, want[1])
  np.testing.assert_allclose(dist, want[0], atol=tol)
  np.testing.assert_allclose(normal, want[2], atol=tol)
  hit = {model.geom(g).name for g in np.unique(geomid) if g >= 0}
  if filtered:
    assert hit == {"box", "target"}  # no static geoms, no group 1 ball
    assert (geomid[:, 0] == model.geom("target").id).all()
    assert (geomid[:, 1::2] != model.geom("target").id).all()
  else:
    assert hit == {"floor", "hills", "wedge", "box", "ball", "target"}


def test_rays_are_a_query():
  model, batch, pnt, vec = _ray_setup(np.float64)
  xpos = batch.bind("xpos")
  before = xpos.copy()
  dist = np.full(pnt.shape[:2], 7.0)
  ids = np.array([1, 4])
  batch.rays(pnt, vec, dist, ids=ids)
  assert (dist[ids] != 7.0).all()
  assert (np.delete(dist, ids, axis=0) == 7.0).all()
  np.testing.assert_array_equal(xpos, before)

  # The pending qpos write the rays just read still reaches the next call.
  batch.forward()
  np.testing.assert_allclose(xpos[:, model.body("box").id], batch.bind("qpos")[:, :3])


def test_rays_see_per_sim_geometry():
  model, batch, _, _ = _ray_setup(np.float64)
  batch.bind("qpos")[:, :7] = model.qpos0[:7]
  half_height = np.linspace(0.05, 0.4, N)
  box = model.geom("box").id
  batch.expand("geom_size")[:, box, 2] = half_height
  pnt = np.tile([0.25, 0.15, 3.0], (N, 1, 1))
  vec = np.tile([0.0, 0.0, -1.0], (N, 1, 1))
  dist = np.empty((N, 1))
  batch.rays(pnt, vec, dist)
  np.testing.assert_allclose(dist[:, 0], 3.0 - (0.5 + half_height))


def test_rays_validation():
  _, batch, pnt, vec = _ray_setup(np.float64)
  dist = np.empty(pnt.shape[:2])
  with pytest.raises(ValueError, match="dist"):
    batch.rays(pnt, vec, dist.astype(np.float32))
  with pytest.raises(ValueError, match="vec"):
    batch.rays(pnt, vec[:, :-1], dist)
  with pytest.raises(ValueError, match="pnt"):
    batch.rays(pnt[:-1], vec[:-1], dist[:-1])
  with pytest.raises(ValueError, match="bodyexclude"):
    batch.rays(pnt, vec, dist, bodyexclude=np.zeros(3, np.int32))


def _jac_setup(dtype):
  """A batch with every sim in another pose, written but not stepped, and one point
  and body per sim."""
  model = mujoco.MjModel.from_xml_string(XML)
  batch = Batch(model, N, 3)
  rng = np.random.default_rng(0)
  batch.bind("qpos")[:] = rng.uniform(-1, 1, (N, model.nq))
  point = rng.uniform(-1, 1, (N, 3)).astype(dtype)
  body = (1 + np.arange(N) % 3).astype(np.int32)  # cart, pole, puck
  return model, batch, point, body


def _reference_jac(model, batch, point, body):
  data = mujoco.MjData(model)
  jacp, jacr = np.empty((N, 3, model.nv)), np.empty((N, 3, model.nv))
  for i in range(N):
    data.qpos[:] = batch.bind("qpos")[i]
    mujoco.mj_forward(model, data)
    mujoco.mj_jac(model, data, jacp[i], jacr[i], point[i].astype(np.float64), body[i])
  return jacp, jacr


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_jac_matches_mj_jac(dtype):
  model, batch, point, body = _jac_setup(dtype)
  jacp = np.empty((N, 3, model.nv), dtype)
  jacr = np.empty((N, 3, model.nv), dtype)
  batch.jac(jacp, jacr, point, body)

  want_p, want_r = _reference_jac(model, batch, point, body)
  tol = 1e-12 if dtype == np.float64 else 1e-6
  np.testing.assert_allclose(jacp, want_p, atol=tol)
  np.testing.assert_allclose(jacr, want_r, atol=tol)
  assert np.abs(want_p).max() > 0.1 and np.abs(want_r).max() > 0.1

  only_p, only_r = np.empty_like(jacp), np.empty_like(jacr)
  batch.jac(only_p, None, point, body)
  batch.jac(None, only_r, point, body)
  np.testing.assert_array_equal(only_p, jacp)
  np.testing.assert_array_equal(only_r, jacr)


def test_jac_is_a_query():
  model, batch, point, body = _jac_setup(np.float64)
  xpos = batch.bind("xpos")
  before = xpos.copy()
  jacp = np.full((N, 3, model.nv), 7.0)
  ids = np.array([1, 4])
  batch.jac(jacp, None, point, body, ids)
  assert (jacp[ids] != 7.0).all()
  assert (np.delete(jacp, ids, axis=0) == 7.0).all()
  np.testing.assert_array_equal(xpos, before)


def test_jac_validation():
  model, batch, point, body = _jac_setup(np.float64)
  jacp = np.empty((N, 3, model.nv))
  with pytest.raises(ValueError, match="jacp"):
    batch.jac(jacp[:, :, :-1], None, point, body)
  with pytest.raises(ValueError, match="jacr"):
    batch.jac(jacp, jacp.astype(np.float32), point, body)
  with pytest.raises(ValueError, match="point"):
    batch.jac(jacp, None, point[:, :2], body)
  with pytest.raises(ValueError, match="num_sims"):
    batch.jac(jacp, None, point, body[:-1])
  with pytest.raises(ValueError, match="nbody"):
    batch.jac(jacp, None, point, np.full(N, model.nbody, np.int32))
