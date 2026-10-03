"""Optional integration tests; install the example dependencies before running."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

cricket = pytest.importorskip("humanoid_cricket.mjbatch_env")


def test_report_uses_asset_identity_not_local_install_path(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
  from cricket_humanoid import report_info

  asset = tmp_path / "humanoid.xml"
  asset.write_text("<mujoco/>")
  info = {"humanoid_asset": str(asset), "position": np.array([1.0, 2.0, 3.0])}
  result = report_info(info)
  assert result["humanoid_asset"] == "gymnasium:envs/mujoco/assets/humanoid.xml"
  assert result["humanoid_asset_sha256"] == hashlib.sha256(b"<mujoco/>").hexdigest()
  assert result["position"] == [1.0, 2.0, 3.0]
  assert info["humanoid_asset"] == str(asset)


@pytest.mark.parametrize("task", ["batting", "bowling"])
@pytest.mark.parametrize("hand", ["right", "left"])
def test_humanoid_batch_matches_serial(task, hand, monkeypatch):
  batch = cricket.HumanoidBatch(task, 2, handedness=hand, threads=2)
  seeds = [17000, 17001]
  batch.reset(seeds)
  refs = [
    cricket.SensorBattingEnv(handedness=hand) if task == "batting" else cricket.CricketDeliveryStrideEnv(hand)
    for _ in seeds
  ]
  for env, seed in zip(refs, seeds, strict=True):
    env.reset(seed=seed)
  actions = np.zeros((2, 7 if task == "batting" else 3))

  def no_serial_step(*args, **kwargs):
    raise AssertionError("batch integration fell back to Python mj_step")

  try:
    for _ in range(200):
      with monkeypatch.context() as context:
        context.setattr(mujoco, "mj_step", no_serial_step)
        results = batch.step(actions)
      for i, result in enumerate(results):
        if result is None:
          continue
        reference = refs[i].step(actions[i])
        np.testing.assert_array_equal(batch.envs[i].data.qpos, refs[i].data.qpos)
        np.testing.assert_array_equal(batch.envs[i].data.qvel, refs[i].data.qvel)
        np.testing.assert_array_equal(result[0], reference[0])
        assert result[1:4] == reference[1:4]
        if task == "batting":
          assert result[4]["bat_contact"] == reference[4]["bat_contact"]
          assert result[4]["runs"] == reference[4]["runs"]
          assert batch.envs[i].maximum_grip_error < 0.01
          assert min(batch.envs[i].elbow_flexion_degrees().values()) > 0
        else:
          assert result[4] == reference[4]
          assert not result[4]["delivery_feet"]["foot_no_ball"]
      if batch.done.all():
        break
    assert batch.done.all()
    assert batch.integration_steps.sum() > batch.batch_calls > 0
  finally:
    batch.close()
    for env in refs:
      env.close()
