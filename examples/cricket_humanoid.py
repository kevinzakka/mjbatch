# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "cricket-gym[video] @ git+https://github.com/kishanpb/gym-cricket.git@128fa627c8aa5ebe592d650f0eedcc230c65f794",
#   "mjbatch==0.1.0",
#   "mujoco==3.11.0",
#   "gymnasium==1.3.0",
#   "jax==0.10.0",
#   "jaxlib==0.10.0",
#   "skrl==2.0.0",
#   "numpy==2.4.4",
#   "scipy==1.17.1",
#   "Pillow==12.2.0",
#   "flax==0.12.5",
# ]
# ///
"""Replay handed humanoid cricket checkpoints with native CPU-batched integration.

The integration and MIT-licensed humanoid model live in the pinned cricket-gym
package; Python controllers and bookkeeping remain serial. No speedup is claimed.
"""

import argparse
import json
from contextlib import ExitStack, closing
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import numpy as np
from cricket_gym.rl.ppo_tools import skrl_policy_fn
from humanoid_cricket import bowling_training, motor_training
from humanoid_cricket.bowling_motion import RELEASE_TIME
from humanoid_cricket.bowling_release import CricketDeliveryStrideEnv
from humanoid_cricket.bowling_video import Views
from humanoid_cricket.mjbatch_env import HumanoidBatch, SensorBattingEnv
from humanoid_cricket.motor_video import FPS, HEIGHT, INK, PAPER, WIDTH, CameraViews, card, font, probe
from PIL import Image, ImageDraw

SEEDS = {"batting": [3101, 3102, 3103, 3104], "bowling": [4101, 4102, 4103, 4104]}
SELECTION = "First and fourth fixed seeds per hand and algorithm, including misses."


def pack(image):
  buffer = BytesIO()
  image.save(buffer, format="JPEG", quality=95)
  return buffer.getvalue()


def unpack(data):
  return Image.open(BytesIO(data)).convert("RGB")


def report_info(info):
  result = json.loads(json.dumps(info, default=lambda value: value.tolist()))
  if "humanoid_asset" in result:
    result["humanoid_asset_sha256"] = motor_training.sha256(result["humanoid_asset"])
    result["humanoid_asset"] = "gymnasium:envs/mujoco/assets/humanoid.xml"
  return result


def replay(task, hand, algorithm, video, stack, checkpoints):
  directory = checkpoints / task
  if task == "batting":
    directory /= hand
  checkpoint = directory / algorithm / "agent_final.pickle"
  temporary = stack.enter_context(TemporaryDirectory())
  training = motor_training if task == "batting" else bowling_training
  kwargs = {"handedness": hand} if task == "batting" else {}
  setup = training.make_setup(algorithm, Path(temporary), **kwargs)
  stack.callback(setup.env.close)
  setup.agent.load(str(checkpoint))
  policy = skrl_policy_fn(setup.agent) if task == "batting" else training.policy_fn(setup.agent)
  seeds = SEEDS[task]
  batch = HumanoidBatch(task, len(seeds), handedness=hand, threads=2, profile="wicket")
  stack.callback(batch.close)
  batch.reset(seeds)
  refs = [
    SensorBattingEnv(profile="wicket", handedness=hand)
    if task == "batting"
    else CricketDeliveryStrideEnv(hand)
    for _ in seeds
  ]
  for env, seed in zip(refs, seeds, strict=True):
    env.reset(seed=seed)
    stack.callback(env.close)
  frames, views = {}, {}
  if video:
    for i in (0, 3):
      env = batch.envs[i]
      frames[i] = []
      row = {"profile": "wicket", "seed": seeds[i], "wide_ball": False, "invalid_delivery": False}
      views[i] = stack.enter_context(
        closing(
          CameraViews(env, algorithm, row, "mjbatch | Motor residuals")
          if task == "batting"
          else Views(env, algorithm, seeds[i])
        )
      )
      if task == "bowling":

        def capture(current, time, index=i):
          frames[index].append((time, pack(views[index].frame(current, time))))

        env.on_frame = capture
  results = [None] * len(seeds)
  returns = np.zeros(len(seeds))
  min_elbow = [180.0] * len(seeds)
  for _ in range(200):
    actions = np.stack([np.asarray(policy(env, None)).clip(-1, 1) for env in batch.envs])
    step = batch.step(actions)
    for i, result in enumerate(step):
      if result is None:
        continue
      ref = refs[i].step(actions[i])
      env = batch.envs[i]
      np.testing.assert_array_equal(env.data.qpos, refs[i].data.qpos)
      np.testing.assert_array_equal(env.data.qvel, refs[i].data.qvel)
      np.testing.assert_array_equal(result[0], ref[0])
      assert result[1:4] == ref[1:4]
      returns[i] += result[1]
      if task == "batting":
        assert result[4]["bat_contact"] == ref[4]["bat_contact"]
        assert result[4]["runs"] == ref[4]["runs"]
        min_elbow[i] = min(min_elbow[i], *env.elbow_flexion_degrees().values())
        assert min_elbow[i] > 0 and env.maximum_grip_error < 0.01
        if i in frames:
          views[i].row.update(result[4])
          frames[i].append((env._event.time, pack(views[i].frame(env, result[2] or result[3]))))
      else:
        assert result[4] == ref[4]
        assert not result[4]["delivery_feet"]["foot_no_ball"]
        assert result[4]["release_hand_gap_m"] < 0.001
      results[i] = result
    if batch.done.all():
      break
  assert batch.done.all(), "unfinished evaluation episode"
  rows, clips = [], []
  for i, result in enumerate(results):
    env = batch.envs[i]
    row = {
      "task": task,
      "algorithm": algorithm,
      "handedness": hand,
      "seed": seeds[i],
      "return": float(returns[i]),
      "info": report_info(result[4]),
      "integration_steps": int(batch.integration_steps[i]),
      "serial_sensor_model_parity": True,
      "checkpoint_sha256": training.sha256(checkpoint),
    }
    if task == "batting":
      row.update(
        min_elbow_flexion_degrees=min_elbow[i],
        maximum_grip_error_m=env.maximum_grip_error,
        contact_events=env.contact_events,
      )
    rows.append(row)
    if i in frames:
      center = (
        env._contact_time
        if task == "batting" and env._event.bat_contact
        else (env.strike_time if task == "batting" else RELEASE_TIME)
      )
      clips.append({"row": row, "frames": frames[i], "center": center})
  print(
    f"{task} {hand} {algorithm}: {len(rows)} completed; exact serial parity; {batch.batch_calls} batch calls",
    flush=True,
  )
  return rows, clips


def render(task, clips, output):
  count, clip_records, thumbnails = 0, [], []
  with closing(
    imageio_ffmpeg.write_frames(
      str(output),
      (WIDTH, HEIGHT),
      fps=FPS,
      codec="libx264",
      pix_fmt_out="yuv420p",
      macro_block_size=1,
      output_params=["-crf", "19", "-movflags", "+faststart"],
    )
  ) as writer:
    writer.send(None)

    def emit(image, repeat=1):
      nonlocal count
      for _ in range(repeat):
        writer.send(np.asarray(image))
      count += repeat

    for clip in clips:
      first = count
      stride = 2 if task == "batting" else 4
      for _, encoded in clip["frames"][::stride]:
        emit(unpack(encoded))
      outcome = unpack(clip["frames"][-1][1])
      draw = ImageDraw.Draw(outcome)
      info = clip["row"]["info"]
      text = (
        f"{'CONTACT' if info['bat_contact'] else 'MISS'} | Runs {info['runs']}"
        if task == "batting"
        else f"Target error {info['target_error_m']:.3f} m | {'WICKET' if info['native_wicket_contact'] else 'NO WICKET'}"
      )
      draw.rectangle((18, 88, 760, 128), fill=PAPER)
      draw.text((28, 92), text, font=font(22), fill=INK)
      emit(outcome, FPS)
      for time, encoded in clip["frames"]:
        if clip["center"] - 0.20 <= time <= clip["center"] + 0.12:
          image = unpack(encoded)
          draw = ImageDraw.Draw(image)
          draw.rectangle((18, 88, 760, 128), fill=PAPER)
          draw.text((28, 92), "mjbatch CPU | Motion replay | 0.25x", font=font(22), fill=INK)
          emit(image, 2 if task == "batting" else 1)
      thumb = unpack(min(clip["frames"], key=lambda frame: abs(frame[0] - clip["center"]))[1])
      thumb.thumbnail((640, 360))
      thumbnails.append(thumb)
      clip_records.append({**clip["row"], "first_frame": first, "end_frame": count})
    emit(
      card(
        "Humanoid cricket | Native mjbatch CPU",
        [
          "PPO + A2C checkpoint transfer; 16 fixed episodes per task.",
          "Both hands simulated, not mirrored video; misses retained.",
          "Scripted stance/run-up; learned arm residuals or release controls.",
          "Serial physics parity; no speedup or algorithm-ranking claim.",
        ],
      ),
      FPS * 4,
    )
  sheet = Image.new("RGB", (2560, 720), PAPER)
  for i, thumbnail in enumerate(thumbnails):
    sheet.paste(thumbnail, (i % 4 * 640, i // 4 * 360))
  sheet.save(output.with_suffix(".png"))
  metadata = probe(output)
  assert metadata["frames"] == count
  return {
    "filename": output.name,
    "sha256": motor_training.sha256(output),
    "metadata": metadata,
    "selection": SELECTION,
    "clips": clip_records,
    "visual_review": "pending",
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--checkpoints", type=Path, required=True)
  parser.add_argument("--video", action="store_true")
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  report = {
    "backend": "mjbatch CPU",
    "runtime": {name: version(name) for name in ("mujoco", "mjbatch", "jax", "skrl")},
    "scope": "Checkpoint-transfer smoke, not retraining or a full tournament.",
    "contact_semantics": "RK4 contact sensors, not legacy last-stage contact buffers; fresh scores only.",
    "rows": [],
    "videos": [],
    "replay_sha256": motor_training.sha256(__file__),
    "source_sha256": {
      "humanoid_cricket/" + path.name: motor_training.sha256(path)
      for path in sorted(Path(motor_training.__file__).parent.glob("*.py"))
    },
  }
  for task in ("batting", "bowling"):
    clips = []
    for hand in ("right", "left"):
      for algorithm in ("ppo", "a2c"):
        with ExitStack() as stack:
          rows, group = replay(task, hand, algorithm, args.video, stack, args.checkpoints)
        report["rows"].extend(rows)
        clips.extend(group)
    if args.video:
      report["videos"].append(render(task, clips, args.output / f"gym_cricket_mjbatch_humanoid_{task}.mp4"))
    motor_training.write_json(args.output / "mjbatch_humanoid_validation.json", report)


if __name__ == "__main__":
  main()
