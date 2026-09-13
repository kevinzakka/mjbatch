# SPDX-License-Identifier: Apache-2.0
# A build pairs with one MuJoCo release: the build environment gets
# mujoco==$MUJOCO_VERSION, the runtime dependency pins it, and a wheel built
# from the checkout is versioned <version>.postMmmpp for MuJoCo M.mm.pp. The
# sdist, and a wheel built from it, keep the bare version.

import os
from pathlib import Path

DEFAULT_MUJOCO = "3.13.0"

_sdist = False


def build_state(state):
  global _sdist
  _sdist = state == "sdist"


def get_requires_for_dynamic_metadata(settings):
  return [f"mujoco=={os.environ.get('MUJOCO_VERSION', DEFAULT_MUJOCO)}"]


def dynamic_wheel(settings):
  return {"dependencies": True}


def dynamic_metadata(settings, project):
  import mujoco

  version = settings["version"]
  if not _sdist and not (Path(__file__).parent / "PKG-INFO").exists():
    version += ".post{}{:02d}{:02d}".format(*map(int, mujoco.__version__.split(".")[:3]))
  return {"version": version, "dependencies": [f"mujoco=={mujoco.__version__}"]}
