# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mjbatch._bindings import Batch as _Batch

_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}  # by mjtJoint: free, ball, slide, hinge
_DOF_WIDTH = {0: 6, 1: 3, 2: 1, 3: 1}


@dataclass(frozen=True)
class Joint:
  qpos: np.ndarray
  qvel: np.ndarray


@dataclass(frozen=True)
class Actuator:
  ctrl: np.ndarray
  force: np.ndarray


@dataclass(frozen=True)
class Body:
  xpos: np.ndarray
  xquat: np.ndarray
  cvel: np.ndarray


@dataclass(frozen=True)
class Site:
  xpos: np.ndarray
  xmat: np.ndarray


class Batch(_Batch):
  """See mjbatch._bindings.Batch. The named accessors return live (N, ...) views of
  the corresponding bound fields, like MjData's sensor(), joint(), body() and site()."""

  def __init__(
    self,
    model: mujoco.MjModel,
    num_sims: int,
    num_threads: int = 0,
    forward: bool = False,
  ) -> None:
    super().__init__(model, num_sims, num_threads, forward)
    self.model = model

  def sensor(self, name: str, dtype: Any = None) -> np.ndarray:
    s = self.model.sensor(name)
    return self.bind("sensordata", dtype)[:, s.adr[0] : s.adr[0] + s.dim[0]]

  def joint(self, name: str, dtype: Any = None) -> Joint:
    j = self.model.joint(name)
    nq, nv = _QPOS_WIDTH[int(j.type[0])], _DOF_WIDTH[int(j.type[0])]
    return Joint(
      self.bind("qpos", dtype)[:, j.qposadr[0] : j.qposadr[0] + nq],
      self.bind("qvel", dtype)[:, j.dofadr[0] : j.dofadr[0] + nv],
    )

  def actuator(self, name: str, dtype: Any = None) -> Actuator:
    i = self.model.actuator(name).id
    return Actuator(self.bind("ctrl", dtype)[:, i], self.bind("actuator_force", dtype)[:, i])

  def body(self, name: str, dtype: Any = None) -> Body:
    i = self.model.body(name).id
    return Body(
      self.bind("xpos", dtype)[:, i],
      self.bind("xquat", dtype)[:, i],
      self.bind("cvel", dtype)[:, i],
    )

  def site(self, name: str, dtype: Any = None) -> Site:
    i = self.model.site(name).id
    return Site(self.bind("site_xpos", dtype)[:, i], self.bind("site_xmat", dtype)[:, i])
