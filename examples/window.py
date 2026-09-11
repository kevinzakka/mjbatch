# SPDX-License-Identifier: Apache-2.0

import time

import glfw
import mujoco
import numpy as np

LINE, CAPSULE, ARROW = mujoco.mjtGeom.mjGEOM_LINE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ARROW
GHOST, CHOSEN = (0.6, 0.6, 0.7, 0.25), (0.62, 0.3, 0.2, 1.0)
CONTACT = {"C": mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, "F": mujoco.mjtVisFlag.mjVIS_CONTACTFORCE}
POINT, FORCE = (0.38, 0.4, 0.28, 1.0), (0.97, 0.94, 0.88, 1.0)


def clock(seconds):
  return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def progress(head, done, elapsed, tail, end=False):
  bar = "━" * round(20 * done) + "─" * (20 - round(20 * done))
  left = clock(elapsed / done - elapsed) if done else "-:--"
  print(f"\r{head} {bar} {clock(elapsed)}, {left} left  {tail}\x1b[K", end="\n" if end else "", flush=True)


def polyline(scn, points, rgba, width, kind=LINE):
  size, pos, mat = np.zeros(3), np.zeros(3), np.eye(3).ravel()
  for a, b in zip(points[:-1], points[1:], strict=True):
    if scn.ngeom == scn.maxgeom:
      return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, kind, size, pos, mat, rgba)
    mujoco.mjv_connector(geom, kind, width, a, b)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR  # casts no shadow
    scn.ngeom += 1


class Window:
  def __init__(self, model, data, camera=None, title="mjbatch"):
    self.model, self.data, self.clock, self.was = model, data, time.perf_counter(), {}
    glfw.ERROR_REPORTING = "raise"  # a headless box fails here, not later inside MjrContext
    glfw.init()
    glfw.window_hint(glfw.SAMPLES, 4)
    self.window = glfw.create_window(1280, 720, title, None, None)
    glfw.make_context_current(self.window)
    glfw.swap_interval(0)  # we pace the frames ourselves
    glfw.poll_events()  # macOS maps the window on its first events; bring it forward
    glfw.focus_window(self.window)
    self.scene = mujoco.MjvScene(model, 4000)
    self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_200)
    self.option, self.perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    self.camera = mujoco.MjvCamera()
    if camera is not None:
      self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
      self.camera.fixedcamid = model.camera(camera).id
    weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
    model.vis.map.force = 0.25 * model.stat.meanmass / (2 * weight)  # m of arrow per body weight
    model.vis.scale.contactwidth, model.vis.scale.contactheight = 0.25, 0.04
    model.vis.scale.forcewidth = 0.05
    model.vis.rgba.contactpoint, model.vis.rgba.contactforce = POINT, FORCE

  def open(self):
    return not glfw.window_should_close(self.window)

  def close(self):
    self.context.free()  # while the GL context is still current
    glfw.terminate()

  def held(self, *keys):
    return any(glfw.get_key(self.window, getattr(glfw, f"KEY_{k}")) == glfw.PRESS for k in keys)

  def pressed(self, key):
    down, was = self.held(key), self.was.get(key, False)
    self.was[key] = down
    return down and not was

  def pace(self, dt):
    time.sleep(max(0.0, dt - (time.perf_counter() - self.clock)))
    self.clock = time.perf_counter()

  def draw(self, traces=(), ghosts=(), legend=None, decor=None):
    for key, flag in CONTACT.items():
      if self.pressed(key):
        self.option.flags[flag] = not self.option.flags[flag]
    opt, pert, scene, cat = self.option, self.perturb, self.scene, mujoco.mjtCatBit
    mujoco.mjv_updateScene(self.model, self.data, opt, pert, self.camera, cat.mjCAT_ALL, scene)
    first = scene.ngeom
    for m, d in ghosts:
      mujoco.mjv_addGeoms(m, d, opt, pert, cat.mjCAT_DYNAMIC, scene)
    for geom in scene.geoms[first : scene.ngeom]:
      geom.category = cat.mjCAT_DECOR
    for trace in traces:
      polyline(scene, *trace)
    if decor is not None:
      decor(scene)
    viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
    mujoco.mjr_render(viewport, scene, self.context)
    if legend is not None:
      font, corner = mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
      mujoco.mjr_overlay(font, corner, viewport, *legend, con=self.context)
    glfw.swap_buffers(self.window)
    glfw.poll_events()
