from typing import Annotated

import numpy
from numpy.typing import NDArray


class Batch:
    """
    N MuJoCo simulations stepped on a C++ thread pool.

    The model is copied at construction; edit it before. Calls are serialized.

    Each simulation is an mjSTATE_INTEGRATION vector plus warning counters, loaded
    into a per-thread mjData for each call, so memory scales with threads, not
    simulations. Models with sleep enabled are rejected: sleep bookkeeping lives
    outside mjtState.

    bind("state") returns the (N, nstate) integration states themselves, in
    mj_getState's mjSTATE_INTEGRATION order: opaque rows for copying, saving and
    restoring simulations, native dtype only. A row written is the simulation's
    state at its next call, with pending writes to copied fields (below) applied on
    top. reset applies it element by element where it changed, like a field write.

    bind(field) returns an (N, ...) array over an mjData field in MjData's layout.
    An mjtNum state component bound in its native dtype (qpos, qvel, ctrl,
    mocap_pos, ...) is a strided view into the state rows, so a write through it and
    a write to the row are one write, the later winning. The other input fields
    (eq_active, warning, and float32 bindings) are copies, copied in before a
    physics call, element by element, where they changed since we last wrote them;
    after a state write they are stale until the next call, and a write of the
    stale value goes unseen. Every bound field is copied out after a call, and a
    derived field bound between calls is filled for a simulation by its next call.
    reset applies pending writes after mj_resetData and then runs mj_forward, so
    derived fields are valid. After step, sensordata and every derived field are
    one substep behind qpos and qvel, as with mj_step itself, unless the batch was
    made with forward=True, which ends every step with mj_forward so they are
    current, at the cost of one forward per simulation per call.

    expand(field) returns (N, ...) per-simulation values of an mjModel field, seeded
    from the model and applied before every physics call for that simulation.
    mjOption fields expand too, a scalar as (N,) and a vector as (N, size), so
    gravity, timestep, integrator and the solver settings are per simulation. Raising
    iterations or ls_iterations, or switching cone to elliptic, can make a simulation
    need more arena than the template sized the worker's mjData for; that surfaces as
    the usual trapped MuJoCo error naming it. enableflags cannot turn sleep on, for
    the reason the constructor rejects it. The first call allocates one model copy
    per thread. set_const runs mj_setConst per simulation and expands every field it
    changed, so derived constants are per simulation too; mjModel scalars it writes
    (flags, stat) are kept per simulation as well.

    Derived constants follow expanded inputs only after set_const, as with
    mj_setConst on one model. A MuJoCo error on a worker raises RuntimeError naming
    the first failing simulation; the others still ran, and the failing one keeps the
    state it had before the call, its writes still pending. The trap is a MuJoCo log
    handler installed at import; installing another handler later disables it.

    rays and jac are queries: mj_ray and mj_jac for each simulation against its own
    geometry, which is its state with pending writes on top and its expanded model
    fields. A query runs only the part of the pipeline it needs, consumes no pending
    write, and updates no bound field.
    """

    def __init__(self, model: object, num_sims: int, num_threads: int = 0, forward: bool = False) -> None:
        """
        num_threads=0 uses every logical CPU, clamped to num_sims. forward=True ends every step with mj_forward, so derived fields are current with the state.
        """

    @property
    def num_sims(self) -> int: ...

    @property
    def num_threads(self) -> int: ...

    @property
    def nstate(self) -> int:
        """The length of a simulation's integration state."""

    def bind(self, name: str, dtype: object | None = None) -> NDArray:
        """dtype is the field's own or float32 for mjtNum fields."""

    def expand(self, name: str, dtype: object | None = None) -> NDArray: ...

    def step(self, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None, nstep: int = 1, history: NDArray | None = None) -> None:
        """
        nstep mj_step calls per simulation, on one worker. ids: sorted unique ints or a bool mask. history: an optional caller-allocated (sims, nstep, nstate) array filled with each selected simulation's state after every substep, in bind("state") order; the rows of a simulation that raises are undefined.
        """

    def forward(self, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None) -> None: ...

    def reset(self, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None, keyframe: int = -1) -> None:
        """
        mj_resetData, or mj_resetDataKeyframe when keyframe >= 0, then mj_forward.
        """

    def set_const(self, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None) -> None: ...

    def rays(self, pnt: NDArray, vec: NDArray, dist: NDArray, geomid: NDArray | None = None, normal: NDArray | None = None, geomgroup: Annotated[NDArray[numpy.uint8], dict(shape=(6), order='C', writable=False)] | None = None, flg_static: bool = True, bodyexclude: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C', writable=False)] | None = None, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None) -> None:
        """
        mj_ray for every ray of every simulation. pnt and vec are (num_sims, nray, 3) origins and directions in the world frame, float32 or the native dtype. dist (num_sims, nray), and the optional geomid (int32) and normal (num_sims, nray, 3), are caller-allocated and filled in place: dist is -1 and normal zero where a ray hits nothing. geomgroup is mj_ray's six-entry uint8 mask, bodyexclude one int32 body id per ray (-1 for none), and ids restricts which rows are computed.
        """

    def jac(self, jacp: NDArray | None, jacr: NDArray | None, point: NDArray, body: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C', writable=False)], ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None) -> None:
        """
        mj_jac for one point per simulation. point is (num_sims, 3) in the world frame, float32 or the native dtype, and body (num_sims,) int32 is the body it moves with. jacp and jacr are caller-allocated (num_sims, 3, nv) arrays filled in place with the translational and rotational Jacobians; either may be None. ids restricts which rows are computed.
        """
