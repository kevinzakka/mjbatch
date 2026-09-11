// SPDX-License-Identifier: Apache-2.0

// Batch: N MuJoCo simulations on a thread pool, with batch buffers for data
// fields and per-sim storage for model fields. Each sim keeps only its
// mjSTATE_INTEGRATION vector and warning counters; the mjData is per worker.
#pragma once

#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "threadpool.h"

namespace nb = nanobind;

// Every logical CPU: MuJoCo's small dense linear algebra stalls on latency, so a
// core runs two threads at 1.3-1.5x the throughput of one (measured on a 7960X).
inline int DefaultThreadCount() {
  return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

enum class Elem { Num, Float, Int, Byte, Bool, Other };

template <class T>
constexpr Elem ElemOf() {
  if constexpr (std::is_same_v<T, mjtNum>) return Elem::Num;
  if constexpr (std::is_same_v<T, float>) return Elem::Float;
  if constexpr (std::is_same_v<T, int>) return Elem::Int;
  if constexpr (std::is_same_v<T, mjtByte>) return Elem::Byte;
  if constexpr (std::is_same_v<T, mjtBool>) return Elem::Bool;
  return Elem::Other;
}

inline const char* DtypeName(Elem e) {
  switch (e) {
    case Elem::Num:
      return sizeof(mjtNum) == 8 ? "float64" : "float32";
    case Elem::Float:
      return "float32";
    case Elem::Int:
      return "int32";
    case Elem::Byte:
      return "uint8";
    case Elem::Bool:
      return "bool";
    default:
      return "";
  }
}

// One array field of mjModel or mjData, sized for a specific model.
struct FieldInfo {
  const char* name;
  void* (*get)(void* obj);
  int nr, nc;
  int ndim;  // per sim: 0 for scalars, 1 when nc is the literal 1, else 2.
  size_t elem_size;
  Elem elem;
  bool input;  // mjData: persists across calls (an mjtState component or warning); else derived.
  bool asset;  // mjModel: large constant data mj_setConst never writes.
  size_t bytes() const { return static_cast<size_t>(nr) * nc * elem_size; }
};

using FieldTable = std::unordered_map<std::string, FieldInfo>;

inline bool IsInput(std::string_view name) {
  for (std::string_view s : {"time", "qpos", "qvel", "act", "history", "qacc_warmstart", "ctrl",
                             "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
                             "userdata", "plugin_state", "warning"}) {
    if (name == s) return true;
  }
  return false;
}

inline bool IsAsset(std::string_view name) {
  for (std::string_view p : {"mesh_", "hfield_", "tex_", "skin_", "bvh_", "oct_"}) {
    if (name.substr(0, p.size()) == p) return true;
  }
  return false;
}

inline FieldTable DataFields(const mjModel* m) {
  FieldTable t;
#undef MJ_M
#define MJ_M(n) (m->n)
#define X(type, name, nr, nc)                                                          \
  t[#name] = FieldInfo{#name,                                                          \
                       [](void* o) -> void* { return static_cast<mjData*>(o)->name; }, \
                       static_cast<int>(MJ_M(nr)),                                     \
                       static_cast<int>(nc),                                           \
                       std::string_view(#nc) == "1" ? 1 : 2,                           \
                       sizeof(type),                                                   \
                       ElemOf<type>(),                                                 \
                       IsInput(#name),                                                 \
                       false};
  MJDATA_POINTERS
#undef X
#undef MJ_M
#define MJ_M(n) n
  t["time"] = FieldInfo{"time",    [](void* o) -> void* { return &static_cast<mjData*>(o)->time; },
                        1,         1,
                        0,         sizeof(mjtNum),
                        Elem::Num, true,
                        false};
  // mjWarningStat is two ints: (lastinfo, number) per warning type.
  t["warning"] =
      FieldInfo{"warning",  [](void* o) -> void* { return static_cast<mjData*>(o)->warning; },
                mjNWARNING, 2,
                2,          sizeof(int),
                Elem::Int,  true,
                false};
  return t;
}

inline FieldTable ModelFields(const mjModel* m) {
  FieldTable t;
#undef MJ_M
#define MJ_M(n) (m->n)
#define X(type, name, nr, nc)                                                           \
  t[#name] = FieldInfo{#name,                                                           \
                       [](void* o) -> void* { return static_cast<mjModel*>(o)->name; }, \
                       static_cast<int>(MJ_M(nr)),                                      \
                       static_cast<int>(nc),                                            \
                       std::string_view(#nc) == "1" ? 1 : 2,                            \
                       sizeof(type),                                                    \
                       ElemOf<type>(),                                                  \
                       false,                                                           \
                       IsAsset(#name)};
  MJMODEL_POINTERS
#undef X
#undef MJ_M
#define MJ_M(n) n
  // mjOption alongside the arrays, its scalars shaped like mjData's time. The two
  // namespaces are disjoint today; a collision would silently shadow an array field.
#define XOPT(type, name, nr, ndim, getter)                                                         \
  if (!t.emplace(#name, FieldInfo{#name, getter, nr, 1, ndim, sizeof(type), ElemOf<type>(), false, \
                                  false})                                                          \
           .second) {                                                                              \
    throw std::runtime_error("mjOption." #name " collides with an mjModel field");                 \
  }
#define X(type, name, size) \
  XOPT(type, name, 1, 0, [](void* o) -> void* { return &static_cast<mjModel*>(o)->opt.name; })
#define XVEC(type, name, size) \
  XOPT(type, name, size, 1, [](void* o) -> void* { return static_cast<mjModel*>(o)->opt.name; })
  MJOPTION_FIELDS
#undef XVEC
#undef X
#undef XOPT
  return t;
}

// A batch buffer over one field: (N, nr[, nc]) rows, native dtype or float32.
// Against a float32 libmujoco the float32 path is a plain memcpy.
struct Slot {
  const FieldInfo* info;
  bool f32;
  size_t row;  // bytes per sim
  std::unique_ptr<uint8_t[]> buf;
  std::unique_ptr<uint8_t[]> mirror;  // buf as last written by us; input data fields only
};

inline void ToBuf(uint8_t* dst, const void* src, const Slot& s) {
  size_t n = static_cast<size_t>(s.info->nr) * s.info->nc;
  if (s.f32 && sizeof(mjtNum) != sizeof(float)) {
    auto* d = reinterpret_cast<float*>(dst);
    auto* x = static_cast<const mjtNum*>(src);
    for (size_t k = 0; k < n; ++k) d[k] = static_cast<float>(x[k]);
  } else {
    std::memcpy(dst, src, s.info->bytes());
  }
}

inline void FromBuf(void* dst, const uint8_t* src, const Slot& s) {
  size_t n = static_cast<size_t>(s.info->nr) * s.info->nc;
  if (s.f32 && sizeof(mjtNum) != sizeof(float)) {
    auto* d = static_cast<mjtNum*>(dst);
    auto* x = reinterpret_cast<const float*>(src);
    for (size_t k = 0; k < n; ++k) d[k] = static_cast<mjtNum>(x[k]);
  } else {
    std::memcpy(dst, src, s.info->bytes());
  }
}

// Copy the elements of buf that differ from mirror into dst, so only what the
// caller wrote lands and the rest of the row keeps whatever physics left there.
inline void CopyChanged(void* dst, const uint8_t* buf, const uint8_t* mirror, const Slot& s) {
  size_t n = static_cast<size_t>(s.info->nr) * s.info->nc;
  size_t e = s.f32 ? sizeof(float) : s.info->elem_size;
  for (size_t k = 0; k < n; ++k) {
    if (std::memcmp(buf + k * e, mirror + k * e, e) == 0) continue;
    if (s.f32 && sizeof(mjtNum) != sizeof(float)) {
      static_cast<mjtNum*>(dst)[k] = reinterpret_cast<const float*>(buf)[k];
    } else {
      std::memcpy(static_cast<uint8_t*>(dst) + k * e, buf + k * e, e);
    }
  }
}

// mjModel scalars mj_setConst writes; kept per sim like expanded fields.
struct Scalars {
  mjtSize ngravcomp;
  mjtBool flg_gravcomp, flg_surfacevel, flg_adhesion;
  mjStatistic stat;
  static Scalars Of(const mjModel* m) {
    return {m->ngravcomp, m->flg_gravcomp, m->flg_surfacevel, m->flg_adhesion, m->stat};
  }
  void Apply(mjModel* m) const {
    m->ngravcomp = ngravcomp;
    m->flg_gravcomp = flg_gravcomp;
    m->flg_surfacevel = flg_surfacevel;
    m->flg_adhesion = flg_adhesion;
    m->stat = stat;
  }
};

// mju_error trap for worker threads: record the message and unwind to the
// worker's setjmp; everything else goes to the handler that was active before.
inline thread_local std::jmp_buf* tls_jmp = nullptr;
inline thread_local std::string tls_error;
inline mjfLogHandler prev_log_handler = nullptr;
inline void LogTrap(const mjLogMessage* msg) {
  if (msg->level == mjLOG_ERROR && tls_jmp) {
    tls_error = msg->subject;
    std::longjmp(*tls_jmp, 1);
  }
  prev_log_handler(msg);
}
inline void InstallLogTrap() { prev_log_handler = mju_setLogHandler(LogTrap); }

class Batch {
 public:
  enum class Op { Step, Forward, Reset, SetConst };
  using Ids = nb::ndarray<nb::ndim<1>, nb::c_contig>;

  Batch(nb::object model, int num_sims, int num_threads, bool forward)
      : num_sims_(num_sims), forward_(forward) {
    if (num_sims < 1) throw nb::value_error("num_sims must be >= 1");
    auto addr = nb::cast<uintptr_t>(model.attr("_address"));
    template_ = mj_copyModel(nullptr, reinterpret_cast<const mjModel*>(addr));
    if (template_->opt.enableflags & mjENBL_SLEEP) throw nb::value_error("sleep is not supported");
    data_fields_ = DataFields(template_);
    model_fields_ = ModelFields(template_);
    for (auto& [name, f] : model_fields_) {
      if (!f.asset) restorable_.push_back(&f);
    }
    scalars_.assign(num_sims, Scalars::Of(template_));
    if (num_threads <= 0) num_threads = DefaultThreadCount();
    pool_ = std::make_unique<ThreadPool>(std::max(1, std::min(num_threads, num_sims)));
    for (int t = 0; t < pool_->size(); ++t) data_.push_back(mj_makeData(template_));
    nstate_ = mj_stateSize(template_, mjSTATE_INTEGRATION);
    states_.resize(static_cast<size_t>(num_sims) * nstate_);
    warnings_.resize(static_cast<size_t>(num_sims) * mjNWARNING);
    for (int i = 0; i < num_sims; ++i) {
      mj_getState(template_, data_[0], State(i), mjSTATE_INTEGRATION);
    }
  }

  ~Batch() {
    pool_.reset();
    for (mjData* d : data_) mj_deleteData(d);
    for (mjModel* m : models_) mj_deleteModel(m);
    mj_deleteModel(template_);
  }
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  int num_sims() const { return num_sims_; }
  int num_threads() const { return pool_->size(); }
  int nstate() const { return nstate_; }

  nb::ndarray<nb::numpy> bind(const std::string& name, std::optional<nb::object> dtype) {
    std::lock_guard<std::mutex> lock(mu_);
    if (name == "state") {
      // The rows are the per-sim mjSTATE_INTEGRATION vectors themselves, so a write to
      // one is authoritative at the sim's next call; field writes land on top of it.
      if (dtype && !dtype->is_none()) throw nb::value_error("state cannot be float32");
      size_t shape[2] = {static_cast<size_t>(num_sims_), static_cast<size_t>(nstate_)};
      return nb::ndarray<nb::numpy>(states_.data(), 2, shape, nb::find(this), nullptr,
                                    nb::dtype<mjtNum>());
    }
    const FieldInfo& f = Field(data_fields_, name);
    bool f32 = ParseDtype(f, dtype);
    for (auto& s : bound_) {
      if (s->info == &f) return Matching(*s, f32);
    }
    bound_.push_back(MakeSlot(f, f32, f.input));
    // A derived field stays zero until the next call fills it.
    if (f.input) {
      for (int i = 0; i < num_sims_; ++i) {
        mj_setState(template_, data_[0], State(i), mjSTATE_INTEGRATION);
        std::memcpy(data_[0]->warning, Warning(i), sizeof(data_[0]->warning));
        CopyOut(*bound_.back(), data_[0], i);
      }
    }
    return View(*bound_.back());
  }

  nb::ndarray<nb::numpy> expand(const std::string& name, std::optional<nb::object> dtype) {
    std::lock_guard<std::mutex> lock(mu_);
    const FieldInfo& f = Field(model_fields_, name);
    bool f32 = ParseDtype(f, dtype);
    for (auto& s : expanded_) {
      if (s->info == &f) return Matching(*s, f32);
    }
    return View(Expand(f, f32));
  }

  void step(std::optional<Ids> ids, int nstep, std::optional<nb::ndarray<>> history) {
    if (nstep < 1) throw nb::value_error("nstep must be >= 1");
    auto sel = Parse(ids);
    mjtNum* hist = nullptr;
    if (history) {
      hist = HistoryPtr(*history, sel ? static_cast<int>(sel->size()) : num_sims_, nstep);
    }
    Run(Op::Step, std::move(sel), nstep, hist);
  }
  void forward(std::optional<Ids> ids) { Run(Op::Forward, Parse(ids), 0); }

  void reset(std::optional<Ids> ids, int keyframe) {
    if (keyframe < -1 || keyframe >= template_->nkey) {
      throw nb::value_error("keyframe out of range");
    }
    Run(Op::Reset, Parse(ids), keyframe);
  }

  // mj_setConst per sim from its expanded fields. Every field it changes
  // becomes expanded, so afterwards each sim steps with a complete set of its
  // own derived constants. A newly expanded field is computed for every sim.
  void set_const(std::optional<Ids> ids) {
    auto sel = Parse(ids);
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    if (models_.empty()) return;
    error_.clear();
    while (true) {
      RunLocked(Op::SetConst, sel, 0);
      if (changed_.empty()) break;
      for (const FieldInfo* f : changed_) Expand(*f, false);
      changed_.clear();
      sel.reset();
    }
    if (!error_.empty()) throw std::runtime_error(error_);
  }

 private:
  const FieldInfo& Field(const FieldTable& table, const std::string& name) {
    auto it = table.find(name);
    if (it == table.end()) {
      std::string msg = "unknown field " + name;
      if (&table == &data_fields_ && model_fields_.count(name)) {
        msg = name + " is an mjModel field; use expand(\"" + name + "\")";
      } else if (&table == &model_fields_ && data_fields_.count(name)) {
        msg = name + " is an mjData field; use bind(\"" + name + "\")";
      }
      throw nb::value_error(msg.c_str());
    }
    if (it->second.elem == Elem::Other) {
      throw nb::value_error(("unsupported element type in " + name).c_str());
    }
    return it->second;
  }

  // True for float32 over an mjtNum field; the native dtype is always allowed.
  bool ParseDtype(const FieldInfo& f, std::optional<nb::object>& dtype) {
    if (!dtype || dtype->is_none()) return false;
    nb::object np = nb::module_::import_("numpy");
    std::string name = nb::cast<std::string>(nb::str(np.attr("dtype")(*dtype).attr("name")));
    if (name == DtypeName(f.elem)) return false;
    if (name == "float32" && f.elem == Elem::Num) return true;
    throw nb::value_error((std::string(f.name) + " cannot be " + name).c_str());
  }

  nb::ndarray<nb::numpy> Matching(Slot& s, bool f32) {
    if (s.f32 != f32)
      throw nb::value_error(
          (std::string(s.info->name) + " is already bound with another dtype").c_str());
    return View(s);
  }

  std::unique_ptr<Slot> MakeSlot(const FieldInfo& f, bool f32, bool mirror) {
    auto s = std::make_unique<Slot>();
    s->info = &f;
    s->f32 = f32;
    s->row = static_cast<size_t>(f.nr) * f.nc * (f32 ? sizeof(float) : f.elem_size);
    s->buf = std::make_unique<uint8_t[]>(s->row * num_sims_);
    if (mirror) s->mirror = std::make_unique<uint8_t[]>(s->row * num_sims_);
    return s;
  }

  Slot& Expand(const FieldInfo& f, bool f32) {
    if (f.asset) throw nb::value_error((std::string(f.name) + " is asset data").c_str());
    auto s = MakeSlot(f, f32, false);
    for (int i = 0; i < num_sims_; ++i) ToBuf(s->buf.get() + i * s->row, f.get(template_), *s);
    expanded_.push_back(std::move(s));
    expanded_set_.insert(&f);
    if (std::string_view(f.name) == "enableflags") enableflags_ = expanded_.back().get();
    if (models_.empty()) {
      for (int t = 0; t < pool_->size(); ++t) models_.push_back(mj_copyModel(nullptr, template_));
    }
    return *expanded_.back();
  }

  mjtNum* HistoryPtr(const nb::ndarray<>& a, int nsel, int nstep) {
    if (a.ndim() != 3 || static_cast<int>(a.shape(0)) != nsel ||
        static_cast<int>(a.shape(1)) != nstep || static_cast<int>(a.shape(2)) != nstate_) {
      throw nb::value_error("history must have shape (sims, nstep, nstate)");
    }
    if (a.dtype() != nb::dtype<mjtNum>()) {
      throw nb::value_error((std::string("history must be ") + DtypeName(Elem::Num)).c_str());
    }
    // A size-0 array (an empty selection) is never written, so its strides are
    // irrelevant; numpy sets them all to zero regardless of contiguity.
    if (a.device_type() != nb::device::cpu::value ||
        (a.size() != 0 && (a.stride(2) != 1 || a.stride(1) != nstate_ ||
                           a.stride(0) != static_cast<int64_t>(nstep) * nstate_))) {
      throw nb::value_error("history must be a C-contiguous CPU array");
    }
    return static_cast<mjtNum*>(a.data());
  }

  nb::ndarray<nb::numpy> View(const Slot& s) {
    const FieldInfo& f = *s.info;
    size_t shape[3] = {static_cast<size_t>(num_sims_), static_cast<size_t>(f.nr),
                       static_cast<size_t>(f.nc)};
    nb::dlpack::dtype dt;
    if (s.f32 || f.elem == Elem::Float) {
      dt = nb::dtype<float>();
    } else if (f.elem == Elem::Num) {
      dt = nb::dtype<mjtNum>();
    } else if (f.elem == Elem::Int) {
      dt = nb::dtype<int>();
    } else if (f.elem == Elem::Bool) {
      dt = nb::dtype<bool>();
    } else {
      dt = nb::dtype<uint8_t>();
    }
    return nb::ndarray<nb::numpy>(s.buf.get(), f.ndim + 1, shape, nb::find(this), nullptr, dt);
  }

  void CopyOut(Slot& s, mjData* d, int i) {
    uint8_t* row = s.buf.get() + i * s.row;
    ToBuf(row, s.info->get(d), s);
    if (s.mirror) std::memcpy(s.mirror.get() + i * s.row, row, s.row);
  }

  void Restore(mjModel* m) {
    for (const FieldInfo* f : restorable_) std::memcpy(f->get(m), f->get(template_), f->bytes());
    m->npolygonmax = template_->npolygonmax;
    m->nmeshdegmax = template_->nmeshdegmax;
    Scalars::Of(template_).Apply(m);
  }

  mjtNum* State(int i) { return states_.data() + static_cast<size_t>(i) * nstate_; }
  mjWarningStat* Warning(int i) { return warnings_.data() + static_cast<size_t>(i) * mjNWARNING; }

  void RunSim(int t, int i, Op op, int arg, mjtNum* hist) {
    mjModel* m = models_.empty() ? template_ : models_[t];
    mjData* d = data_[t];
    if (op == Op::SetConst) Restore(m);
    for (auto& s : expanded_) FromBuf(s->info->get(m), s->buf.get() + i * s->row, *s);
    if (op == Op::SetConst) {
      mj_setConst(m, d);
      for (auto& s : expanded_) ToBuf(s->buf.get() + i * s->row, s->info->get(m), *s);
      scalars_[i] = Scalars::Of(m);
      for (const FieldInfo* f : restorable_) {
        if (!expanded_set_.count(f) && std::memcmp(f->get(m), f->get(template_), f->bytes())) {
          std::lock_guard<std::mutex> lock(changed_mu_);
          changed_.insert(f);
        }
      }
      Restore(m);
      return;
    }
    if (!models_.empty()) scalars_[i].Apply(m);
    if (op == Op::Reset) {
      if (arg >= 0) {
        mj_resetDataKeyframe(m, d, arg);
      } else {
        mj_resetData(m, d);
      }
    } else {
      mj_setState(m, d, State(i), mjSTATE_INTEGRATION);
      std::memcpy(d->warning, Warning(i), sizeof(d->warning));
    }
    for (auto& s : bound_) {
      if (!s->mirror) continue;
      const uint8_t* row = s->buf.get() + i * s->row;
      if (std::memcmp(row, s->mirror.get() + i * s->row, s->row) != 0) {
        CopyChanged(s->info->get(d), row, s->mirror.get() + i * s->row, *s);
      }
    }
    switch (op) {
      case Op::Step:
        for (int k = 0; k < arg; ++k) {
          mj_step(m, d);
          if (hist) mj_getState(m, d, hist + static_cast<size_t>(k) * nstate_, mjSTATE_INTEGRATION);
        }
        if (forward_) mj_forward(m, d);  // derived fields current with the new state
        break;
      case Op::Forward:
      case Op::Reset:
        mj_forward(m, d);
        break;
      case Op::SetConst:
        break;
    }
    mj_getState(m, d, State(i), mjSTATE_INTEGRATION);
    std::memcpy(Warning(i), d->warning, sizeof(d->warning));
    for (auto& s : bound_) CopyOut(*s, d, i);
  }

  std::optional<std::vector<int>> Parse(const std::optional<Ids>& ids) {
    if (!ids) return std::nullopt;
    if (ids->dtype() == nb::dtype<bool>()) {
      if (static_cast<int>(ids->shape(0)) != num_sims_) {
        throw nb::value_error("a boolean ids mask must have num_sims entries");
      }
      std::vector<int> out;
      for (int i = 0; i < num_sims_; ++i) {
        if (static_cast<const bool*>(ids->data())[i]) out.push_back(i);
      }
      return out;
    }
    std::vector<int> out(ids->shape(0));
    for (size_t j = 0; j < out.size(); ++j) {
      if (ids->dtype() == nb::dtype<int64_t>()) {
        out[j] = static_cast<int>(static_cast<const int64_t*>(ids->data())[j]);
      } else if (ids->dtype() == nb::dtype<int32_t>()) {
        out[j] = static_cast<const int32_t*>(ids->data())[j];
      } else {
        throw nb::value_error("ids must be int32, int64 or a bool mask");
      }
      if (out[j] < 0 || out[j] >= num_sims_ || (j > 0 && out[j] <= out[j - 1])) {
        throw nb::value_error("ids must be sorted, unique and in range");
      }
    }
    return out;
  }

  void Guarded(int t, int i, Op op, int arg, mjtNum* hist) {
    std::jmp_buf jb;
    tls_jmp = &jb;
    if (setjmp(jb) == 0) {
      RunSim(t, i, op, arg, hist);
    } else {
      // The sim's state was not written back; the worker's mjData, left
      // mid-call with its stack and arena in use, serves other sims next.
      tls_jmp = nullptr;
      if (op == Op::SetConst) Restore(models_[t]);
      mj_resetData(template_, data_[t]);
      std::lock_guard<std::mutex> lock(changed_mu_);
      if (error_.empty()) error_ = "sim " + std::to_string(i) + ": " + tls_error;
    }
    tls_jmp = nullptr;
  }

  void Run(Op op, std::optional<std::vector<int>> sel, int arg, mjtNum* hist = nullptr) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    error_.clear();
    RunLocked(op, sel, arg, hist);
    if (!error_.empty()) throw std::runtime_error(error_);
  }

  void RunLocked(Op op, const std::optional<std::vector<int>>& sel, int arg,
                 mjtNum* hist = nullptr) {
    const int* p = sel ? sel->data() : nullptr;
    const int n = sel ? static_cast<int>(sel->size()) : num_sims_;
    // A per-sim enableflags must not switch on what the constructor refused. Raised
    // here rather than on a worker, where the setjmp path only yields a RuntimeError.
    if (enableflags_) {
      for (int j = 0; j < n; ++j) {
        int i = p ? p[j] : j;
        int flags = *reinterpret_cast<const int*>(enableflags_->buf.get() + i * enableflags_->row);
        if (flags & mjENBL_SLEEP) {
          throw nb::value_error(("sim " + std::to_string(i) + ": sleep is not supported").c_str());
        }
      }
    }
    auto fn = [this, op, arg, p, hist](int t, int j) {
      Guarded(t, p ? p[j] : j, op, arg,
              hist ? hist + static_cast<size_t>(j) * arg * nstate_ : nullptr);
    };
    if (pool_->size() == 1) {
      for (int j = 0; j < n; ++j) fn(0, j);
    } else {
      pool_->Run(n, fn);
    }
  }

  int num_sims_;
  bool forward_;  // step ends with mj_forward
  mjModel* template_ = nullptr;
  FieldTable data_fields_;
  FieldTable model_fields_;
  std::vector<const FieldInfo*> restorable_;
  std::vector<mjData*> data_;     // per worker
  std::vector<mjModel*> models_;  // per worker, once anything is expanded
  int nstate_;
  std::vector<mjtNum> states_;           // per sim, mjSTATE_INTEGRATION
  std::vector<mjWarningStat> warnings_;  // per sim
  std::vector<Scalars> scalars_;         // per sim
  std::vector<std::unique_ptr<Slot>> bound_;
  std::vector<std::unique_ptr<Slot>> expanded_;
  std::set<const FieldInfo*> expanded_set_;
  const Slot* enableflags_ = nullptr;   // scanned for mjENBL_SLEEP before every call
  std::set<const FieldInfo*> changed_;  // set_const outputs not yet expanded
  std::unique_ptr<ThreadPool> pool_;
  std::mutex mu_;          // serializes calls; held with the GIL released in Run
  std::mutex changed_mu_;  // changed_ and error_ from workers
  std::string error_;
};
