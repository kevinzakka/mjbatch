// SPDX-License-Identifier: Apache-2.0

// Persistent thread pool: a blocking parallel-for with sticky slices.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Runs fn(worker, i) for i in [0, n) and blocks until done. Item i belongs to
// the slice of worker i * T / n, which keeps a sim on the same core across
// calls; a worker that finishes its slice claims from the others, so no worker
// waits on the slowest. The calling thread is worker 0. Only one Run may be
// active at a time; fn must not throw.
//
// Between runs a worker spins briefly before it parks, so back-to-back calls
// skip the wake-up; the caller does the same while it waits for the others.
class ThreadPool {
 public:
  explicit ThreadPool(int nthreads) : nthreads_(nthreads), next_(new Counter[nthreads]) {
    for (int t = 1; t < nthreads; ++t) {
      threads_.emplace_back([this, t] { Worker(t); });
    }
  }

  ~ThreadPool() {
    stop_.store(true);
    epoch_.fetch_add(1);
    { std::lock_guard<std::mutex> lock(mu_); }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
  }

  int size() const { return nthreads_; }

  void Run(int n, const std::function<void(int, int)>& fn) {
    fn_ = &fn;
    n_ = n;
    for (int t = 0; t < nthreads_; ++t) next_[t].value.store(Start(t), std::memory_order_relaxed);
    pending_.store(nthreads_ - 1, std::memory_order_relaxed);
    epoch_.fetch_add(1);
    if (sleeping_.load() > 0) {
      {
        std::lock_guard<std::mutex> lock(mu_);
      }
      wake_.notify_all();
    }
    Work(0);
    const auto deadline = Clock::now() + kSpin;
    while (pending_.load(std::memory_order_acquire) != 0) {
      if (Clock::now() < deadline) continue;
      std::unique_lock<std::mutex> lock(mu_);
      done_.wait(lock, [this] { return pending_.load() == 0; });
    }
  }

 private:
  using Clock = std::chrono::steady_clock;
  // Long enough to bridge back-to-back calls, short enough to cost nothing otherwise.
  static constexpr std::chrono::microseconds kSpin{50};
  // One cache line each: every worker advances its own counter on every item.
  struct alignas(64) Counter {
    std::atomic<int> value;
  };

  int Start(int t) const { return static_cast<int>(static_cast<int64_t>(t) * n_ / nthreads_); }

  void Work(int worker) {
    for (int k = 0; k < nthreads_; ++k) {
      const int t = (worker + k) % nthreads_;
      const int end = Start(t + 1);
      std::atomic<int>& next = next_[t].value;
      for (int i = next.fetch_add(1, std::memory_order_relaxed); i < end;
           i = next.fetch_add(1, std::memory_order_relaxed)) {
        (*fn_)(worker, i);
      }
    }
  }

  void Worker(int worker) {
    uint64_t seen = 0;
    while (true) {
      const auto deadline = Clock::now() + kSpin;
      while (epoch_.load(std::memory_order_acquire) == seen) {
        if (Clock::now() < deadline) continue;
        std::unique_lock<std::mutex> lock(mu_);
        sleeping_.fetch_add(1);
        wake_.wait(lock, [this, seen] { return epoch_.load() != seen; });
        sleeping_.fetch_sub(1);
      }
      if (stop_.load()) return;
      seen = epoch_.load(std::memory_order_acquire);
      Work(worker);
      if (pending_.fetch_sub(1) == 1) {
        {
          std::lock_guard<std::mutex> lock(mu_);
        }
        done_.notify_one();
      }
    }
  }

  const int nthreads_;
  std::unique_ptr<Counter[]> next_;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable wake_;
  std::condition_variable done_;
  const std::function<void(int, int)>* fn_ = nullptr;
  int n_ = 0;
  std::atomic<int> pending_{0};   // workers still in the current run
  std::atomic<int> sleeping_{0};  // workers parked on wake_
  std::atomic<uint64_t> epoch_{0};
  std::atomic<bool> stop_{false};
};
