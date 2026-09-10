// SPDX-License-Identifier: Apache-2.0

// Persistent thread pool: a blocking parallel-for with sticky slices.
#pragma once

#include <atomic>
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
// waits on the slowest. Only one Run may be active at a time; fn must not throw.
class ThreadPool {
 public:
  explicit ThreadPool(int nthreads) : nthreads_(nthreads), next_(new std::atomic<int>[nthreads]) {
    for (int t = 0; t < nthreads; ++t) {
      threads_.emplace_back([this, t] { Worker(t); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
  }

  int size() const { return nthreads_; }

  void Run(int n, std::function<void(int, int)> fn) {
    std::unique_lock<std::mutex> lock(mu_);
    fn_ = std::move(fn);
    n_ = n;
    for (int t = 0; t < nthreads_; ++t) next_[t].store(Start(t), std::memory_order_relaxed);
    active_ = nthreads_;
    ++epoch_;
    wake_.notify_all();
    done_.wait(lock, [this] { return active_ == 0; });
    fn_ = nullptr;
  }

 private:
  int Start(int t) const { return static_cast<int>(static_cast<int64_t>(t) * n_ / nthreads_); }

  void Worker(int worker) {
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mu_);
    while (true) {
      wake_.wait(lock, [this, &seen] { return stop_ || epoch_ != seen; });
      if (stop_) return;
      seen = epoch_;
      const std::function<void(int, int)>* fn = &fn_;
      lock.unlock();
      for (int k = 0; k < nthreads_; ++k) {
        const int t = (worker + k) % nthreads_;
        const int end = Start(t + 1);
        for (int i = next_[t].fetch_add(1, std::memory_order_relaxed); i < end;
             i = next_[t].fetch_add(1, std::memory_order_relaxed)) {
          (*fn)(worker, i);
        }
      }
      lock.lock();
      if (--active_ == 0) done_.notify_one();
    }
  }

  const int nthreads_;
  std::unique_ptr<std::atomic<int>[]> next_;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::function<void(int, int)> fn_;
  int n_ = 0;
  int active_ = 0;
  uint64_t epoch_ = 0;
  bool stop_ = false;
};
