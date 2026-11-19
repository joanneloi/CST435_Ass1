// ==============================================================
// CST435 METHOD 3: POSIX THREADS (Pthreads) K-MEANS
// ==============================================================
// Different from OpenMP, here we explicitly create the workers,
// divide point ranges, synchronize at two barriers per iteration,
// and join the workers after convergence.
//
// On Windows, MSYS2 UCRT64 provides the winpthreads implementation.

#include "kmeans.hpp"

#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
struct SharedState {
    const Dataset& data;
    const Options& settings;
    RunResult& result;
    const int P;
    std::mutex barrier_mutex;
    std::condition_variable barrier_condition;
    int barrier_waiting = 0;
    std::uint64_t barrier_generation = 0;

    std::vector<double> worker_sums;
    std::vector<std::uint64_t> worker_counts;
    std::vector<double> all_sums;
    std::vector<std::uint64_t> all_counts;
    std::atomic<bool> done{false};

    void barrier_wait() {
        std::unique_lock<std::mutex> lock(barrier_mutex);
        const auto generation = barrier_generation;
        if (++barrier_waiting == P) {
            barrier_waiting = 0;
            ++barrier_generation;
            lock.unlock();
            barrier_condition.notify_all();
            return;
        }
        barrier_condition.wait(lock, [this, generation] {
            return barrier_generation != generation;
        });
    }

    SharedState(const Dataset& x, const Options& o, RunResult& r)
        : data(x), settings(o), result(r), P(o.threads),
          worker_sums(static_cast<std::size_t>(P) * o.k * x.d, 0.0),
          worker_counts(static_cast<std::size_t>(P) * o.k, 0),
          all_sums(static_cast<std::size_t>(o.k) * x.d, 0.0),
          all_counts(o.k, 0) {}
};

struct WorkerArgs {
    SharedState* shared;
    int id;
};

void* worker_function(void* input) {
    WorkerArgs* args = static_cast<WorkerArgs*>(input);
    SharedState& s = *args->shared;
    const int K = s.settings.k;
    const int D = s.data.d;
    const int P = s.P;
    const int id = args->id;
    const std::size_t KD = static_cast<std::size_t>(K) * D;

    double* my_sums = s.worker_sums.data() + static_cast<std::size_t>(id) * KD;
    std::uint64_t* my_counts = s.worker_counts.data() +
                               static_cast<std::size_t>(id) * K;

    // Explicitly divide the observations among P worker threads.
    const std::uint64_t begin = s.data.n * id / P;
    const std::uint64_t end = s.data.n * (id + 1) / P;

    for (int iteration = 0; iteration < s.settings.max_iter; ++iteration) {
        std::fill_n(my_sums, KD, 0.0);
        std::fill_n(my_counts, K, 0);

        // Every worker reads the current centroids and updates local sums.
        for (std::uint64_t i = begin; i < end; ++i) {
            const double* point = s.data.x.data() + i * D;
            const int c = nearest_center(point, s.result.centers.data(), K, D);
            for (int j = 0; j < D; ++j) {
                my_sums[static_cast<std::size_t>(c) * D + j] += point[j];
            }
            ++my_counts[c];
        }

        // BARRIER 1: wait until every worker finishes local aggregation.
        s.barrier_wait();

        // Worker zero merges results and updates centers exactly once.
        if (id == 0) {
            std::fill(s.all_sums.begin(), s.all_sums.end(), 0.0);
            std::fill(s.all_counts.begin(), s.all_counts.end(), 0);
            for (int worker = 0; worker < P; ++worker) {
                const double* sums = s.worker_sums.data() +
                                     static_cast<std::size_t>(worker) * KD;
                const std::uint64_t* counts = s.worker_counts.data() +
                                              static_cast<std::size_t>(worker) * K;
                for (std::size_t j = 0; j < KD; ++j) s.all_sums[j] += sums[j];
                for (int c = 0; c < K; ++c) s.all_counts[c] += counts[c];
            }
            s.result.iterations = iteration + 1;
            const bool finished = update_centers(s.result.centers, s.all_sums,
                                s.all_counts, K, D, s.settings.tolerance);
            s.result.converged = finished;
            s.done.store(finished, std::memory_order_relaxed);
        }

        // BARRIER 2: everybody sees the new centroids before next iteration.
        s.barrier_wait();
        if (s.done.load(std::memory_order_relaxed)) break;
    }
    return nullptr;
}
}  // namespace

const char* backend_name() { return "pthreads"; }

RunResult run_backend(const Dataset& data,
                      const Options& settings,
                      const std::vector<double>& initial_centers) {
    RunResult result;
    result.centers = initial_centers;
    SharedState state(data, settings, result);

    std::vector<pthread_t> handles(static_cast<std::size_t>(settings.threads - 1));
    std::vector<WorkerArgs> args(static_cast<std::size_t>(settings.threads));
    for (int t = 0; t < settings.threads; ++t) args[t] = {&state, t};

    const auto start = std::chrono::steady_clock::now();
    for (int t = 1; t < settings.threads; ++t) {
        if (pthread_create(&handles[t - 1], nullptr,
                           worker_function, &args[t]) != 0) {
            throw std::runtime_error("pthread_create failed");
        }
    }

    // Main thread also participates; it is worker zero.
    worker_function(&args[0]);
    for (pthread_t handle : handles) pthread_join(handle, nullptr);

    result.iteration_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}
