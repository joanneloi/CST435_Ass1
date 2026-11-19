// ==============================================================
// CST435 METHOD 2: OPENMP PARALLEL K-MEANS (multicore CPU)
// ==============================================================
// OpenMP automatically divides the point-assignment loop among
// CPU threads. Every thread writes into its OWN cluster sums and
// counts, so there is no race condition during the large loop.
// The partial sums are merged in a fixed order by one thread.

#include "kmeans.hpp"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

const char* backend_name() { return "openmp"; }

RunResult run_backend(const Dataset& data,
                      const Options& settings,
                      const std::vector<double>& initial_centers) {
    RunResult result;
    result.centers = initial_centers;

    const int K = settings.k;
    const int D = data.d;
    const int P = settings.threads;
    const std::size_t KD = static_cast<std::size_t>(K) * D;

    // Each worker has different memory for its own sums and counts.
    std::vector<double> worker_sums(static_cast<std::size_t>(P) * KD, 0.0);
    std::vector<std::uint64_t> worker_counts(
        static_cast<std::size_t>(P) * K, 0);
    std::vector<double> all_sums(KD, 0.0);
    std::vector<std::uint64_t> all_counts(K, 0);
    bool finished = false;

    omp_set_dynamic(0);
    const auto start = std::chrono::steady_clock::now();

    // Keep one OpenMP worker team alive over all K-Means iterations.
    // Creating threads once reduces repeated startup overhead.
#pragma omp parallel num_threads(P) shared(result, finished, worker_sums, worker_counts, all_sums, all_counts)
    {
        const int id = omp_get_thread_num();
        double* my_sums = worker_sums.data() + static_cast<std::size_t>(id) * KD;
        std::uint64_t* my_counts =
            worker_counts.data() + static_cast<std::size_t>(id) * K;

        for (int iteration = 0; iteration < settings.max_iter; ++iteration) {
            std::fill_n(my_sums, KD, 0.0);
            std::fill_n(my_counts, K, 0);

            // OpenMP splits all real observations into equal-sized chunks.
            // Each worker uses private local aggregation arrays.
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(data.n); ++i) {
                const double* point = data.x.data() + static_cast<std::size_t>(i) * D;
                const int c = nearest_center(point, result.centers.data(), K, D);
                for (int j = 0; j < D; ++j) {
                    my_sums[static_cast<std::size_t>(c) * D + j] += point[j];
                }
                ++my_counts[c];
            }

            // After the 'for' barrier, only one worker combines sums.
#pragma omp single
            {
                std::fill(all_sums.begin(), all_sums.end(), 0.0);
                std::fill(all_counts.begin(), all_counts.end(), 0);

                const int actual_workers = omp_get_num_threads();
                for (int worker = 0; worker < actual_workers; ++worker) {
                    const double* s = worker_sums.data() +
                                      static_cast<std::size_t>(worker) * KD;
                    const std::uint64_t* n = worker_counts.data() +
                                             static_cast<std::size_t>(worker) * K;
                    for (std::size_t j = 0; j < KD; ++j) all_sums[j] += s[j];
                    for (int c = 0; c < K; ++c) all_counts[c] += n[c];
                }

                result.iterations = iteration + 1;
                finished = update_centers(result.centers, all_sums,
                                          all_counts, K, D, settings.tolerance);
                result.converged = finished;
            }  // Implicit barrier publishes updated centroids.

            if (finished) break;
        }
    }

    result.iteration_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}
