// ==============================================================
// CST435 METHOD 1: SEQUENTIAL K-MEANS (single CPU thread)
// ==============================================================
// This is the reference algorithm against which all parallel
// versions are checked. Every observation is processed in order.
//
// Input:   Standardized real data X[N][D], K, initial K centroids.
// Output:  Final centroids, convergence state and execution time.
//
// The common file kmeans.cpp handles input, output, WCSS scoring,
// timing output and command-line options for ALL three methods.

#include "kmeans.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

const char* backend_name() {
    return "sequential";
}

RunResult run_backend(const Dataset& data,
                      const Options& settings,
                      const std::vector<double>& initial_centers) {
    RunResult result;
    result.centers = initial_centers;

    const int K = settings.k;
    const int D = data.d;
    const std::size_t center_values = static_cast<std::size_t>(K) * D;

    // Sum of all points assigned to each centroid.
    std::vector<double> cluster_sums(center_values, 0.0);
    std::vector<std::uint64_t> cluster_counts(K, 0);

    const auto start = std::chrono::steady_clock::now();

    // Lloyd's algorithm: assignment -> aggregation -> centroid update.
    for (int iteration = 0; iteration < settings.max_iter; ++iteration) {
        std::fill(cluster_sums.begin(), cluster_sums.end(), 0.0);
        std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

        // STEP A. Each point is assigned to the nearest centroid.
        // STEP B. Add its feature values to its centroid's total.
        for (std::uint64_t i = 0; i < data.n; ++i) {
            const double* point = data.x.data() + i * D;
            const int c = nearest_center(point, result.centers.data(), K, D);

            for (int feature = 0; feature < D; ++feature) {
                cluster_sums[static_cast<std::size_t>(c) * D + feature]
                    += point[feature];
            }
            ++cluster_counts[c];
        }

        // STEP C. Calculate the average feature vector in each cluster.
        // Stop if the maximum centroid movement is <= tolerance.
        result.iterations = iteration + 1;
        result.converged = update_centers(
            result.centers, cluster_sums, cluster_counts,
            K, D, settings.tolerance);

        if (result.converged) {
            break;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    result.iteration_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}
