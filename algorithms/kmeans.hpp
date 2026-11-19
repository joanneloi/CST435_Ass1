#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct Dataset {
    std::uint64_t n = 0;
    int d = 0;
    std::vector<double> x; // normalized row-major x[n*d]
};
struct Options {
    std::string data, centers_file, output_prefix;
    int k = 5, d = 0, threads = 4, max_iter = 100, sample = 3000;
    double tolerance = 1.0e-4;
};
struct RunResult {
    std::vector<double> centers;
    int iterations = 0;
    bool converged = false;
    double iteration_ms = 0.0;
    double transfer_ms = 0.0;
    double setup_ms = 0.0;
};

Options parse_options(int argc, char** argv);
Dataset read_kmbin(const std::string& path, int dimensions);
std::vector<double> read_centers(const std::string& path, int k, int d);
int nearest_center(const double* x, const double* centers, int k, int d);
double squared_distance(const double* a, const double* b, int d);
bool update_centers(std::vector<double>& centers, const std::vector<double>& sums,
                    const std::vector<std::uint64_t>& counts, int k, int d, double tol);
void save_outputs(const Options& opt, const Dataset& data, const RunResult& r,
                  const char* method, double wcss, double total_ms);

// Each executable provides its own method implementation.
const char* backend_name();
RunResult run_backend(const Dataset& data, const Options& options,
                      const std::vector<double>& initial_centers);
