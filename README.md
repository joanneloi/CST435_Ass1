# CST435 K-Means on NYC Yellow Taxi Trips

This project compares Sequential, OpenMP, and Pthreads implementations of K-Means on real NYC yellow taxi trips.

## 1. Project files

| Path | Purpose |
|---|---|
| `algorithms/sequential.cpp` | Sequential implementation |
| `algorithms/openmp.cpp` | OpenMP implementation |
| `algorithms/pthreads.cpp` | POSIX threads implementation |
| `algorithms/kmeans.cpp` and `algorithms/kmeans.hpp` | Shared data loading, K-Means logic, timing, and output |
| `benchmark.py` | Runs each test, checks correctness, summarizes runtimes, and generates plots |
| `RUN.ps1` | Windows-only runner that builds the three programs and starts the benchmark |
| `requirements.txt` | Python dependencies |
| `data/` | Dataset manifest and preview |
| `results/` | Benchmark output |

## 2. Set up the project

Open a terminal in the project folder. Both platforms use the preprocessed datasets listed in section 3.

### Windows

Install Python and the MSYS2 UCRT64 C++ toolchain. From the MSYS2 UCRT64 terminal, install the compiler if needed.

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-gcc
```

In PowerShell, install the benchmark dependencies.

```powershell
python -m pip install numpy matplotlib
```

The runner checks the default MSYS2 location at `C:\msys64\ucrt64\bin`. If MSYS2 is installed elsewhere, add its UCRT64 `bin` folder to your PATH.

### macOS

Install Python and the Xcode Command Line Tools. Use Homebrew to install `libomp` for OpenMP, then build the three executables.

```sh
xcode-select --install
brew install libomp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib

OMP_PREFIX="$(brew --prefix libomp)"
mkdir -p build
clang++ -std=c++17 -O3 algorithms/kmeans.cpp algorithms/sequential.cpp -o build/kmeans_sequential
clang++ -std=c++17 -O3 -I"$OMP_PREFIX/include" -Xpreprocessor -fopenmp \
  algorithms/kmeans.cpp algorithms/openmp.cpp -L"$OMP_PREFIX/lib" -Wl,-rpath,"$OMP_PREFIX/lib" \
  -lomp -o build/kmeans_openmp
clang++ -std=c++17 -O3 -pthread algorithms/kmeans.cpp algorithms/pthreads.cpp -o build/kmeans_pthreads
```

## 3. Preprocessed datasets

Download the preprocessed `.kmbin` files from the [GitHub dataset release](https://github.com/joanneloi/CST435_Ass1/releases/tag/taxi-datasets-v1) and place them in `data/`. The default benchmark uses the first four files. The full dataset is also available.

| Trips | Download |
|---:|---|
| 100,000 | [taxi_100000.kmbin](https://github.com/joanneloi/CST435_Ass1/releases/download/taxi-datasets-v1/taxi_100000.kmbin) |
| 1,000,000 | [taxi_1000000.kmbin](https://github.com/joanneloi/CST435_Ass1/releases/download/taxi-datasets-v1/taxi_1000000.kmbin) |
| 5,000,000 | [taxi_5000000.kmbin](https://github.com/joanneloi/CST435_Ass1/releases/download/taxi-datasets-v1/taxi_5000000.kmbin) |
| 10,000,000 | [taxi_10000000.kmbin](https://github.com/joanneloi/CST435_Ass1/releases/download/taxi-datasets-v1/taxi_10000000.kmbin) |
| 14,011,666 | [taxi_all.kmbin](https://github.com/joanneloi/CST435_Ass1/releases/download/taxi-datasets-v1/taxi_all.kmbin) |

Each trip has four standardized features (distance, duration, fare, and tip). The subsets come from 14,011,666 cleaned NYC yellow taxi trips from January through April 2025. Each smaller subset is contained in the next larger one. See [the dataset manifest](data/taxi_manifest.json) for details and [the preview CSV](data/taxi_real_500_preview.csv) for examples in the original units.

## 4. Run the benchmark

The current Windows runner defaults use **100K, 1M, 5M, and 10M trips**, **1, 2, and 4 threads**, **5 clusters**, and **10 measured repeats per configuration**. Sequential uses one thread. OpenMP and Pthreads use four threads on every dataset and also test one and two threads on the 1M dataset.

On Windows, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN.ps1
```

On macOS, run the equivalent benchmark with the virtual environment active:

```sh
python benchmark.py --sizes 100000,1000000,5000000,10000000 --threads 1,2,4,8 --repeats 10 --k 5
```

### Save a copy of your results

Each benchmark recreates `results/taxi/`. To keep a run, use the command for your operating system from the project folder after it finishes and before starting another benchmark.

**macOS terminal**

```sh
cp -R results/taxi "results/taxi_backup_$(date +%Y%m%d_%H%M%S)"
```

**Windows PowerShell**

```powershell
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Copy-Item -Path '.\results\taxi' -Destination ".\results\taxi_backup_$timestamp" -Recurse
```

The copy appears inside `results/` with the date and time in its folder name.

## 5. Results

| Path under `results/taxi/` | Contents |
|---|---|
| `benchmark_raw.csv` | One row per measured run |
| `benchmark_summary.csv` | Median runtimes, speedups, and thread efficiency |
| `correctness.csv` | First-run comparisons of cluster centers and WCSS against Sequential |
| `RESULTS_SUMMARY.txt` | Runtime and speedup summary |
| `centers_N*_K*_D*.csv` | Shared initial cluster centers |
| `details/` | Detailed clustering output |
| `figures/` | Charts for runtime, speedup, thread efficiency, and taxi clusters |
