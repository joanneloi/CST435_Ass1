taxi_all.kmbin    = all 14,011,666 cleaned REAL NYC yellow taxi trip records.
taxi_100000.kmbin  = 100,000 REAL trips, selected without replacement.
taxi_1000000.kmbin = 1,000,000 REAL trips, selected without replacement.
taxi_5000000.kmbin and taxi_10000000.kmbin = larger real subsets, created
using `python prepare_taxi.py --from-clean` from taxi_all.kmbin.

NO synthetic data. NO Mall Customers data.

Each KMBIN001 binary has a 20-byte header (magic 8 bytes, row count uint64,
feature count uint32), followed by float64 row-major data, 4 features:
  [standardized distance, duration, fare, tip].
Use the real 500-row preview CSV and manifest to inspect original units.
