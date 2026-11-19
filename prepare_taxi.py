"""Prepare the user's four REAL NYC Yellow Taxi Parquet files for K-Means.

Nothing synthetic is generated. Each output row is drawn from an actual trip.
Use --from-clean if data/taxi_all.kmbin was downloaded separately.

Input: data/raw/yellow_tripdata_2025-01.parquet (through 2025-04)
Output: data/taxi_all.kmbin plus reproducible nested real-row subsets.
"""
import argparse
import json
import struct
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
RAW=DATA/'raw'
HEADER=struct.Struct('<8sQI')
MAGIC=b'KMBIN001'
FEATURES=['trip_distance_miles','duration_minutes','fare_amount_usd','tip_amount_usd']


def data_batches(files):
    """Read real records in small batches, without loading a whole month into RAM."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError('Run: python -m pip install pyarrow pandas numpy') from e
    columns=['tpep_pickup_datetime','tpep_dropoff_datetime',
             'trip_distance','fare_amount','tip_amount']
    for file in files:
        parquet=pq.ParquetFile(str(file))
        missing=set(columns)-set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f'{file.name} is missing required columns: {missing}')
        for batch in parquet.iter_batches(batch_size=100_000,columns=columns):
            df=batch.to_pandas()
            # Duration is derived from two REAL timestamps, not invented.
            duration=(df['tpep_dropoff_datetime']-df['tpep_pickup_datetime']).dt.total_seconds().to_numpy()/60.
            dist=df['trip_distance'].to_numpy(dtype=np.float64,na_value=np.nan)
            fare=df['fare_amount'].to_numpy(dtype=np.float64,na_value=np.nan)
            tip=df['tip_amount'].to_numpy(dtype=np.float64,na_value=np.nan)
            mask=(np.isfinite(dist)&np.isfinite(duration)&np.isfinite(fare)&np.isfinite(tip)&
                  (dist>0)&(dist<=200)&(duration>=1)&(duration<=240)&
                  (fare>0)&(fare<=1000)&(tip>=0)&(tip<=200))
            values=np.column_stack((dist[mask],duration[mask],fare[mask],tip[mask]))
            yield file,len(df),values


def create_full(files,force=False):
    full=DATA/'taxi_all.kmbin'
    manifest=DATA/'taxi_manifest.json'
    if full.exists() and manifest.exists() and not force:
        print('Using existing cleaned real taxi data:',full)
        return full
    if not all(f.is_file() for f in files):
        raise FileNotFoundError('Put January-April 2025 Yellow Taxi .parquet files in data/raw/ first.')
    n=0
    totals={f.name:{'original':0,'kept':0} for f in files}
    summed=np.zeros(4,dtype=np.float64)
    sumsq=np.zeros(4,dtype=np.float64)
    # Pass 1: record cleaning statistics and global z-score parameters.
    print('Pass 1: reading all four real Parquet files and calculating statistics...',flush=True)
    for file,original,arr in data_batches(files):
        n+=len(arr)
        totals[file.name]['original']+=original
        totals[file.name]['kept']+=len(arr)
        summed+=arr.sum(axis=0)
        sumsq+=np.square(arr).sum(axis=0)
    if n==0: raise RuntimeError('No valid taxi trips after cleaning')
    mean=summed/n
    std=np.sqrt(np.maximum(sumsq/n-mean*mean,0.))
    std=np.where(std>1e-12,std,1.)
    # Pass 2: process ALL valid trips, not only a small sample.
    print('Pass 2: writing all cleaned taxi trips...',flush=True)
    full.parent.mkdir(exist_ok=True)
    tmp=full.with_suffix('.tmp')
    written=0
    with tmp.open('wb') as out:
        out.write(HEADER.pack(MAGIC,n,4))
        for file,original,arr in data_batches(files):
            out.write(np.ascontiguousarray((arr-mean)/std,dtype='<f8').tobytes())
            written+=len(arr)
    if written!=n: raise RuntimeError('Raw input data changed between passes')
    tmp.replace(full)
    info={
        'source':'2025 NYC TLC Yellow Taxi actual trip records (January-April)',
        'raw_month_files':[f.name for f in files],
        'original_rows':sum(x['original'] for x in totals.values()),
        'valid_rows':n,
        'dropped_rows':sum(x['original'] for x in totals.values())-n,
        'rows_by_month':totals,
        'features':FEATURES,
        'cleaning_rules':{
            'distance_miles':'0 < distance <= 200',
            'duration_minutes':'1 <= duration <= 240',
            'fare_usd':'0 < fare <= 1000',
            'tip_usd':'0 <= tip <= 200',
            'all_values':'must be finite'},
        'mean':mean.tolist(),'population_standard_deviation':std.tolist(),
        'normalization':'(raw feature - mean) / standard deviation, on ALL cleaned rows',
        'selection':'No synthetic samples; each row is from an original taxi trip'}
    manifest.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(f'FULL REAL DATASET: {n:,} trips; removed {info["dropped_rows"]:,} invalid rows.')
    return full


def create_subsets(full,sizes,seed=2026,force=False):
    with full.open('rb') as f:
        magic,n,d=HEADER.unpack(f.read(HEADER.size))
    if magic!=MAGIC or d!=4 or full.stat().st_size!=HEADER.size+n*d*8:
        raise RuntimeError('Full taxi binary is missing or corrupt')
    sizes=sorted(set(sizes))
    if max(sizes)>n: raise ValueError(f'Only {n:,} cleaned real trips are available')
    missing=[s for s in sizes if force or not (DATA/f'taxi_{s}.kmbin').is_file()]
    if not missing:
        print('All requested REAL taxi subsets already exist.')
        return
    arr=np.memmap(full,mode='r',offset=HEADER.size,dtype='<f8',shape=(n,d))
    # Samples without replacement. The smaller datasets are nested subsets
    # of the same uniformly randomized set of genuine trips.
    rng=np.random.default_rng(seed)
    chosen=rng.choice(n,size=min(n,10_000_000),replace=False,shuffle=True)
    for size in sizes:
        path=DATA/f'taxi_{size}.kmbin'
        if path.exists() and not force: continue
        with path.open('wb') as out:
            out.write(HEADER.pack(MAGIC,size,d))
            for i in range(0,size,100_000):
                block=np.asarray(arr[chosen[i:min(i+100_000,size)]])
                out.write(np.ascontiguousarray(block,dtype='<f8').tobytes())
        print(f'Saved REAL subset: {path.name}, {size:,} actual trips')


def main():
    p=argparse.ArgumentParser(description='Use ONLY January-April 2025 REAL NYC Taxi Parquet data')
    p.add_argument('--from-clean',action='store_true',help='Use already processed data/taxi_all.kmbin; do not read Parquet')
    p.add_argument('--sizes',default='100000,1000000,5000000,10000000')
    p.add_argument('--force',action='store_true',help='Overwrite existing processed datasets')
    a=p.parse_args()
    sizes=[int(x) for x in a.sizes.split(',')]
    if not sizes or min(sizes)<5: p.error('All subset sizes must be at least 5')
    files=[RAW/f'yellow_tripdata_2025-{m:02d}.parquet' for m in range(1,5)]
    full=DATA/'taxi_all.kmbin' if a.from_clean else create_full(files,a.force)
    if not full.is_file():
        raise FileNotFoundError('Missing data/taxi_all.kmbin; put the full preprocessed file there or process raw Parquet.')
    create_subsets(full,sizes,force=a.force)
    print('COMPLETE — no synthetic records and no customer data.')

if __name__=='__main__':
    main()
