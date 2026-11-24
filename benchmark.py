"""Benchmark three methods on ONLY uploaded real NYC taxi trip records.
All the binary datasets have real original taxi rows; there is no data generator.
"""
import argparse
import csv
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
HEADER = struct.Struct('<8sQI')
MAGIC = b'KMBIN001'
METHODS = ('sequential', 'openmp', 'pthreads')

def read_binary(path):
    path=Path(path)
    with path.open('rb') as f:
        header=f.read(HEADER.size)
    if len(header)!=HEADER.size:
        raise RuntimeError(f'Invalid binary dataset header: {path}')
    magic,n,d=HEADER.unpack(header)
    if magic!=MAGIC or n<=0 or d<=0 or n>200_000_000:
        raise RuntimeError(f'Unsupported taxi binary file: {path}')
    expected=HEADER.size+n*d*8
    if path.stat().st_size!=expected:
        raise RuntimeError(f'Truncated taxi file: {path}. Expected {expected} bytes.')
    return np.memmap(path,dtype='<f8',mode='r',offset=HEADER.size,shape=(n,d))

def generate_initial_centers(path, k, d, seed, target):
    """Deterministic K-Means++: write exact same initial centers for all methods."""
    if target.exists():
        return target
    arr = read_binary(path)
    n, full_d = arr.shape
    if not 1<=k<=n or not 1<=d<=full_d:
        raise ValueError('K or feature count invalid for input')
    rng = np.random.default_rng(seed)
    centers = np.empty((k,d), dtype=np.float64)
    nearest = np.full(n, np.inf, dtype=np.float64)
    ids = set()
    ix = int(rng.integers(n))
    ids.add(ix)
    centers[0] = arr[ix,:d]
    for c in range(1,k):
        last_center = centers[c-1]
        for a in range(0,n,100000):
            block = np.asarray(arr[a:a+100000,:d])
            delta = block-last_center
            sq = np.einsum('ij,ij->i', delta, delta)
            nearest[a:a+len(block)] = np.minimum(nearest[a:a+len(block)],sq)
        total = float(nearest.sum())
        if total<=0 or not math.isfinite(total):
            ix = next(i for i in range(n) if i not in ids)
        else:
            v = rng.random()*total
            running = 0.0
            ix = n-1
            for a in range(0,n,100000):
                p = nearest[a:a+100000]
                v_sum = float(p.sum())
                if v<=running+v_sum:
                    pos = int(np.searchsorted(np.cumsum(p), v-running,side='left'))
                    ix = min(n-1, a+pos)
                    break
                running += v_sum
            if ix in ids:
                ix = next(i for i in range(n) if i not in ids)
        ids.add(ix)
        centers[c] = arr[ix,:d]
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target,centers,delimiter=',',fmt='%.17g')
    return target


def exe(method):
    return ROOT / 'build' / (f'kmeans_{method}'+('.exe' if os.name=='nt' else ''))


def call(method, path, initial, k, d, threads, prefix=None, sample=0):
    args = [str(exe(method)), '--data', str(path), '--centers', str(initial),
            '--k', str(k), '--dims', str(d), '--threads',str(threads),
            '--max-iter','100', '--tol','0.0001', '--sample',str(sample)]
    if prefix:
        args += ['--output-prefix',str(prefix)]
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, errors='replace')
    if p.returncode:
        raise RuntimeError(f'{method} failed (exit {p.returncode}):\n{p.stdout}\n{p.stderr}')
    result = [line[len('RESULT_JSON '):] for line in p.stdout.splitlines()
              if line.startswith('RESULT_JSON ')]
    if len(result)!=1:
        raise RuntimeError(f'No parseable result from {method}:\n{p.stdout}')
    return json.loads(result[0])


def check_quality(raw_rows, dest):
    """Compare each first-run centroid/WCSS to the same sequential baseline."""
    grouped=defaultdict(list)
    for r in raw_rows:
        if r['repeat']==1:
            grouped[(r['dataset'],r['K'],r['D'])].append(r)
    checks=[]
    for key, group in sorted(grouped.items()):
        base = next((r for r in group if r['method']=='sequential'),None)
        if base is None:
            raise RuntimeError(f'Missing sequential result for {key}')
        c0 = np.loadtxt(base['prefix']+'.centroids.csv',skiprows=1,delimiter=',',ndmin=2)[:,1:]
        for r in group:
            c = np.loadtxt(r['prefix']+'.centroids.csv',skiprows=1,delimiter=',',ndmin=2)[:,1:]
            delta = float(np.max(np.abs(c-c0)))
            wdiff = abs(r['wcss']-base['wcss'])/max(1,abs(base['wcss']))
            ok = delta <= 2e-3 and wdiff <= 1e-5 and r['converged']
            checks.append({'dataset':r['dataset'],'K':r['K'],'D':r['D'],
                           'method':r['method'],'threads':r['threads'],
                           'status':'PASS' if ok else 'FAIL',
                           'max_center_diff':delta,'relative_WCSS_diff':wdiff,
                           'iterations':r['iterations'],
                           'same_iteration_count':r['iterations']==base['iterations']})
    write_csv(dest/'correctness.csv',checks)
    passed=sum(r['status']=='PASS' for r in checks)
    print(f'Correctness: {passed}/{len(checks)} comparisons PASSED')
    if passed!=len(checks):
        raise RuntimeError('Some clustering results differ beyond tolerance. Inspect correctness.csv before reporting performance.')
    return checks


def write_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def summarize_benchmark(raw):
    groups=defaultdict(list)
    for r in raw:
        groups[(r['dataset'],r['N'],r['K'],r['D'],r['method'],r['threads'])].append(r)
    summary=[]
    for key,rows in sorted(groups.items()):
        dataset,n,k,d,method,t=key
        ms=[r['iteration_ms'] for r in rows]
        total=[r['total_ms'] for r in rows]
        summary.append({'dataset':dataset,'N':n,'K':k,'D':d,'method':method,'threads':t,
                        'repeats':len(rows),'experiments':rows[0]['experiments'],
                        'median_iteration_ms':float(np.median(ms)),
                        'median_total_ms':float(np.median(total)),
                        'stdev_ms':float(np.std(ms,ddof=1)) if len(ms)>1 else 0.0,
                        'median_iterations':float(np.median([r['iterations'] for r in rows])),
                        'median_wcss':float(np.median([r['wcss'] for r in rows]))})
    baselines={(r['dataset'],r['K'],r['D']):r['median_iteration_ms'] for r in summary if r['method']=='sequential'}
    for r in summary:
        base=baselines[(r['dataset'],r['K'],r['D'])]
        r['speedup']=base/r['median_iteration_ms']
        r['efficiency_percent']=100*r['speedup']/r['threads'] if r['method'] in ('openmp','pthreads') else ''
    return summary


def write_results_summary(summary, output):
    summary_lines = ['CST435: REAL NYC TAXI BENCHMARK RESULTS',
                     'These measurements were performed on actual taxi trips.',
                     '']
    for r in summary:
        if 'dataset_scaling' in r['experiments'].split(';'):
            summary_lines.append(f'{r["method"]:11} N={r["N"]:>10,} threads={r["threads"]} '
                                 f'median={r["median_iteration_ms"]:>10.4f} ms '
                                 f'speedup={r["speedup"]:>7.3f}x')
    thread_rows = [r for r in summary
                   if 'thread_scaling' in r['experiments'].split(';')]
    if thread_rows:
        summary_lines.extend(['', 'THREAD SCALING', 'Records     Method       Threads   Median (ms)   Speedup'])
        for r in sorted(thread_rows, key=lambda row: (row['N'], row['threads'], row['method'])):
            summary_lines.append(f'{r["N"]:>9,}   {r["method"]:11}  {r["threads"]:>7}   '
                                 f'{r["median_iteration_ms"]:>11.2f}   {r["speedup"]:>6.3f}x')
    (output/'RESULTS_SUMMARY.txt').write_text('\n'.join(summary_lines)+'\n',encoding='utf-8')


def plot_clusters(trips, centers, labels, dataset_size, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout='constrained')
    colors = plt.get_cmap('tab10')
    for cluster_id, center in enumerate(centers):
        members = trips[labels == cluster_id]
        color = colors(cluster_id % colors.N)
        ax.scatter(members[:, 0], members[:, 1], s=12, alpha=0.35, color=color,
                   linewidths=0, label=f'Cluster {cluster_id} ({len(members):,})')
        ax.scatter(center[0], center[1], marker='X', s=110, color=color,
                   edgecolor='black', linewidth=0.8, zorder=3)

    ax.set_title('K-means clusters')
    ax.set_xlabel('Trip distance (miles)')
    ax.set_ylabel('Trip duration (minutes)')
    ax.scatter([], [], marker='X', s=90, color='#596778', edgecolor='black',
               label='Cluster center')
    ax.legend(title=f'{len(trips):,} sampled trips\nDataset: {dataset_size:,} trips',
              loc='upper left', markerscale=1.4)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(figures / 'kmeans_clusters.png', dpi=240, facecolor='white')
    plt.close(fig)


def write_cluster_results(raw, output):
    sample = next((r for r in raw if r['method']=='sequential' and r['dataset']==max(raw, key=lambda x:x['N'])['dataset'] and r['repeat']==1 and 'dataset_scaling' in r['experiments'].split(';')),None)
    if sample:
        sample_path = Path(sample['prefix']+'.sample.csv')
        if sample_path.exists():
            values=np.genfromtxt(sample_path,delimiter=',',skip_header=1)
            if values.ndim==1:
                values=values[None,:]
            manifest=json.loads((DATA/'taxi_manifest.json').read_text(encoding='utf-8'))
            means=manifest.get('means',manifest.get('mean'))
            stds=manifest.get('stds',manifest.get('population_standard_deviation'))
            # Cluster profiles are inverse-standardized reference centroids.
            centers_csv=Path(sample['prefix']+'.centroids.csv')
            center_values=np.loadtxt(centers_csv,delimiter=',',skiprows=1,ndmin=2)[:,1:]
            real_centers=center_values*np.asarray(stds[:center_values.shape[1]])+np.asarray(means[:center_values.shape[1]])
            profiles=[]
            ids=values[:,-1].astype(int)
            for cluster_id,center in enumerate(real_centers):
                profiles.append({'cluster':cluster_id,
                                 'distance_miles':float(center[0]),
                                 'duration_minutes':float(center[1]),
                                 'fare_usd':float(center[2]) if len(center)>2 else None,
                                 'tip_usd':float(center[3]) if len(center)>3 else None,
                                 'sampled_points':int(np.count_nonzero(ids==cluster_id))})
            write_csv(output/'cluster_profiles_real_units.csv',profiles)
            trips = values[:, :2] * np.asarray(stds[:2]) + np.asarray(means[:2])
            plot_clusters(trips, real_centers, ids, sample['N'], output)


def benchmark(args, paths):
    methods=METHODS
    if not all(exe(name).exists() for name in methods):
        raise RuntimeError('Build all three CPU executables first (RUN.ps1 compiles them).')
    out = RESULTS / 'taxi'
    if out.exists():
        shutil.rmtree(out)
    (out/'details').mkdir(parents=True,exist_ok=True)
    threads=sorted(set(args.threads))
    cases={}
    def add(tag,path,k,d,method,t):
        key=(str(path),k,d,method,t)
        cases.setdefault(key,set()).add(tag)
    max_t=max(threads)
    d=4
    for path in paths:
        for method in methods:
            add('dataset_scaling',path,args.k,d,method,1 if method=='sequential' else max_t)
    scale = max((p for p in paths if read_binary(p).shape[0]<=1000000),key=lambda p:read_binary(p).shape[0],default=paths[0])
    for method in ('sequential','openmp','pthreads'):
        for t in ([1] if method=='sequential' else threads):
            add('thread_scaling',scale,args.k,d,method,t)
    if args.all_combinations:
        for path in paths:
            for method in ('openmp','pthreads'):
                for t in threads:
                    add('size_thread_sweep',path,args.k,d,method,t)
    if args.extended:
        for k in (3,5,10,20):
            for method in methods:
                add('K_scaling',scale,k,d,method,1 if method=='sequential' else max_t)
        for features in (2,3,4):
            for method in methods:
                add('feature_scaling',scale,args.k,features,method,1 if method=='sequential' else max_t)
    ordered=sorted(cases,key=lambda c:(read_binary(Path(c[0])).shape[0],c[1],c[2],METHODS.index(c[3]),c[4]))
    print(f'\nRUNNING {len(ordered)} test configurations, {args.repeats} measured repetitions each.')
    print('Methods: '+', '.join(methods)+'\n')
    centers={}
    for path,k,d,method,t in ordered:
        cen = out / f'centers_N{read_binary(Path(path)).shape[0]}_K{k}_D{d}.csv'
        key=(path,k,d)
        if key not in centers:
            centers[key]=generate_initial_centers(Path(path),k,d,2026,cen)
    # Warm each configuration, then interleave repetitions to reduce test-order bias.
    for path,k,d,method,t in ordered:
        call(method,Path(path),centers[(path,k,d)],k,d,t)
    raw=[]
    rng=np.random.default_rng(2026)
    for rep in range(1,args.repeats+1):
        indices=rng.permutation(len(ordered))
        print(f'--- Measured repeat {rep}/{args.repeats} ---',flush=True)
        print(f'{"Method":<10}  {"Records":>11}  {"Threads":>7}  {"Compute (ms)":>13}  {"WCSS":>14}')
        for j in indices:
            path,k,d,method,t=ordered[int(j)]
            n=read_binary(Path(path)).shape[0]
            stem=f'{Path(path).stem}_K{k}_D{d}_{method}_T{t}'
            prefix=(out/'details'/stem) if rep==1 else None
            r=call(method,Path(path),centers[(path,k,d)],k,d,t,prefix=prefix,sample=5000)
            r.update({'dataset':Path(path).name,'experiments':';'.join(sorted(cases[(path,k,d,method,t)])),
                      'repeat':rep,'prefix':str(prefix) if prefix else ''})
            raw.append(r)
            print(f'{method:<10}  {n:>11,}  {t:>7}  {r["iteration_ms"]:>13.3f}  {r["wcss"]:>14.4f}')
    write_csv(out/'benchmark_raw.csv',raw)
    summary = summarize_benchmark(raw)
    write_csv(out/'benchmark_summary.csv',summary)
    write_results_summary(summary, out)
    check_quality(raw,out)
    write_cluster_results(raw, out)
    print('\n'+'='*65)
    print('COMPLETE — real-data results (NOT synthetic)')
    print(f'Files: {out}')
    print(f'  benchmark_raw.csv | benchmark_summary.csv | correctness.csv')
    print('  figures/kmeans_clusters.png | Run plot_results.py for performance charts')
    print('\nLARGEST DATASET COMPARISON (median computation time):')
    max_n=max(row['N'] for row in summary)
    for row in summary:
        if row['N']==max_n and 'dataset_scaling' in row['experiments'].split(';'):
            print(f'  {row["method"]:12} {row["median_iteration_ms"]:>10.3f} ms  speedup {row["speedup"]:>6.2f}x')
    print('='*65)


def main():
    parser=argparse.ArgumentParser(description='CST435 Taxi-only K-Means benchmarking')
    parser.add_argument('--sizes',default='100000,1000000',help='Available prepared REAL taxi dataset sizes')
    parser.add_argument('--threads',type=lambda s:[int(v) for v in s.split(',')],default=[1,2,4,8])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--all-combinations',action='store_true',
                        help='Run OpenMP and Pthreads at every requested size and thread count')
    parser.add_argument('--k',type=int,default=5)
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    if args.repeats<1 or args.k<1 or any(t<1 for t in args.threads):
        parser.error('Invalid K, thread count or repetitions')
    try:
        sizes=[int(s) for s in args.sizes.split(',')]
        if any(n<5 for n in sizes):
            raise ValueError('Dataset size must be 5 or more')
        paths=[DATA/f'taxi_{n}.kmbin' for n in sizes]
        missing=[p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                'Missing preprocessed taxi files: '+', '.join(str(x) for x in missing)+
                '. Run python prepare_taxi.py after placing the four original Parquet files in data/raw, '
                'or copy the separate processed dataset into data/ and run prepare_taxi.py --from-clean.')
        for p in paths:
            data=read_binary(p)
            if data.shape[0] != int(p.stem.split('_')[-1]) or data.shape[1]!=4:
                raise RuntimeError(f'Unexpected taxi binary header: {p}')
            print(f'Using REAL taxi data: {p.name}, {len(data):,} trips, 4 features')
        benchmark(args,paths)
    except Exception as error:
        print(f'ERROR: {error}',file=sys.stderr)
        raise SystemExit(1)

if __name__=='__main__':
    main()
