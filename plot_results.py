"""Create performance figures from saved benchmark summaries."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


METHODS = (
    ('openmp', 'OpenMP', '#1769aa', 'o'),
    ('pthreads', 'Pthreads', '#b6501f', 's'),
)
SIZES = (100_000, 1_000_000, 5_000_000, 10_000_000)
THREADS = (1, 2, 4, 8)
RUNTIME = 'Median K-means iteration-phase time (ms)'


def plot_series(ax, x, y, name, color, marker, linestyle='-'):
    ax.plot(x, y, label=name, color=color, marker=marker,
            linestyle=linestyle, linewidth=2, markersize=6)


def plot_dataset_runtime(ax, summary, parallel_threads):
    from matplotlib.ticker import StrMethodFormatter

    medians = {(r['N'], r['method'], r['threads']): r['median_iteration_ms'] for r in summary}
    positions = [n / 1e6 for n in SIZES]
    series = (('sequential', 'Sequential', '#34465a', 'D', 1), *(
        (method, f'{name} ({parallel_threads} threads)', color, marker, parallel_threads)
        for method, name, color, marker in METHODS
    ))
    for method, name, color, marker, threads in series:
        values = [medians.get((n, method, threads), math.nan) for n in SIZES]
        plot_series(ax, positions, values, name, color, marker)
    ax.set_xticks(positions, ['0.1M', '1M', '5M', '10M'])
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))


def plot_thread_metric(ax, summary, records, metric):
    from matplotlib.ticker import PercentFormatter

    rows = {(r['method'], r['threads']): r for r in summary if r['N'] == records}
    for method, name, color, marker in METHODS:
        values = [rows.get((method, threads), {}).get(metric, math.nan) for threads in THREADS]
        plot_series(ax, THREADS, values, name, color, marker)

    if metric == 'median_iteration_ms':
        if records == 1_000_000:
            ax.axhline(377.58, linestyle='--', color='#596778', label='Sequential reference (377.58 ms)')
        elif ('sequential', 1) in rows:
            baseline = rows['sequential', 1]['median_iteration_ms']
            ax.axhline(baseline, linestyle='--', color='#596778', label=f'Sequential ({baseline:.2f} ms)')
    elif metric == 'speedup':
        plot_series(ax, THREADS, THREADS, 'Ideal speedup', '#596778', 'D', '--')
    else:
        ax.axhline(100, linestyle='--', color='#596778', label='100% efficiency')
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xticks(THREADS)


def make_plots(summaries, figures, labels=None, records=1_000_000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    specs = (
        ('execution_time_by_dataset_size', 'Execution time by dataset size', 'Number of records', RUNTIME, None),
        ('execution_time_by_thread_count', 'Execution time by thread count', 'Threads', RUNTIME, 'median_iteration_ms'),
        ('thread_speedup', 'Speedup by thread count', 'Threads', 'Speedup (×)', 'speedup'),
        ('thread_efficiency', 'Parallel efficiency by thread count',
         'Threads', 'Parallel efficiency (%)', 'efficiency_percent'),
    )
    layout = (2, 2) if labels else (1, 1)
    size = (11, 7.5) if labels else (8.4, 5.2)
    if labels:
        parallel_threads = 8
    else:
        parallel_threads = max(
            (r['threads'] for r in summaries[0]
             if r['method'] != 'sequential' and 'dataset_scaling' in r['experiments'].split(';')),
            default=8,
        )

    with plt.rc_context({'font.size': 11, 'legend.frameon': False, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        for filename, title, xlabel, ylabel, metric in specs:
            fig, axes = plt.subplots(*layout, figsize=size, squeeze=False,
                                     sharex=True, sharey=True, layout='constrained')
            fig.suptitle(title)
            fig.supxlabel(f'{xlabel} ({records:,} records)' if metric else xlabel)
            fig.supylabel(ylabel)
            for ax, summary, label in zip(axes.flat, summaries, labels or ['']):
                if metric:
                    plot_thread_metric(ax, summary, records, metric)
                else:
                    plot_dataset_runtime(ax, summary, parallel_threads)
                ax.set_title(label)
                ax.grid(axis='y', color='#dbe2ea')
                ax.set_axisbelow(True)
            first = axes[0, 0]
            first.legend(loc='best', fontsize=9)
            first.set_ylim(bottom=0)
            fig.savefig(figures / f'{filename}.png', dpi=240, facecolor='white')
            plt.close(fig)
    print(f'Generated charts: {figures}')


def read_summary(path):
    if path.is_dir():
        path = path / 'benchmark_summary.csv'
    with path.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f'{path} contains no benchmark rows')
    for row in rows:
        try:
            for field in ('N', 'K', 'D', 'threads'):
                row[field] = int(row[field])
            for field in ('median_iteration_ms', 'speedup'):
                row[field] = float(row[field])
                if not math.isfinite(row[field]) or row[field] <= 0:
                    raise ValueError(f'{field} must be finite and positive')
            if row['method'] in ('openmp', 'pthreads'):
                row['efficiency_percent'] = float(row['efficiency_percent'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid benchmark row in {path}: {error}') from error
    return rows


def shared_speedups(summaries, labels, records, threads):
    configurations = set()
    results = []
    for summary, label in zip(summaries, labels):
        rows = [r for r in summary if r['N'] == records and r['threads'] == threads]
        speedups = []
        for method, name, _, _ in METHODS:
            matches = [r for r in rows if r['method'] == method]
            if len(matches) != 1:
                raise ValueError(f'{label} needs exactly one {name} result at {records:,} records and {threads} threads')
            row = matches[0]
            configurations.add((row['dataset'], row['K'], row['D']))
            speedups.append(row['speedup'])
        results.append(speedups)
    if len(configurations) != 1:
        raise ValueError('The speedup comparison requires the same dataset, cluster count, and feature count')
    return results


def print_speedup_table(labels, results, records, threads):
    width = max(len('Computer'), *(len(label) for label in labels))
    print(f'\nSpeedup at {records:,} records and {threads} threads')
    print(f'{"Computer":<{width}}  {"OpenMP":>9}  {"Pthreads":>9}')
    averages = [mean(values) for values in zip(*results)]
    for label, (openmp, pthreads) in zip([*labels, 'Mean'], [*results, averages]):
        print(f'{label:<{width}}  {openmp:>8.3f}×  {pthreads:>8.3f}×')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path, metavar='RESULT',
                        help='One or four benchmark_summary.csv files or directories containing them')
    parser.add_argument('--labels', nargs=4, metavar='NAME', help='Computer labels in input order')
    parser.add_argument('--records', type=int, default=1_000_000, help='Record count for the comparison')
    parser.add_argument('--threads', type=int, default=4, help='Thread count for the speedup table')
    parser.add_argument('--output', type=Path, help='Directory for the generated figures')
    args = parser.parse_args()

    paths = [path / 'benchmark_summary.csv' if path.is_dir() else path for path in args.results]
    if len(paths) not in (1, 4) or len({path.resolve() for path in paths}) != len(paths):
        parser.error('provide one summary or four distinct summaries')
    if args.records < 1 or args.threads < 1:
        parser.error('records and threads must be positive')
    try:
        summaries = [read_summary(path) for path in paths]
        if len(paths) == 1:
            if args.labels:
                parser.error('--labels requires four summaries')
            output = args.output or paths[0].parent / 'figures'
            make_plots(summaries, output, records=args.records)
        else:
            labels = args.labels or [f'Computer {i}' for i in range(1, 5)]
            output = args.output or Path('results/comparison')
            results = shared_speedups(summaries, labels, args.records, args.threads)
            make_plots(summaries, output, labels, args.records)
            print_speedup_table(labels, results, args.records, args.threads)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
