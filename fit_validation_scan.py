import os
import json

import numpy as np
import collections

import facebook_violating_exposures_estimation.distribution_fit_validation as dfv
import facebook_violating_exposures_estimation.distribution_fit as df
import facebook_violating_exposures_estimation.plot as plot


# TikTok
U_MEAN = 1.96
U_STD = 0.4

L_MEAN = 0.7
L_STD = 0.35

# YouTube
U = -3.0
L_MEAN_YT = 2.182037045531944
L_STD_YT = 0.1261763156535121


def fit_simulation_run(u, l, x_min, x_max, x_hist_max, hist_bin_width, n_hist_samples, rounding, n_extra_bins, manual_x=None, manual_y=None, cache_dir=None, tag=None):
    cache_file = "SIM_{}{}_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
        tag or '', u, l, x_min, x_max, x_hist_max,
        hist_bin_width, n_hist_samples,
        n_extra_bins, rounding,
    )

    if cache_dir is not None and os.path.isfile(os.path.join(cache_dir, cache_file)):
        return json.load(open(os.path.join(cache_dir, cache_file)))

    print(cache_file)

    xs = None
    ys = None
    if manual_x is None or manual_y is None:
        xs = np.linspace(x_min, x_max, num=round(100 * (x_max - x_min)), endpoint=False)
        d = xs[1] - xs[0]
        ys = df.normal_distribution(xs, [u, l, 1.0])
        area = d * sum(ys)
        ys = [y / area for y in ys]

        x_sels = [x + d/2.0 for x in xs]
        ys_norm = [y * d for y in ys]
    else:
        x_sels = manual_x
        ys_norm = manual_y
        xs = manual_x
        d = xs[1] - xs[0]
        area = sum(ys_norm) * d
        ys = [y / area for y in ys_norm]

    samples = None
    avg = None

    samples = np.random.choice(x_sels, p=ys_norm, size=n_hist_samples)
    avg = sum(10**samples) * 1.0 / n_hist_samples

    x_bins = []
    x = hist_bin_width / 2.0
    bins_for_np = []
    while x < x_hist_max:
        x_bins.append(x)
        bins_for_np.append(x + hist_bin_width / 2.0)
        x += hist_bin_width

    bin_ids = np.digitize(samples, bins_for_np)
    hist = {int(k): v for k, v in collections.Counter(bin_ids).items()}

    for i in range(len(bins_for_np)):
        if i not in hist:
            hist[i] = 0

    hist = [y for _, y in sorted(list(hist.items()), key=lambda x: x[0])]

    if len(hist) > len(bins_for_np):
        hist[-2] = hist[-2] + hist[-1]
        hist = hist[:-1]

    hist_area = sum(hist) * hist_bin_width

    hist_normed = [c * 1.0 / hist_area for c in hist]
    hist_rounded = [round(y, rounding) for y in hist_normed]
    hist_uncert = [5 * 10**(-1 * (rounding + 1)) for y in hist_rounded]

    fit = df.estimate_views_of_histogram(x_bins, hist_rounded, hist_uncert, n=100, n_samples=15000, n_extra_bins=n_extra_bins, zero_frac=0.0)

    r_doc = {
        'true_average': avg,
        'true_dist_x': [float(v) for v in xs],
        'true_dist_y': [float(v) for v in ys],
        'true_histogram_x': [float(v) for v in x_bins],
        'true_histogram_y': [float(v) for v in hist_rounded],
        'true_histogram_e': [float(v) for v in hist_uncert],
        'fit': fit,
        'run_params': {
            'u': u,
            'l': l,
            'x_min': x_min,
            'x_max': x_max,
            'x_hist_max': x_hist_max,
            'hist_bin_width': hist_bin_width,
            'n_hist_samples': n_hist_samples,
            'rounding': rounding,
            'n_extra_bins': n_extra_bins,
            'manual_x': manual_x,
            'manual_y': manual_y,
            'tag': tag,
        }
    }

    if cache_dir is not None:
        json.dump(r_doc, open(os.path.join(cache_dir, cache_file), 'w'), indent=2)

    return r_doc


def fit_simulation_run_wrap(doc, cache_dir=None):
    if 'status_print' in doc:
        print(doc['status_print'])
    try:
        return fit_simulation_run(
            doc['u'], doc['l'],
            0.0, doc['sample_x_max'],
            doc['bin_max'], doc['bin_width'],
            doc.get('n_hist_samples', 300000000),
            doc['rounding'], doc['n_extra_bins'],
            cache_dir=doc.get('cache_dir', cache_dir),
            manual_x=doc.get('manual_x', None),
            manual_y=doc.get('manual_y', None),
            tag=doc.get('tag', None),
        )
    except:
        print("FAILURE")
        print(json.dumps(doc, indent=2))
        return None


def manual_x_y_from_fitdata(infile, fit_type='spline'):
    doc = json.load(open(infile))

    x = doc[fit_type]['curves']['x']
    y = doc[fit_type]['curves']['rescaled_fit']
    d = x[1] - x[0]
    a = sum(y)
    ys_norm = [yy / a for yy in y]
    x_sels = [xx + 0.5 * d for xx in x]

    return x_sels, ys_norm


def get_manual_runs(infiles, runs, fit_type='spline'):
    MANUAL_RUNS = []
    for infile in infiles:
        manual_x, manual_y = manual_x_y_from_fitdata(infile)
        for run in runs:
            r = dict(run)
            r['u'] = infile.split('[')[-1].split('-')[0]
            r['l'] = infile.split(']')[-2].split('-')[-1].replace('_', ' ').strip().replace(' ', '_')
            r['manual_x'] = manual_x
            r['manual_y'] = manual_y
            if fit_type=='normal':
                r['tag'] = 'normal_'
            elif fit_type=='spline':
                r['tag'] = 'spline_'
            MANUAL_RUNS.append(r)

    return MANUAL_RUNS


YOUTUBE_US = [-3.0]
YOUTUBE_LS = [
    L_MEAN_YT - L_STD_YT,
    L_MEAN_YT,
    L_MEAN_YT + L_STD_YT,
]
YOUTUBE_LS = [round(l, 3) for l in YOUTUBE_LS]

TIKTOK_US = [U_MEAN - 1.5 * U_STD, U_MEAN, U_MEAN + U_STD]
TIKTOK_US = [round(u, 3) for u in TIKTOK_US]
TIKTOK_LS = [L_MEAN - L_STD, L_MEAN, L_MEAN + 1.5 * L_STD]
TIKTOK_LS = [round(l, 3) for l in TIKTOK_LS]

N_EXTRA_BINS = [0, 1, 2, 3]
BIN_WIDTHS = [0.1, 0.5, 1.0]
BIN_MAX = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
SAMPLE_X_MAX = [9.0]
ROUNDING = [2, 3, 4, 5, 12]


RUNS = []
for n_extra_bins in N_EXTRA_BINS:
    for bin_width in BIN_WIDTHS:
        for bin_max in BIN_MAX:
            for sample_x_max in SAMPLE_X_MAX:
                for rounding in ROUNDING:
                    RUNS.append({
                        'n_extra_bins': n_extra_bins,
                        'bin_width': bin_width,
                        'bin_max': bin_max,
                        'sample_x_max': sample_x_max,
                        'rounding': rounding,
                    })
YT_RUNS = []
for u in YOUTUBE_US:
    for l in YOUTUBE_LS:
        for run in RUNS:
            r = dict(run)
            r['u'] = u
            r['l'] = l
            YT_RUNS.append(r)

TT_RUNS = []
for u in TIKTOK_US:
    for l in TIKTOK_LS:
        for run in RUNS:
            r = dict(run)
            r['u'] = u
            r['l'] = l
            TT_RUNS.append(r)


ALL_RUNS = YT_RUNS + TT_RUNS


if __name__=="__main__":
    import sys
    import random
    from multiprocessing import Pool

    cache_dir = sys.argv[1]
    n_processes = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    manual_files = []
    if len(sys.argv) >= 4:
        manual_files = sys.argv[3:]

    ALL_RUNS += get_manual_runs(manual_files, RUNS, 'spline')
    ALL_RUNS += get_manual_runs(manual_files, RUNS, 'normal')

    final_runs = []
    for i, r in enumerate(ALL_RUNS):
        r['cache_dir'] = cache_dir
        r['status_print'] = "({}/{})".format(i, len(ALL_RUNS))
        final_runs.append(r)

    random.shuffle(final_runs)

    if n_processes is None:
        with Pool() as p:
            p.map(fit_simulation_run_wrap, final_runs)
    else:
        with Pool(n_processes) as p:
            p.map(fit_simulation_run_wrap, final_runs)
