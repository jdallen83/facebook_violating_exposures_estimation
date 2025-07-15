import os
import json
import math
import random
import numpy as np
import sys

import facebook_violating_exposures_estimation.distribution_fit as distribution_fit
import facebook_violating_exposures_estimation.plot as plot


def sample_curve(curve_x, curve_y):
    d = curve_x[1] - curve_x[0]
    r = random.random()

    i = 0
    s = 0.0
    while s <= r:
        s += d * curve_y[i]
        i += 1

    f = (s - r) / (d * curve_y[i-1])
    return f * d + curve_x[i-1]


def bin_samples(xs, samples):
    d = xs[1] - xs[0]
    bins = []
    for s in samples:
        for i, x in enumerate(xs):
            if s >= x - d/2.0 and s < x + d/2.0:
                bins.append(x)
                break
            elif i==len(xs) - 1:
                bins.append(x)
                break
    bin_counts = {x: len([b for b in bins if b==x]) for x in xs}
    return [bin_counts[x] for x in xs]


def generate_sampled_histogram(xs, curve_x, curve_y, rounding=3, n_samples=1000000, zero_frac=0.0):
    d = curve_x[1] - curve_x[0]
    width = xs[1] - xs[0]

    area = sum(curve_y) * d
    curve_y_norm = [y / area for y in curve_y]

    mean_x = sum([x * y * d for x, y in zip(curve_x, curve_y_norm)])
    mean_views = sum([10**x * y * d for x, y in zip(curve_x, curve_y_norm)])


    samples = [sample_curve(curve_x, curve_y_norm) for i in range(n_samples)]
    total_views = sum([10**s for s in samples])
    avg_views = total_views * 1.0 / n_samples

    bins = bin_samples(xs, samples)

    binned_area = sum(bins) * width
    bins_norm = [b / binned_area * (1.0 - zero_frac) for b in bins]
    bins_round = [round(b, rounding) for b in bins_norm]
    bins_error = [5 * 10**(-1 * (rounding+1)) for b in bins_round]

    return {
        'xs': xs,
        'ys': bins_round,
        'es': bins_error,
        'ys_no_rounding': bins_norm,
        'es_no_rounding': [10E-7 for b in bins_norm],
        'mean_x': mean_x,
        'mean_x_with_0': mean_x * (1.0 - zero_frac),
        'mean_views': mean_views,
        'mean_views_with_0': mean_views * (1.0 - zero_frac),
        'total_views': total_views,
        'average_views': avg_views,
    }


def parse_doc(doc):
    if 'normal' in doc:
        return {
            'data_xs': doc['histo_bins']['x'],
            'data_ys': doc['histo_bins']['y'],
            'data_es': doc['histo_bins']['e'],
            'data_zero_frac': doc['histo_bins']['zero_frac'],
            'fit_curve_x': doc['normal']['curves']['x'],
            'fit_curve_normal': doc['normal']['curves']['best_fit'],
            'fit_curve_normal_rescaled': doc['normal']['curves']['rescaled_fit'],
            'fit_curve_spline': doc['spline']['curves']['best_fit'],
            'fit_curve_spline_rescaled': doc['spline']['curves']['rescaled_fit'],
        }
    elif 'histo_bins' in doc:
        return {
            'data_xs': doc['histo_bins']['x'],
            'data_ys': doc['histo_bins']['y'],
            'data_es': doc['histo_bins']['e'],
            'data_zero_frac': doc['histo_bins']['zero_frac'],
            'fit_curve_x': doc['curves']['x'],
            'fit_curve_normal': doc['curves']['best_fit'],
            'fit_curve_normal_rescaled': doc['curves']['scaled_fit'],
            'fit_curve_spline': doc['curves_spline']['best_fit'],
            'fit_curve_spline_rescaled': doc['curves_spline']['scaled_fit'],
    }
    else:
        return {
            'data_xs': doc['data']['x'],
            'data_ys': doc['data']['y'],
            'data_es': doc['data']['es'],
            'data_zero_frac': doc['data']['zero_frac'],
            'fit_curve_x': doc['curves']['x'],
            'fit_curve_normal': doc['curves']['best_fit'],
            'fit_curve_normal_rescaled': doc['curves']['scaled_fit'],
            'fit_curve_spline': doc['curves_spline']['best_fit'],
            'fit_curve_spline_rescaled': doc['curves_spline']['scaled_fit'],
    }


if __name__=="__main__":
    dir = sys.argv[1]
    rounding = int(sys.argv[2])
    out_dir = sys.argv[3]

    print(dir, rounding, out_dir)

    FILES = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.json')])

    docs = [json.load(open(f)) for f in FILES]

    for f, doc in zip(FILES, docs):
        tag = f.split('/')[-1]
        print(tag)
        out_file = os.path.join(out_dir, tag + '__refitting.json')
        if os.path.isfile(out_file):
            continue

        parsed_doc = parse_doc(doc)
        xye = sorted(zip(parsed_doc['data_xs'], parsed_doc['data_ys'], parsed_doc['data_es']), key=lambda x: x[0])
        parsed_doc['data_xs'] = [x for x, y, e in xye]
        parsed_doc['data_ys'] = [y for x, y, e in xye]
        parsed_doc['data_es'] = [e for x, y, e in xye]
        sampled_histograms = {}
        fits = {}
        for curve in ('fit_curve_normal_rescaled', 'fit_curve_spline_rescaled', 'fit_curve_normal', 'fit_curve_spline'):
            print("\t", curve)
            sampled_histogram = generate_sampled_histogram(
                parsed_doc['data_xs'],
                parsed_doc['fit_curve_x'],
                parsed_doc[curve],
                rounding=rounding,
                n_samples=1000000,
                zero_frac=parsed_doc['data_zero_frac'],
            )

            for k, v in sampled_histogram.items():
                print(k, v)

            fit = distribution_fit.estimate_views_of_histogram(
                sampled_histogram['xs'],
                sampled_histogram['ys'],
                sampled_histogram['es'],
                n=100,
                n_samples=5000,
                n_extra_bins=1,
                zero_frac=parsed_doc['data_zero_frac'],
            )
            fits[curve] = {}
            fits[curve]['fit'] = fit
            fits[curve]['sampled_histogram'] = sampled_histogram

        out_doc = {}
        out_doc['doc'] = doc
        out_doc['parsed_doc'] = parsed_doc
        out_doc['fits'] = fits

        json.dump(out_doc, open(out_file, 'w'), indent=2)
