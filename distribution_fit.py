import scipy as sp
import math
from facebook_violating_exposures_estimation import plot as plot
import numpy as np
import random
import json
import datetime


def normal_distribution(x, p):
    # p = [u, s, n]
    return p[2] / (p[1]*np.sqrt(2.0 * math.pi)) * np.exp(-1.0 * (x - p[0])**2 / (2.0 * p[1]**2))


FIXED_U_FOR_TAIL_DISTRIBUTION = -3.0
def normal_distribution_tail(x, p, u=FIXED_U_FOR_TAIL_DISTRIBUTION):
    # p = [s, n], u is fixed
    return p[1] / (p[0]*np.sqrt(2.0 * math.pi)) * np.exp(-1.0 * (x - u)**2 / (2.0 * p[0]**2))


def bin_sampled_distribution(x, p, func, n=1000, n_extra_bins=1):
    # A wrapper style function so that the *area* of the function can be fit to given area values.
    # x is the list of x values for the fit points.
    # p is the parameter list
    # n is the number of slices to cut the function into (infinity for perfect integral etc)
    # n_extra_bins is the number of extra bins to carry on the area calculation for. This is because
    # the final bins in the data are always >= N bins, so the last data point includes full area above.

    width = x[1] - x[0]
    delta = width * 1.0 / n

    bin_sampled_ys = []
    for i, xx in enumerate(x):
        if i < len(x) - 1:
            xs = np.linspace(
                xx - width / 2.0,
                xx + width / 2.0,
                num=n, endpoint=False
            )
        else:
            xs = np.linspace(
                xx - width / 2.0,
                xx + width / 2.0 + width * n_extra_bins,
                num=n*(n_extra_bins+1), endpoint=False
            )
        ys = func(xs, p)
        bin_sampled_ys.append(sum(ys) * delta)

    return np.array(bin_sampled_ys)


def chi2_fit(func, x, y, e, p):
    error_func = lambda p, x, y, e: (func(x, p) - y) / e
    fit, hess_inv, infodict, errmsg, success = sp.optimize.leastsq(error_func, p, args=(np.array(x), np.array(y), np.array(e)), full_output=1)
    res_variance = (error_func(fit, np.array(x), np.array(y), np.array(e))**2).sum()/(len(y)-len(p) )

    for i in range(5):
        if hess_inv is None or res_variance is None:
            fit, hess_inv, infodict, errmsg, success = sp.optimize.leastsq(error_func, fit, args=(np.array(x), np.array(y), np.array(e)), full_output=1)
            res_variance = (error_func(fit, np.array(x), np.array(y), np.array(e))**2).sum()/(len(y)-len(p) )
        else:
            break

    if hess_inv is None or res_variance is None:
        print("ERROR: BAD CHI2 FIT")
        print(x)
        print(y)
        print(e)
        print(p)
        print(fit)
        print(hess_inv)
        print(infodict)
        print(errmsg)
        print(success)
        print(res_variance)
        if 'are at most 0.000000 and the relative error between two consecutive iterates is at' in errmsg:
            cov = None
        elif 'Both actual and predicted relative reductions in the sum of squares' in errmsg:
            cov = None
        elif 'The cosine of the angle between func(x) and any column of the' in errmsg:
            cov = None
        elif 'The relative error between two consecutive iterates is at most 0.000000' in errmsg:
            cov = None
        else:
            raise ValueError
    else:
        cov = hess_inv * res_variance
    return fit, cov


def fit_histogram_with_normal(xs, ys, es, n=100, n_extra_bins=1):
    d = xs[1] - xs[0]

    # Now, we want to function to fit the distribution, so fit to y*d
    fit_ys = [y * d for y in ys]
    fit_es = [e * d for e in es]

    if ys[1] < 0.66 * ys[0] and ys[0]==max(ys):
        # Use the normal_distribution_tail case...
        func = normal_distribution_tail
        p0 = [2.0, ys[0]*3.0]
    else:
        # Just use normal distribution...
        func = normal_distribution
        p0 = [1.0, 2.0, 1.0]

    func_fit = lambda x, p: bin_sampled_distribution(x, p, func, n=n, n_extra_bins=n_extra_bins)

    fit_p, fit_cov = chi2_fit(func_fit, xs, fit_ys, fit_es, p0)

    best_fit_xs = np.linspace(xs[0]-d/2.0, xs[-1]+d/2.0+d*n_extra_bins, num=n*(len(xs)+n_extra_bins), endpoint=False)
    best_fit_ys = func(best_fit_xs, fit_p)

    best_fit_xs = [float(x) for x in best_fit_xs]
    best_fit_ys = [float(y) for y in best_fit_ys]

    return best_fit_xs, best_fit_ys


def fit_histogram_with_spline(xs, ys, es, n_extra_bins=1, n=100):
    width = xs[1] - xs[0]
    xs_spl = list(xs)
    ys_spl = list(ys)

    r = ((ys_spl[0] / ys_spl[1]) - 1.0) / 2.0 + 1.0

    xs_spl.insert(0, xs_spl[0] - width / 2.0)
    ys_spl.insert(0, ys_spl[0] * r)

    xs_spl.append(xs_spl[-1] + (n_extra_bins + 0.5) * width)
    ys_spl.append(0.0)

    while xs_spl[0] >= 0.5:
        xs_spl.insert(0, xs_spl[0] - width / 2.0)
        ys_spl.insert(0, ys_spl[0] * r)


    min_x_for_zero = -1.0
    for x, y in zip(xs_spl, ys_spl):
        if y > 0.0:
            break
        min_x_for_zero = x + width

    x_spl = []
    y_spl = []
    spl = None

    redo = True
    while redo:
        redo = False

        try:
            spl = sp.interpolate.make_interp_spline(xs_spl, ys_spl, bc_type="clamped", k=3)
        except:
            print("Failed to fit spline")
            print(xs_spl)
            print(ys_spl)
            raise ValueError

        x_spl = np.linspace(0, xs_spl[-1], num=n * (len(xs) + n_extra_bins), endpoint=False)
        y_spl = spl(x_spl)

        x_spl = [float(x) for x in x_spl]
        y_spl = [float(y) for y in y_spl]

        for x, y in zip(x_spl, y_spl):
            if y < 0 and x < xs_spl[-2] and x >= min_x_for_zero:
                redo = True
                try:
                    x_l = max([xx for xx in xs_spl if xx <= x])
                    x_r = min([xx for xx in xs_spl if xx > x])
                except:
                    print(xs_spl)
                    print(x)
                    raise ValueError
                y_l = [yy for xx, yy in zip(xs_spl, ys_spl) if xx==x_l][0]
                y_r = [yy for xx, yy in zip(xs_spl, ys_spl) if xx==x_l][0]
                i = [ii for ii, xx in enumerate(xs_spl) if xx==x_r][0]
                xs_spl.insert(i, 0.5 * (x_l + x_r))
                ys_spl.insert(i, 0.5 * (y_l + y_r))
                break

        y_spl = np.maximum(y_spl, 0.0)

    x_spl = list([float(x) for x in x_spl])
    y_spl = list([float(y) for y in y_spl])

    return x_spl, y_spl


def rescale_curves(xs, curve_x, samp_ys, curve_ys, n=100, n_extra_bins=1):
    width = xs[1] - xs[0]
    d = width * 1.0 / n

    final_scaled_ys = [[] for i in range(len(curve_ys))]
    for i in range(len(xs) + n_extra_bins):
        ii = i
        i_l = n * i
        i_h = n * (i + 1)
        i_l_area = n * i
        i_h_area = n * (i + 1)
        if i >= len(xs)-1:
            i_l_area = n * (len(xs)-1)
            i_h_area = n * (len(xs) + n_extra_bins)
            ii = len(xs) - 1

        areas = [sum(curve_y[i_l_area:i_h_area])*d for curve_y in curve_ys]
        scale_factors = [y[ii]*width / a if a > 0.0 else 0.0 for y, a in zip(samp_ys, areas)]

        scaled_ys = [[sf * y for y in curve_ys[j][i_l: i_h]] for j, sf in enumerate(scale_factors)]

        for j, scaled_y in enumerate(scaled_ys):
            final_scaled_ys[j] += scaled_y

    return final_scaled_ys


def bin_stats_from_curves(xs, curve_x, curve_ys, n=100, n_extra_bins=1):
    width = xs[1] - xs[0]
    d = width * 1.0 / n

    mean_bins = []
    for i in range(len(xs) + n_extra_bins):
        i_l = i * n
        i_h = (i + 1) * n

        areas = [sum(curve_y[i_l:i_h])*d for curve_y in curve_ys]
        mean_xs = [sum([x*y*d for x, y in zip(curve_x[i_l:i_h], curve_y[i_l:i_h])]) / a if a > 0.0 else 0.0 for curve_y, a in zip(curve_ys, areas)]
        views = [a * 10**x for x, a in zip(mean_xs, areas)]

        mb = {
            'bin_id': width / 2.0 + i * width,
            'weight_mean': float(np.mean(areas)),
            'weight_std': float(np.std(areas)),
            'log_10_views_mean': float(np.mean(mean_xs)),
            'log_10_views_std': float(np.mean(mean_xs)),
            'weighted_views_mean': float(np.mean(views)),
            'weighted_views_std': float(np.std(views)),
        }
        mean_bins.append(mb)

    fit_bins = {k: [fb[k] for fb in mean_bins] for k in mean_bins[0].keys()}
    return fit_bins


def average_curves(curves):
    lens = [len(curve) for curve in curves]
    pivot_curve = [[curve[i] for curve in curves] for i in range(len(curves[0]))]
    mean_curve = [float(np.mean(pc)) for pc in pivot_curve]
    std_curve = [float(np.std(pc)) for pc in pivot_curve]
    return mean_curve, std_curve


def regenerate_distribution_within_errors(xs, ys, es, n=1):
    width = xs[1] - xs[0]
    area = sum(ys) * width
    sampled_ys = [[max([0.0, random.uniform(y-e, y+e)]) for y, e in zip(ys, es)] for i in range(n)]
    areas = [sum(sampled_y)*width for sampled_y in sampled_ys]
    sampled_ys = [[y * area / a for y in sampled_y] for sampled_y, a in zip(sampled_ys, areas)]
    return sampled_ys[0] if n==1 else sampled_ys


def low_mid_high_views_bins(xs, ys):
    width = xs[1] - xs[0]

    bins = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        bins.append({
            'bin_id': x,
            'views_low': 10**(x - width/2.0) * y * width,
            'views_mid': 10**x * y * width,
            'views_high': 10**(x + width/2.0) * y * width,
        })
    totals = {
        'low': sum([b['views_low'] for b in bins]),
        'mid': sum([b['views_mid'] for b in bins]),
        'high': sum([b['views_high'] for b in bins]),
    }

    return totals, {k: [b[k] for b in bins] for k in bins[0].keys()}


def min_max_views_bins(xs, sample_ys):
    samples = [low_mid_high_views_bins(xs, ys) for ys in sample_ys]
    totals = [t for t, _ in samples]
    bins = [b for _, b in samples]
    totals = {
        'views_min_mean': float(np.mean([t['low'] for t in totals])),
        'views_min_std': float(np.std([t['low'] for t in totals])),
        'views_mid_mean': float(np.mean([t['mid'] for t in totals])),
        'views_mid_std': float(np.std([t['mid'] for t in totals])),
        'views_max_mean': float(np.mean([t['high'] for t in totals])),
        'views_max_std': float(np.std([t['high'] for t in totals])),
    }


    mean_bins = []
    for i in range(len(xs)):
        bin = {
            'views_min_mean': float(np.mean([b['views_low'][i] for b in bins])),
            'views_min_std': float(np.std([b['views_low'][i] for b in bins])),
            'views_mid_mean': float(np.mean([b['views_mid'][i] for b in bins])),
            'views_mid_std': float(np.std([b['views_mid'][i] for b in bins])),
            'views_max_mean': float(np.mean([b['views_high'][i] for b in bins])),
            'views_max_std': float(np.std([b['views_high'][i] for b in bins])),
        }
        mean_bins.append(bin)

    bins = {k: [b[k] for b in mean_bins] for k in mean_bins[0].keys()}
    bins['x'] = xs

    return totals, bins


def rough_min_max_views(xs, ys, es):
    width = xs[1] - xs[0]
    d = width / 2.0

    area = sum(ys) * width
    ys = [y / area for y in ys]
    es = [e / area for e in es]

    ls = [x-d for x in xs]
    hs = [x+d for x in xs]

    ys_maxed = list(ys)
    error_to_dist = es[-1]
    ys_maxed[-1] = ys_maxed[-1] + es[-1]
    i = 0
    while error_to_dist > 0:
        if error_to_dist < es[i]:
            ys_maxed[i] = ys_maxed[i] - error_to_dist
            error_to_dist = 0.0
        else:
            ys_maxed[i] = ys_maxed[i] - es[i]
            error_to_dist = error_to_dist - es[i]
        i += 1

    ys_min = list(ys)
    error_to_remove = es[-1]
    i = -1
    while error_to_remove > 0.0:
        if error_to_remove <= es[i]:
            ys_min[i] = ys_min[i] - error_to_remove
            if ys_min[i] < 0.0:
                error_to_remove = ys_min[i] * -1
                ys_min[i] = 0.0
            else:
                error_to_remove = 0.0
        else:
            ys_min[i] = ys_min[i] - es[i]
            if ys_min[i] < 0.0:
                error_to_remove = error_to_remove - es[i] - ys_min[i]
                ys_min[i] = 0.0
            else:
                error_to_remove = error_to_remove - es[i]
        i -= 1

    i = 0
    error_to_dist = es[-1]
    while error_to_dist > 0.0:
        if error_to_dist <= es[i]:
            ys_min[i] = ys_min[i] + error_to_dist
            error_to_dist = 0.0
        else:
            ys_min[i] = ys_min[i] + es[i]
            error_to_dist = error_to_dist - es[i]
        i += 1

    area_min = sum(ys_min) * width
    ys_min = [y / area_min for y in ys_min]
    area_max = sum(ys_maxed) * width
    ys_maxed = [y / area_max for y in ys_maxed]


    mid = sum([10**x * width * y for x, y in zip(xs, ys)])

    binning_min = sum([10**x * width * y for x, y in zip(ls, ys)])
    binning_max = sum([10**x * width * y for x, y in zip(hs, ys)])

    rounding_min = sum([10**x * width * y for x, y in zip(xs, ys_min)])
    rounding_max = sum([10**x * width * y for x, y in zip(xs, ys_maxed)])

    total_min = sum([10**x * width * y for x, y in zip(ls, ys_min)])
    total_max = sum([10**x * width * y for x, y in zip(hs, ys_maxed)])

    return {
        'dist_mid_views': mid,
        'dist_binning_min_views': binning_min,
        'dist_binning_max_views': binning_max,
        'dist_rounding_min_views': rounding_min,
        'dist_rounding_max_views': rounding_max,
        'dist_total_min_views': total_min,
        'dist_total_max_views': total_max,
    }


def data_views(xs, ys, es, sample_ys, zero_frac=None):
    data_totals, data_bins = low_mid_high_views_bins(xs, ys)
    sampled_totals, sampled_bins = min_max_views_bins(xs, sample_ys)

    data = {
        'bin_id': xs,
        'data': ys,
        'data_uncert': es,
        'data_views_low': data_bins['views_low'],
        'data_views_mid': data_bins['views_mid'],
        'data_views_high': data_bins['views_high'],
        'data_views_rounding_std': sampled_bins['views_mid_std'],
        'data_views_min': [max([0.0, m-s]) for m, s in zip(sampled_bins['views_min_mean'], sampled_bins['views_min_std'])],
        'data_views_max': [m+s for m, s in zip(sampled_bins['views_max_mean'], sampled_bins['views_max_std'])],
    }

    if zero_frac is not None:
        nzf = 1.0 - zero_frac
        data_with_0 = {k+'_with_0': [vv*nzf for vv in v] if k!='bin_id' else v for k, v in data.items()}
        data.update(data_with_0)

    return data


def estimate_views_using_model(xs, ys, es, model_func, n=100, n_samples=1000, n_extra_bins=1, zero_frac=None):
    # Now, if there are multiple training 0 bins, the fit+projection
    # will get wonky (Due to rounding uncertainty).
    # So truncate the distribution such that there
    # is at most 1 trailing 0 bin.
    while ys[-2]==0.0 and ys[-1]==0.0:
        xs = xs[:-1]
        ys = ys[:-1]
        es = es[:-1]


    sampled_ys = regenerate_distribution_within_errors(xs, ys, es, n=n_samples)

    best_fit_x, best_fit_y = model_func(xs, ys, es, n=n, n_extra_bins=n_extra_bins)

    fits = [model_func(xs, cur_ys, es, n=n, n_extra_bins=n_extra_bins) for cur_ys in sampled_ys]

    fit_x, _ = fits[0]
    fit_ys = [fit_y for _, fit_y in fits]

    rescaled_ys = rescale_curves(xs, fit_x, sampled_ys, fit_ys, n=n, n_extra_bins=n_extra_bins)

    rescaled_curve, rescaled_curve_std = average_curves(rescaled_ys)
    rescaled_curve_low = [max([0.0, y-e]) for y, e in zip(rescaled_curve, rescaled_curve_std)]
    rescaled_curve_high = [y+e for y, e in zip(rescaled_curve, rescaled_curve_std)]
    best_fit_low = [max([0.0, y-e]) for y, e in zip(best_fit_y, rescaled_curve_std)]
    best_fit_high = [y+e for y, e in zip(best_fit_y, rescaled_curve_std)]

    fit_bins_rescaled = bin_stats_from_curves(xs, fit_x, rescaled_ys, n=n, n_extra_bins=n_extra_bins)
    fit_bins_rescaled_low = bin_stats_from_curves(xs, fit_x, [rescaled_curve_low], n=n, n_extra_bins=n_extra_bins)
    fit_bins_rescaled_high = bin_stats_from_curves(xs, fit_x, [rescaled_curve_high], n=n, n_extra_bins=n_extra_bins)

    fit_bins_best = bin_stats_from_curves(xs, fit_x, [best_fit_y], n=n, n_extra_bins=n_extra_bins)
    fit_bins_best_low = bin_stats_from_curves(xs, fit_x, [best_fit_low], n=n, n_extra_bins=n_extra_bins)
    fit_bins_best_high = bin_stats_from_curves(xs, fit_x, [best_fit_high], n=n, n_extra_bins=n_extra_bins)

    fit_bins = fit_bins_rescaled
    fit_bins.update({k+'_low': v for k, v in fit_bins_rescaled.items()})
    fit_bins.update({k+'_high': v for k, v in fit_bins_rescaled_high.items()})
    fit_bins.update({k+'_best': v for k, v in fit_bins_best.items()})
    fit_bins.update({k+'_best_low': v for k, v in fit_bins_best_low.items()})
    fit_bins.update({k+'_best_high': v for k, v in fit_bins_best_high.items()})

    estimated_views = sum(fit_bins_rescaled['weighted_views_mean'])
    estimated_views_uncert = math.sqrt(sum([v*v for v in fit_bins_rescaled['weighted_views_std']]))

    fit = {
        'estimate': (estimated_views, estimated_views_uncert),
        'curves': {
            'x': fit_x,
            'best_fit': best_fit_y,
            'best_fit_low': best_fit_low,
            'best_fit_high': best_fit_high,
            'rescaled_fit': rescaled_curve,
            'rescaled_fit_low': rescaled_curve_low,
            'rescaled_fit_high': rescaled_curve_high,
        },
        'fit_bins': fit_bins,
    }

    if zero_frac is not None:
        nzf = 1.0 - zero_frac
        fit['estimate_with_0'] = (fit['estimate'][0]*nzf, fit['estimate'][0]*nzf)
        fit['curves_with_0'] = {k: [y*nzf for y in v] if k!='x' else v for k, v in fit['curves'].items()}
        fit['fit_bins_with_0'] = {k: [vv*nzf for vv in v] if 'bin_id' not in k and 'log_10' not in k else v for k, v in fit['fit_bins'].items()}

    return fit


def estimate_views_of_histogram(xs, ys, es, n=100, n_samples=1000, n_extra_bins=1, zero_frac=None):
    # Sort the histogram
    srt = sorted(zip(xs, ys, es), key=lambda x: x[0], reverse=False)
    xs = [x for x, y, e in srt]
    ys = [y for x, y, e in srt]
    es = [e for x, y, e in srt]

    # Normalize the histogram...
    width = xs[1] - xs[0]
    area = sum([y * width for y in ys])

    fit_ys = [y / area for y in ys]
    fit_es = [e / area for e in es]

    normal_dist_fit = estimate_views_using_model(xs, fit_ys, fit_es, fit_histogram_with_normal, n=n, n_samples=n_samples, n_extra_bins=n_extra_bins, zero_frac=zero_frac)

    spline_dist_fit = estimate_views_using_model(xs, fit_ys, fit_es, fit_histogram_with_spline, n=n, n_samples=n_samples, n_extra_bins=n_extra_bins, zero_frac=zero_frac)

    sampled_ys = regenerate_distribution_within_errors(xs, fit_ys, fit_es, n=n_samples)
    data_fit_bins = data_views(xs, fit_ys, fit_es, sampled_ys, zero_frac=zero_frac)


    data_estimates = rough_min_max_views(xs, fit_ys, fit_es)

    estimates = {
        'Binning Uncertainty': (data_estimates['dist_binning_min_views'], data_estimates['dist_mid_views'], data_estimates['dist_binning_max_views']),
        'Rounding Uncertainty': (data_estimates['dist_rounding_min_views'], data_estimates['dist_mid_views'], data_estimates['dist_rounding_max_views']),
        'Total Uncertainty': (data_estimates['dist_total_min_views'], data_estimates['dist_mid_views'], data_estimates['dist_total_max_views']),
        'Modeled (Normal)': (
            max([normal_dist_fit['estimate'][0] - normal_dist_fit['estimate'][1], 0.0]),
            normal_dist_fit['estimate'][0],
            normal_dist_fit['estimate'][0] + normal_dist_fit['estimate'][1]
        ),
        'Modeled (Spline)': (
            max([spline_dist_fit['estimate'][0] - spline_dist_fit['estimate'][1], 0.0]),
            spline_dist_fit['estimate'][0],
            spline_dist_fit['estimate'][0] + spline_dist_fit['estimate'][1]
        )
    }

    fit = {
        'spline': spline_dist_fit,
        'normal': normal_dist_fit,
        'data': data_fit_bins,
        'estimates': estimates,
    }

    if zero_frac is not None:
        nzf = 1.0 - zero_frac
        fit['estimates_with_0'] = {
            k: (v[0]*nzf, v[1]*nzf, v[2]*nzf) for k, v in fit['estimates'].items()
        }
        fit['data_with_0'] = {k.replace('_with_0', ''): v for k, v in fit['data'].items() if '_with_0' in k or k=='bin_id'}

    return fit


def plot_estimation_from_discrete_distribution(data, estimates, fit_bins, curves, filetag, label=None, style=None):
    fit = {
        'fit_bins': fit_bins,
        'curves': curves,
        'estimates': estimates,
        'data': data,
    }
    if label is None:
        label = filetag.split('/')[-1].strip()

    estimates_uncert = {k: (v[1], (v[1]-v[0], v[2]-v[1])) for k, v in estimates.items()}

    wz = ''

    xs = data['bin_id']
    ys = data['data']
    es = data['data_uncert']
    es2x = [2*w for w in fit_bins['weight_std']]

    ys_l = [max([0.0, m-s]) for m, s in zip(fit_bins['weight_mean'], fit_bins['weight_std'])]
    ys_h = [m+s for m, s in zip(fit_bins['weight_mean'], fit_bins['weight_std'])]

    wvses2x = [v*2 for v in fit_bins['weighted_views_std']]
    wvs = fit_bins['weighted_views_mean']
    wvs_l = [max([0.0, m-s]) for m, s in zip(fit_bins['weighted_views_mean'], fit_bins['weighted_views_std'])]
    wvs_h = [m+s for m, s in zip(fit_bins['weighted_views_mean'], fit_bins['weighted_views_std'])]

    bfvs_l = fit_bins['weighted_views_mean_best_low']
    bfvs_h = fit_bins['weighted_views_mean_best_high']

    dvs_l = data['data_views_low']
    dvs_h = data['data_views_high']
    dvs_ll = data['data_views_min']
    dvs_hh = data['data_views_max']

    x_dvs = list(xs)
    while dvs_h[-1]==0.0:
        dvs_h = dvs_h[:-1]
        dvs_l = dvs_l[:-1]
        x_dvs = x_dvs[:-1]
    x_dvsmm = list(xs)
    while dvs_hh[-1]==0.0:
        dvs_hh = dvs_hh[:-1]
        dvs_ll = dvs_ll[:-1]
        x_dvsmm = x_dvsmm[:-1]

    curves = fit['curves']

    plot.plot([
        ('bar', xs, ys, {'color': 'tab:blue', 'width': 0.9, 'label': 'Provided Data', 'alpha': 0.6}),
        ('fill_between', curves['x'], curves['best_fit_low'], curves['best_fit_high'], {'alpha': 0.60, 'color': 'tab:orange', 'label': None}),
        ('plot', curves['x'], curves['best_fit'], {'color': 'tab:orange', 'label': 'Best Fit'}),
        ('errorbar', xs, ys, {'color': 'tab:blue', 'yerr': es, 'marker': 'o', 'linestyle': ''}),
        ],
        title="Best Fit Views Distribution [{}]".format(label),
        xlabel="Log10(Views)",
        ylabel="Distribution (a.u.)",
        xlim=[0.0, None],
        ylim=[0.0, None],
        style=style,
        show=False,
        save="{}_Best_Fit.png".format(filetag),
    )

    plot.plot([
        ('bar', xs, ys, {'color': 'tab:blue', 'width': 0.9, 'label': 'Provided Data', 'alpha': 0.6}),
        ('fill_between', curves['x'], curves['rescaled_fit_low'], curves['rescaled_fit_high'], {'alpha': 0.60, 'color': 'tab:orange', 'label': None}),
        ('plot', curves['x'], curves['rescaled_fit'], {'color': 'tab:orange', 'label': 'Modeled Fit'}),
        ('errorbar', xs, ys, {'color': 'tab:blue', 'yerr': es, 'marker': 'o', 'linestyle': ''}),
        ],
        title="Modeled Views Distribution [{}]".format(label),
        xlabel="Log10(Views)",
        ylabel="Distribution (a.u.)",
        xlim=[0.0, None],
        ylim=[0.0, None],
        style=style,
        show=False,
        save="{}_Modeled_Fit.png".format(filetag),
    )

    plot.plot([
        ('bar', fit_bins['bin_id'], wvses2x, {'color': 'tab:orange', 'width': 0.85, 'bottom': wvs_l, 'label': 'Model', 'alpha': 0.6})
        ],
        title="Views Per Bin [{}]".format(label),
        xlabel="Log10(Views) Bin",
        ylabel="Views",
        xlim=[0.0, None],
        ylim=[0.0, None],
        style=style,
        show=False,
        save="{}_Modeled_Views_Per_Bin.png".format(filetag),
    )

    plot.plot([
        ('fill_between', fit_bins['bin_id'], bfvs_l, bfvs_h, {'color': 'tab:orange', 'alpha': 0.5, 'label': 'Best Fit'}),
        ('fill_between', x_dvs, dvs_l, dvs_h, {'color': 'tab:blue', 'alpha': 0.5, 'label': 'Data'}),],
        title="Views Per Bin [{}]".format(label),
        xlabel="Log10(Views) Bin",
        ylabel="Views",
        xlim=[0.0, None],
        ylim=[0.0, None],
        style=style,
        show=False,
        save="{}_Best_Fit_Views_Per_Bin.png".format(filetag),
    )

    plot.plot([
        ('fill_between', x_dvsmm, dvs_ll, dvs_hh, {'color': 'tab:blue', 'alpha': 0.3, 'label': 'Max Data Range'}),
        ('fill_between', x_dvs, dvs_l, dvs_h, {'color': 'tab:blue', 'alpha': 0.5, 'label': 'Data'}),
        ('plot', fit_bins['bin_id'], wvs, {'color': 'tab:orange', 'label': 'Model', 'marker': 'o', 'linestyle': '-'}),
        ],
        title="Views Per Bin [{}]".format(label),
        xlabel="Log10(Views) Bin",
        ylabel="Views",
        xlim=[0.0, None],
        ylim=[0.0, None],
        style=style,
        show=False,
        save="{}_Modeled_Data_Views_Per_Bin.png".format(filetag),
    )

    estimates_uncert_pairs = list(estimates_uncert.items())
    estimates_xs = [i+1 for i in range(len(estimates_uncert_pairs))]
    estimates_labels = [k for k, b in estimates_uncert_pairs]
    estimates_ys = [v[0] for k, v in estimates_uncert_pairs]
    estimates_ebs = [[v[1][0] for k, v in estimates_uncert_pairs], [v[1][1] for k, v in estimates_uncert_pairs]]

    models_i = [i for i, l in enumerate(estimates_labels) if 'modeled' in l.lower()]
    models_xs = [x for i, x in enumerate(estimates_xs) if i in models_i]
    models_labels = [x for i, x in enumerate(estimates_labels) if i in models_i]
    models_ys = [x for i, x in enumerate(estimates_ys) if i in models_i]
    models_ebs = [
        [e for i, e in enumerate(estimates_ebs[0]) if i in models_i],
        [e for i, e in enumerate(estimates_ebs[1]) if i in models_i],
    ]

    markers = ['o' if 'modeled' in l.lower() else '' for l in estimates_labels]
    plot.plot([
        ('errorbar', estimates_ys, estimates_xs, {'color': 'tab:blue', 'marker': '', 'capsize': 2, 'linestyle': '', 'xerr': estimates_ebs, 'label': 'Data Limits'}),
        ('errorbar', models_ys, models_xs, {'color': 'tab:orange', 'marker': 'o', 'capsize': 2, 'linestyle': '', 'xerr': models_ebs, 'label': 'Modeled'}),
        ],
        xlabel="Views",
        ylabel="Estimate Type",
        yticks={'ticks': estimates_xs, 'labels': estimates_labels},
        title="Estimated Views Per Video [{}]".format(label),
        style=style,
        show=False,
        save="{}_View_Estimates.png".format(filetag)
    )


if __name__=="__main__":
    data = {
    "x": [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        6.5
    ],
    "y": [
        0.005,
        0.121,
        0.028,
        0.004,
        0.006,
        0.0,
        0.0
    ],
    "es": [
        0.0005,
        0.0005,
        0.0005,
        0.0005,
        0.0005,
        0.0005,
        0.0005
    ],
    "zero_frac": 0.835
    }


    xs = data['x']
    ys = data['y']
    es = data['es']
    zero_frac = data['zero_frac']

    n = 100
    n_samples = 1000
    n_extra_bins = 1

    fit = estimate_views_of_histogram(xs, ys, es, n=n, n_samples=n_samples, n_extra_bins=n_extra_bins, zero_frac=zero_frac)

    plot_estimation_from_discrete_distribution(
        fit['data'], fit['estimates'], fit['normal']['fit_bins'], fit['normal']['curves'],
        '/tmp/test_norm_', label="normal w/o 0",
    )

    plot_estimation_from_discrete_distribution(
        fit['data'], fit['estimates'], fit['spline']['fit_bins'], fit['spline']['curves'],
        '/tmp/test_spline_', label="spline w/o 0",
    )

    plot_estimation_from_discrete_distribution(
        fit['data_with_0'], fit['estimates_with_0'], fit['normal']['fit_bins_with_0'], fit['normal']['curves_with_0'],
        '/tmp/test_norm_w0_', label="normal w/ 0",
    )

    plot_estimation_from_discrete_distribution(
        fit['data_with_0'], fit['estimates_with_0'], fit['spline']['fit_bins_with_0'], fit['spline']['curves_with_0'],
        '/tmp/test_spline_w0_', label="spline w/0",
    )

