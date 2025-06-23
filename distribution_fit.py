import scipy as sp
import math
from facebook_violating_exposures_estimation import plot as plot
import numpy as np
import random
import json


SMALL_FLOAT = 1E-280 # Have to define a minimum float size to round to 0, for division by 0 issues


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


def single_bin_estimate(x_low, x_high, func, p, cov, area, area_l, area_h, area_uncert, n=100, n_samples=300):
    # This function will produce best fit curves, scaled curve fit to the observed distribution,
    # uncertainties in the distribution, and estimates of views from the bin.
    # n is the number of slices for the integration.
    # n_samples is the number of samples to run for uncertainty estimation.
    xs = []
    bf = []
    bf_l = []
    bf_h = []
    sf = []
    sf_l = []
    sf_h = []

    width = x_high - x_low
    d = 1.0 * width / n
    if area is None and area_l is not None and area_h is not None:
        area = 0.5 * (area_l + area_h)
    if area_l is None and area_h is None and area is not None and area_uncert is not None:
        area_l = area - area_uncert if area - area_uncert > 0.0 else 0.0
        area_h = area + area_uncert

    xs = np.linspace(x_low, x_high, num=n, endpoint=False)
    bf = [float(y) for y in func(xs, p)]

    sampled_ps = [np.random.multivariate_normal(p, cov) for i in range(n_samples)]

    sampled_ys = [func(xs, pp) for pp in sampled_ps]

    sampled_areas = [sum(ys)*d for ys in sampled_ys]
    target_areas = [random.uniform(area_l, area_h) for _ in xs] if area_uncert is None else [np.random.normal(area, area_uncert) for _ in xs]
    scale_factors = [ta / sa if sa > SMALL_FLOAT else 0.0 for ta, sa in zip(target_areas, sampled_areas)]
    scaled_ys = [sf * sy for sf, sy in zip(scale_factors, sampled_ys)]

    mean_ys = [np.mean([sy[i] for sy in sampled_ys]) for i in range(n)]
    std_ys = [np.std([sy[i] for sy in sampled_ys]) for i in range(n)]
    mean_scaled_ys = [np.mean([sy[i] for sy in scaled_ys]) for i in range(n)]
    std_scaled_ys = [np.std([sy[i] for sy in scaled_ys]) for i in range(n)]

    bf_l = [float(y-e) if y-e>0.0 else 0.0 for y, e in zip(bf, std_ys)]
    bf_h = [float(y+e) for y, e in zip(bf, std_ys)]
    sf = [float(y) for y in mean_scaled_ys]
    sf_l = [float(y-e) if y-e>0.0 else 0.0 for y, e in zip(mean_scaled_ys, std_scaled_ys)]
    sf_h = [float(y+e) for y, e in zip(mean_scaled_ys, std_scaled_ys)]

    xs = [float(x) for x in xs]

    means = [sum([x*y for x, y in zip(xs, sy)])/sum(sy) if sum(sy) > SMALL_FLOAT else 0.0 for sy in scaled_ys]
    views = [10**m for m in means]
    weighted_views = [10**m * ta for ta, m in zip(target_areas, means)]

    bf_mean = sum([x*y for x, y in zip(xs, bf)]) / sum(bf)
    views_bf = 10**bf_mean * d * sum(bf)
    views_bf_l = 10**bf_mean * d * sum(bf_l)
    views_bf_h = 10**bf_mean * d * sum(bf_h)

    views_data_l = 10**x_low * area
    views_data_h = 10**x_high * area
    views_data_ll = 10**x_low * area_l
    views_data_hh = 10**x_high * area_h
    views_data_mid = 10**(0.5*(x_low + x_high)) * area

    return {
        'bin_id': (x_low+x_high)/2.0,
        'x_low': x_low,
        'x_high': x_high,
        'weight_mean': float(np.mean(target_areas)),
        'weight_std': float(np.std(target_areas)),
        'log_views_mean': float(np.mean(means)),
        'log_views_std': float(np.std(means)),
        'weighted_views_mean': float(np.mean(weighted_views)),
        'weighted_views_std': float(np.std(weighted_views)),
        'views_mean': float(np.mean(views)),
        'views_std': float(np.std(views)),
        'log_views_mean_best_fit': bf_mean,
        'views_best_fit': views_bf,
        'views_best_fit_low': views_bf_l,
        'views_best_fit_high': views_bf_h,
        'views_data_low': views_data_l,
        'views_data_min': views_data_ll,
        'views_data_mid': views_data_mid,
        'views_data_high': views_data_h,
        'views_data_max': views_data_hh,
        'fit_curves': {
            'x': xs,
            'best_fit': bf,
            'best_fit_low': bf_l,
            'best_fit_high': bf_h,
            'scaled_fit': sf,
            'scaled_fit_low': sf_l,
            'scaled_fit_high': sf_h,
        },
    }


def single_tail_bin_estimate(x_low, x_high, func, p, cov, area, area_low, area_high, area_uncert, n=100, n_samples=300, n_extra_bins=1):
    width = x_high - x_low
    d = 1.0 * width / n
    if area is None and area_low is not None and area_high is not None:
        area = 0.5 * (area_low + area_high)
    if area_low is None and area_high is None and area is not None and area_uncert is not None:
        area_low = area - area_uncert if area - area_uncert > 0.0 else 0.0
        area_high = area + area_uncert

    min_x = x_low
    max_x = x_low + width * (n_extra_bins + 1)
    xs_full = np.linspace(min_x, max_x, num=n*(n_extra_bins+1), endpoint=False)
    bf_full = [float(y) for y in func(xs_full, p)]

    target_areas_full = [random.uniform(area_low, area_high) for _ in range(n_samples)] if area_uncert is None else [np.random.normal(area, area_uncert) for _ in range(n_samples)]

    sampled_ps = [np.random.multivariate_normal(p, cov) for _ in range(n_samples)]
    sampled_ys_full = [func(xs_full, pp) for pp in sampled_ps]
    sampled_areas_full = [sum(ys)*d for ys in sampled_ys_full]

    scale_factors = [taf / saf if saf > SMALL_FLOAT else 0.0 for taf, saf in zip(target_areas_full, sampled_areas_full)]
    scaled_ys_full = [sf * syf for sf, syf in zip(scale_factors, sampled_ys_full)]
    scaled_areas_full = [sum(ysf)*d for ysf in scaled_ys_full]

    bins = []

    for ii in range(n_extra_bins+1):
        ll = x_low + width * ii
        hh = x_high + width * ii
        i_l = n * ii
        i_h = n * ii + n
        xs = xs_full[i_l:i_h]
        bf = bf_full[i_l: i_h]

        sampled_ys = [ys[i_l:i_h] for ys in sampled_ys_full]
        scaled_ys = [ys[i_l:i_h] for ys in scaled_ys_full]

        sampled_areas = [sum(ys)*d for ys in sampled_ys]
        target_areas = [taf * (sa / saf) if saf > SMALL_FLOAT else 0.0 for taf, sa, saf in zip(target_areas_full, sampled_areas, sampled_areas_full)]

        mean_ys = [np.mean([sy[i] for sy in sampled_ys]) for i in range(n)]
        std_ys = [np.std([sy[i] for sy in sampled_ys]) for i in range(n)]
        mean_scaled_ys = [np.mean([sy[i] for sy in scaled_ys]) for i in range(n)]
        std_scaled_ys = [np.std([sy[i] for sy in scaled_ys]) for i in range(n)]

        bf_l = [float(y-e) if y-e>0.0 else 0.0 for y, e in zip(bf, std_ys)]
        bf_h = [float(y+e) for y, e in zip(bf, std_ys)]
        sf = [float(y) for y in mean_scaled_ys]
        sf_l = [float(y-e) if y-e>0.0 else 0.0 for y, e in zip(mean_scaled_ys, std_scaled_ys)]
        sf_h = [float(y+e) for y, e in zip(mean_scaled_ys, std_scaled_ys)]

        xs = [float(x) for x in xs]

        means = [sum([x*y for x, y in zip(xs, sy)])/sum(sy) if sum(sy) > 0.0 else 0.0 for sy in scaled_ys]
        views = [10**m if m > 0.0 else 0.0 for m in means]
        weighted_views = [10**m * ta if m > 0.0 else 0.0 for m, ta in zip(means, target_areas)]

        bf_mean = sum([x*y for x, y in zip(xs, bf)]) / sum(bf)
        views_bf = 10**bf_mean * d * sum(bf)
        views_bf_l = 10**bf_mean * d * sum(bf_l)
        views_bf_h = 10**bf_mean * d * sum(bf_h)

        if ii==0:
            views_data_l = 10**x_low * area
            views_data_h = 10**x_high * area
            views_data_ll = 10**x_low * area_low
            views_data_hh = 10**x_high * area_high
            views_data_mid = 10**(0.5*(x_low + x_high)) * area
        else:
            views_data_l = 0.0
            views_data_h = 0.0
            views_data_ll = 0.0
            views_data_hh = 0.0
            views_data_mid = 0.0

        bins.append({
            'bin_id': (ll+hh)/2.0,
            'x_low': ll,
            'x_high': hh,
            'weight_mean': float(np.mean(target_areas)),
            'weight_std': float(np.std(target_areas)),
            'log_views_mean': float(np.mean(means)),
            'log_views_std': float(np.std(means)),
            'weighted_views_mean': float(np.mean(weighted_views)),
            'weighted_views_std': float(np.std(weighted_views)),
            'views_mean': float(np.mean(views)),
            'views_std': float(np.std(views)),

            'views_best_fit': views_bf,
            'views_best_fit_low': views_bf_l,
            'views_best_fit_high': views_bf_h,
            'views_data_low': views_data_l,
            'views_data_min': views_data_ll,
            'views_data_mid': views_data_mid,
            'views_data_high': views_data_h,
            'views_data_max': views_data_hh,

            'fit_curves': {
                'x': xs,
                'best_fit': bf,
                'best_fit_low': bf_l,
                'best_fit_high': bf_h,
                'scaled_fit': sf,
                'scaled_fit_low': sf_l,
                'scaled_fit_high': sf_h,
            },
        })


    return bins


def bin_and_curve_estimates(xs, ys, es, p, cov, func, n=100, n_samples=300, n_extra_bins=1):
    width = xs[1] - xs[0]
    fit_bins = []

    xxs = []
    bf = []
    bf_l = []
    bf_h = []
    sf = []
    sf_l = []
    sf_h = []

    for i in range(len(xs)-1):
        l = i * width
        h = l + width

        x = xs[i]
        y = ys[i] * width # Want to normalize to the areas...
        e = es[i] * width

        y_l = y - e if y - e > 0.0 else 0.0
        y_h = y + e

        cur_fit = single_bin_estimate(l, h, func, p, cov, y, y_l, y_h, None, n=n, n_samples=n_samples)
        cur_curves = cur_fit.pop('fit_curves')

        fit_bins.append(cur_fit)

        xxs += cur_curves['x']
        bf += cur_curves['best_fit']
        bf_l += cur_curves['best_fit_low']
        bf_h += cur_curves['best_fit_high']
        sf += cur_curves['scaled_fit']
        sf_l += cur_curves['scaled_fit_low']
        sf_h += cur_curves['scaled_fit_high']

    tail_l = xs[-1] - width / 2.0
    tail_h = xs[-1] + width / 2.0
    tail_y = ys[-1] * width
    tail_y_l = (ys[-1] - es[-1]) * width if ys[-1] - es[-1] > 0.0 else 0.0
    tail_y_h = (ys[-1] + es[-1]) * width

    tail_bins = single_tail_bin_estimate(tail_l, tail_h, func, p, cov, tail_y, tail_y_l, tail_y_h, None, n=n, n_samples=n_samples, n_extra_bins=n_extra_bins)

    for fit in tail_bins:
        cur_curves = fit.pop('fit_curves')
        fit_bins.append(fit)

        xxs += cur_curves['x']
        bf += cur_curves['best_fit']
        bf_l += cur_curves['best_fit_low']
        bf_h += cur_curves['best_fit_high']
        sf += cur_curves['scaled_fit']
        sf_l += cur_curves['scaled_fit_low']
        sf_h += cur_curves['scaled_fit_high']

    curves = {
        'x': xxs,
        'best_fit': bf,
        'best_fit_low': bf_l,
        'best_fit_high': bf_h,
        'scaled_fit': sf,
        'scaled_fit_low': sf_l,
        'scaled_fit_high': sf_h,
    }
    return fit_bins, curves


def chi2_fit(func, x, y, e, p):
    error_func = lambda p, x, y, e: (func(x, p) - y) / e
    fit, hess_inv, infodict, errmsg, success = sp.optimize.leastsq(error_func, p, args=(np.array(x), np.array(y), np.array(e)), full_output=1)
    res_variance = (error_func(fit, np.array(x), np.array(y), np.array(e))**2).sum()/(len(y)-len(p) )
    cov = hess_inv * res_variance
    return fit, cov


def min_mid_max_views(xs, ys):
    width = xs[1] - xs[0]

    cur_area = sum([y * width for y in ys])
    ysn = [y / cur_area for y in ys]

    min_sum = 0.0
    max_sum = 0.0
    mid_sum = 0.0
    for x, y in zip(xs, ysn):
        l = x - width / 2.0
        h = x + width / 2.0

        min_sum += 10**l * y * width
        max_sum += 10**h * y * width
        mid_sum += 10**x * y * width

    return min_sum, mid_sum, max_sum


def sample_min_max_views(xs, ys, es, n=1000):
    width = xs[1] - xs[0]

    min_sum, mid_sum, max_sum = min_mid_max_views(xs, ys)

    mins = []
    maxs = []
    mids = []
    for _ in range(n):
        ys_samp = [random.uniform(y-e, y+e) for y, e in zip(ys, es)]
        ys_samp = [y if y > 0.0 else 0.0 for y in ys_samp]
        cur_min, cur_mid, cur_max = min_mid_max_views(xs, ys_samp)
        mins.append(cur_min)
        maxs.append(cur_max)
        mids.append(cur_mid)

    return {
        'min_est': min_sum,
        'min_est_avg': float(np.mean(mins)),
        'min_est_std': float(np.std(mins)),
        'mid_est': mid_sum,
        'mid_est_avg': float(np.mean(mids)),
        'mid_est_std': float(np.std(mids)),
        'max_est': max_sum,
        'max_est_avg': float(np.mean(maxs)),
        'max_est_std': float(np.std(maxs)),
    }


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

def estimate_views_from_discrete_distribution(xs, ys, es, n=100, n_samples=1500, n_extra_bins=1, zero_frac=None):
    # Important to make sure the distribution is properly normalized
    # Meaning, area is 1.0. So sum(y*d) = 1.0
    d = xs[1] - xs[0]
    cur_area = sum([y * d for y in ys])
    ys = [y / cur_area for y in ys]
    es = [e / cur_area for e in es]

    # Now, if there are multiple training 0 bins, the fit+projection
    # will get wonky (Due to rounding uncertainty).
    # So truncate the distribution such that there
    # is at most 1 trailing 0 bin.
    while ys[-2]==0.0 and ys[-1]==0.0:
        xs = xs[:-1]
        ys = ys[:-1]
        es = es[:-1]

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
    fit_uncert = np.sqrt(np.diag(fit_cov))

    fit_bins, curves = bin_and_curve_estimates(xs, ys, es, fit_p, fit_cov, func, n=n, n_samples=n_samples, n_extra_bins=n_extra_bins)

    total_views = sum([fb['weighted_views_mean'] for fb in fit_bins])
    total_view_errors = [fb['weighted_views_std'] for fb in fit_bins]
    total_view_uncert = math.sqrt(sum([e*e for e in total_view_errors]))

    sampled_min_max = sample_min_max_views(xs, ys, es, n=n)
    min_max_views = rough_min_max_views(xs, ys, es)

    r = {
        'estimated_views': total_views,
        'estimated_views_uncert': total_view_uncert,

        'min_views': sampled_min_max['min_est'],
        'min_views_avg': sampled_min_max['min_est_avg'],
        'min_views_uncert': sampled_min_max['min_est_std'],
        'mid_views': sampled_min_max['mid_est'],
        'mid_views_avg': sampled_min_max['mid_est_avg'],
        'mid_views_uncert': sampled_min_max['mid_est_std'],
        'max_views': sampled_min_max['max_est'],
        'max_views_avg': sampled_min_max['max_est_avg'],
        'max_views_uncert': sampled_min_max['max_est_std'],

        'fit_parameters': [float(p) for p in fit_p],
        'fit_covariance': [[float(p) for p in fc] for fc in fit_cov],
        'fit_uncertainty': [float(p) for p in fit_uncert],
        'fit_bins': fit_bins,
        'curves': curves,
    }

    r.update(min_max_views)

    estimates = {
        'Binning Uncertainty': (r['dist_binning_min_views'], r['dist_mid_views'], r['dist_binning_max_views']),
        'Rounding Uncertainty': (r['dist_rounding_min_views'], r['dist_mid_views'], r['dist_rounding_max_views']),
        'Total Uncertainty': (r['dist_total_min_views'], r['dist_mid_views'], r['dist_total_max_views']),
        'Sampled Variance': (r['min_views']-r['min_views_uncert'], r['mid_views_avg'], r['max_views']+r['max_views_uncert']),
        'Modeled': (r['estimated_views'] - r['estimated_views_uncert'], r['estimated_views'], r['estimated_views'] + r['estimated_views_uncert']),
    }
    r['estimates'] = estimates

    if zero_frac is not None:
        nzf = 1.0 - zero_frac
        rr = {"{}_with_0".format(k): v * nzf for k, v in r.items() if '_views' in k}
        fit_bins_with_0 = []
        for fb in r['fit_bins']:
            fbw0 = dict(fb)
            for k in ('weight_mean', 'weight_std', 'weighted_views_mean', 'weighted_views_std', 'views_data_low', 'views_data_high', 'views_data_min', 'views_data_max'):
                fbw0[k] = fbw0[k] * nzf
            fit_bins_with_0.append(fbw0)
        rr['fit_bins_with_0'] = fit_bins_with_0
        rr['curves_with_0'] = {
            'x': r['curves']['x'],
            'best_fit': [y*nzf for y in r['curves']['best_fit']],
            'best_fit_low': [y*nzf for y in r['curves']['best_fit_low']],
            'best_fit_high': [y*nzf for y in r['curves']['best_fit_high']],
            'scaled_fit': [y*nzf for y in r['curves']['scaled_fit']],
            'scaled_fit_low': [y*nzf for y in r['curves']['scaled_fit_low']],
            'scaled_fit_high': [y*nzf for y in r['curves']['scaled_fit_high']],
        }
        rr['estimates_with_0'] = dict([
            (k, [vv * nzf for vv in v]) for k, v in r['estimates'].items()
        ])
        r.update(rr)


    return r


def plot_estimation_from_discrete_distribution(estimates, fit_bins, curves, filetag, label=None):
    fit = {
        'fit_bins': fit_bins,
        'curves': curves,
        'estimates': estimates,
    }
    if label is None:
        label = filetag.split('/')[-1].strip()

    estimates_uncert = {k: (v[1], (v[1]-v[0], v[2]-v[1])) for k, v in estimates.items()}

    xs = [fb['bin_id'] for fb in fit['fit_bins']]
    ys = [fb['weight_mean'] for fb in fit['fit_bins']]
    es = [fb['weight_std'] for fb in fit['fit_bins']]
    es2x = [2*fb['weight_std'] for fb in fit['fit_bins']]
    ys_l = [fb['weight_mean']-fb['weight_std'] for fb in fit['fit_bins']]
    ys_l = [y if y>0.0 else 0.0 for y in ys_l]
    ys_h = [fb['weight_mean']+fb['weight_std'] for fb in fit['fit_bins']]

    wvses2x = [fb['weighted_views_std']*2 for fb in fit['fit_bins']]
    wvs = [fb['weighted_views_mean'] for fb in fit['fit_bins']]
    wvs_l = [fb['weighted_views_mean']-fb['weighted_views_std'] for fb in fit['fit_bins']]
    wvs_l = [wv if wv>0.0 else 0.0 for wv in wvs_l]
    wvs_h = [fb['weighted_views_mean']+fb['weighted_views_std'] for fb in fit['fit_bins']]

    bfvs_l = [f['views_best_fit_low'] for f in fit['fit_bins']]
    bfvs_h = [f['views_best_fit_high'] for f in fit['fit_bins']]
    dvs_l = [f['views_data_low'] for f in fit['fit_bins']]
    dvs_h = [f['views_data_high'] for f in fit['fit_bins']]
    dvs_ll = [f['views_data_min'] for f in fit['fit_bins']]
    dvs_hh = [f['views_data_max'] for f in fit['fit_bins']]

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
        show=False,
        save="{}_Best_Fit.png".format(filetag),
    )
    plot.plot([
        ('bar', xs, ys, {'color': 'tab:blue', 'width': 0.9, 'label': 'Provided Data', 'alpha': 0.6}),
        ('fill_between', curves['x'], curves['scaled_fit_low'], curves['scaled_fit_high'], {'alpha': 0.60, 'color': 'tab:orange', 'label': None}),
        ('plot', curves['x'], curves['scaled_fit'], {'color': 'tab:orange', 'label': 'Best Fit'}),
        ('errorbar', xs, ys, {'color': 'tab:blue', 'yerr': es, 'marker': 'o', 'linestyle': ''}),
        ],
        title="Modeled Views Distribution [{}]".format(label),
        xlabel="Log10(Views)",
        ylabel="Distribution (a.u.)",
        xlim=[0.0, None],
        ylim=[0.0, None],
        show=False,
        save="{}_Scaled_Fit.png".format(filetag),
    )
    plot.plot([
        ('bar', xs, wvses2x, {'color': 'tab:orange', 'width': 0.85, 'bottom': wvs_l, 'label': 'Model', 'alpha': 0.6})
        ],
        title="Views Per Bin [{}]".format(label),
        xlabel="Log10(Views) Bin",
        ylabel="Views",
        xlim=[0.0, None],
        ylim=[0.0, None],
        show=False,
        save="{}_Modeled_Views_Per_Bin.png".format(filetag),
    )
    plot.plot([
        ('fill_between', xs, bfvs_l, bfvs_h, {'color': 'tab:orange', 'alpha': 0.5, 'label': 'Best Fit'}),
        ('fill_between', x_dvs, dvs_l, dvs_h, {'color': 'tab:blue', 'alpha': 0.5, 'label': 'Data'}),],
        title="Views Per Bin [{}]".format(label),
        xlabel="Log10(Views) Bin",
        ylabel="Views",
        xlim=[0.0, None],
        ylim=[0.0, None],
        show=False,
        save="{}_Best_Fit_Views_Per_Bin.png".format(filetag),
    )
    plot.plot([
        ('fill_between', x_dvsmm, dvs_ll, dvs_hh, {'color': 'tab:blue', 'alpha': 0.3, 'label': 'Max Data Range'}),
        ('fill_between', x_dvs, dvs_l, dvs_h, {'color': 'tab:blue', 'alpha': 0.5, 'label': 'Data'}),
        ('plot', xs, wvs, {'color': 'tab:orange', 'label': 'Model', 'marker': 'o', 'linestyle': '-'}),
        ],
        title="Views Per Bin [{}]".format(label),
        xlabel="Log10(Views) Bin",
        ylabel="Views",
        xlim=[0.0, None],
        ylim=[0.0, None],
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
    show=False,
    save="{}_View_Estimates.png".format(filetag))
