import scipy as sp
import math
from facebook_violating_exposures_estimation import plot as plot
import numpy as np
import random
import json


SMALL_FLOAT = 1E-280 # Have to define a minimum float size to round to 0, for division by 0 issues


def sample_distribution(xs, ys, es, n=1):
    sampled_ys = [[max([0.0, random.uniform(y-e, y+e)]) for y, e in zip(ys, es)] for i in range(n)]
    return sampled_ys[0] if n==1 else sampled_ys


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

        bf_mean = sum([x*y for x, y in zip(xs, bf)]) / sum(bf) if sum(bf) > 0.0 else xs[0]
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


def single_estimate_views_in_bins_with_spline(xs, ys, n_extra_bins=1, n=100):
    h = list(zip(xs, ys))
    h = sorted(h, key=lambda x: x[0])
    xs = [x for x, y in h]
    ys = [y for x, y in h]

    width = xs[1] - xs[0]
    cur_area = sum([y * width for y in ys])
    ys = [y / cur_area for y in ys]



    xs_spl = list(xs)
    ys_spl = list(ys)


    r = ((ys_spl[0] / ys_spl[1]) - 1.0) / 2.0 + 1.0

    xs_spl.insert(0, xs_spl[0] - width / 2.0)
    ys_spl.insert(0, ys_spl[0] * r)

    xs_spl.append(xs_spl[-1] + (n_extra_bins + 0.5) * width)
    ys_spl.append(0.0)

    x_spl = []
    y_spl = []
    spl = None

    redo = True
    while redo:
        redo = False

        spl = sp.interpolate.make_interp_spline(xs_spl, ys_spl, bc_type="clamped", k=3)

        x_spl = np.linspace(0, xs_spl[-1], num=n * (len(xs) + n_extra_bins), endpoint=False)
        y_spl = spl(x_spl)

        for x, y in zip(x_spl, y_spl):
            if y < 0 and x < xs_spl[-2]:
                print("Found bad at", x, y)
                redo = True

                x_l = max([xx for xx in xs_spl if xx <= x])
                x_r = min([xx for xx in xs_spl if xx > x])
                y_l = [yy for xx, yy in zip(xs_spl, ys_spl) if xx==x_l][0]
                y_r = [yy for xx, yy in zip(xs_spl, ys_spl) if xx==x_l][0]
                i = [ii for ii, xx in enumerate(xs_spl) if xx==x_r][0]
                xs_spl.insert(i, 0.5 * (x_l + x_r))
                ys_spl.insert(i, 0.5 * (y_l + y_r))
                break

        y_spl = np.maximum(y_spl, 0.0)

    x_spl = list([float(x) for x in x_spl])
    y_spl = list([float(y) for y in y_spl])

    y_spl_scaled = []

    #plot.plot([
    #        ('plot', x_spl, y_spl, {'color': 'tab:orange', 'marker': '', 'linestyle': '-'}),
    #        ('plot', xs, ys, {'color': 'tab:blue', 'linestyle': '', 'marker': 'o'})
    #    ],
    #)
    #plot.pl.show()


    bins = []
    d = width * 1.0 / n

    fit_bins = []

    for i in range(len(xs)+n_extra_bins):
        x_cspl = np.array(list(x_spl[i * n:(i+1) * n]))
        y_cspl = np.array(list(y_spl[i * n:(i+1) * n]))

        if i < len(xs) - 1:
            xl = xs[i] - width / 2.0
            xh = xs[i] + width / 2.0
            x = xs[i]
            y = ys[i]
            target_area = y * width
            cur_area = sum(y_cspl*d)
        else:
            xl = xs[-1] - width / 2.0 + (i - len(xs) + 1) * width
            xh = xs[-1] + width / 2.0 + (i - len(xs) + 1) * width
            x = (xl + xh) / 2.0

            if i == len(xs) - 1:
                y = ys[i]
            else:
                y = 0.0

            target_full_area = ys[-1] * width

            full_y = np.array(list(y_spl[(len(xs)-1)*n:]))

            full_y_area = sum(full_y) * d
            cur_area = sum(y_cspl) * d

            target_area = target_full_area * cur_area / full_y_area


        scale_factor = target_area / cur_area if cur_area > 0.0 else 0.0

        y_cspl_scale = y_cspl * scale_factor
        y_spl_scaled = y_spl_scaled + [float(f) for f in y_cspl_scale]

        weight = sum(y_cspl_scale*d)

        x_mean = sum(y_cspl_scale * x_cspl * d) / weight if weight > 0.0 else 0.0

        views = 10**x_mean * weight
        views_l = 10**xl * weight
        views_h = 10**xh * weight

        fit_bins.append({
            'x': float(x),
            'y': float(y),
            'target_weight': float(target_area),
            'weight': float(weight),
            'x_mean': float(x_mean),
            'views': float(views),
            'views_l': float(views_l),
            'views_h': float(views_h),
        })

    return {
        'fit_bins': fit_bins,
        'curves': {
            'x': x_spl,
            'y': y_spl,
            'y_scaled': y_spl_scaled,
        }
    }


def average_sampled_estimates(fits, zero_frac=None):
    mean_bins = []
    for i in range(len(fits[0]['fit_bins'])):
        x = [f['fit_bins'][i]['x'] for f in fits]
        y = [f['fit_bins'][i]['y'] for f in fits]
        target_weights = [f['fit_bins'][i]['target_weight'] for f in fits]
        weights = [f['fit_bins'][i]['weight'] for f in fits]
        x_means = [f['fit_bins'][i]['x_mean'] for f in fits]
        weighted_views = [f['fit_bins'][i]['views'] for f in fits]
        views = [10**f['fit_bins'][i]['x_mean'] for f in fits]

        width = fits[0]['fit_bins'][1]['x'] - fits[0]['fit_bins'][0]['x']
        x_low = fits[0]['fit_bins'][i]['x'] - width / 2.0
        x_high = fits[0]['fit_bins'][i]['x'] + width / 2.0
        x_mid = fits[0]['fit_bins'][i]['x']

        mb = {
            'bin_id': float(np.mean(x)),
            'weight_mean': float(np.mean(target_weights)),
            'weight_std': float(np.std(target_weights)),
            'log_views_mean': float(np.std(x_means)),
            'log_views_std': float(np.std(x_means)),
            'weighted_views_mean': float(np.mean(weighted_views)),
            'weighted_views_std': float(np.std(weighted_views)),
            'views_mean': float(np.mean(views)),
            'views_std': float(np.std(views)),
            'views_data_low': 10**x_low * float(np.mean(target_weights)),
            'views_data_high': 10**x_high * float(np.mean(target_weights)),
            'views_data_mid': 10**x_mid * float(np.mean(target_weights)),
            'views_data_min': 10**x_low * max([0.0, float(np.mean(target_weights)) - float(np.std(target_weights))]),
            'views_data_max': 10**x_high * (float(np.mean(target_weights)) + float(np.std(target_weights))),
            'views_best_fit_low': float(np.mean(weighted_views)) - float(np.std(weighted_views)),
            'views_best_fit_high': float(np.mean(weighted_views)) + float(np.std(weighted_views)),
        }

        if zero_frac is not None:
            for k in list((k for k in mb.keys() if k not in ('bin_id', 'x', 'x_mean', 'x_mean_std'))):
                mb[k + '_with_0'] = mb[k] * (1.0 - zero_frac)

        mean_bins.append(mb)

    pivot_bins = {k: [b[k] for b in mean_bins] for k in mean_bins[0].keys()}
    views = sum(pivot_bins['weighted_views_mean'])
    views_uncert = float(np.sqrt(sum([v**2 for v in pivot_bins['weighted_views_std']])))

    curves = [f['curves'] for f in fits]
    curves_y = [c['y'] for c in curves]
    curves_y_scaled = [c['y_scaled'] for c in curves]

    y_samples = [[yy[i] for yy in curves_y] for i in range(len(curves[0]['y']))]
    y_scaled_samples = [[yy[i] for yy in curves_y_scaled] for i in range(len(curves[0]['y_scaled']))]

    y_mean = [float(np.mean(yy)) for yy in y_samples]
    y_std = [float(np.std(yy)) for yy in y_samples]

    best_fit = list(y_mean)
    best_fit_h = [y + e for y, e in zip(y_mean, y_std)]
    best_fit_l = [y - e if y - e > 0.0 else 0.0 for y, e in zip(y_mean, y_std)]


    y_scaled_mean = [float(np.mean(yy)) for yy in y_scaled_samples]
    y_scaled_std = [float(np.std(yy)) for yy in y_scaled_samples]

    best_fit_scaled = list(y_scaled_mean)
    best_fit_scaled_h = [y + e for y, e in zip(y_scaled_mean, y_scaled_std)]
    best_fit_scaled_l = [y - e if y - e > 0.0 else 0.0 for y, e in zip(y_scaled_mean, y_scaled_std)]

    x_bf = curves[0]['x']

    curves = {
        'x': x_bf,
        'best_fit': best_fit,
        'best_fit_low': best_fit_l,
        'best_fit_high': best_fit_h,
        'scaled_fit': best_fit_scaled,
        'scaled_fit_low': best_fit_scaled_l,
        'scaled_fit_high': best_fit_scaled_h,
    }
    if zero_frac is not None:
        for k in list((k for k in curves.keys() if k not in ('x'))):
            curves[k + '_with_0'] = [v * (1.0 - zero_frac) for v in curves[k]]

    return mean_bins, curves, (views, views_uncert), (views * (1.0 - zero_frac) if zero_frac is not None else None, views_uncert * (1.0 - zero_frac) if zero_frac is not None else None)


def estimate_views_with_splines(xs, ys, es, n_extra_bins=1, n=100, n_samples=10000, zero_frac=None):
    # Fit can get very inaccurate for distributions with 2 trailing 0 bins
    # This is due to the fit spline oscialting around 0, which can put a lot of weight in very high bins
    # Best option is to truncate multiple trailing 0s

    cont = True
    while cont:
        if ys[-1] <= 0.0 and ys[-2] <= 0.0:
            xs = xs[:-1]
            ys = ys[:-1]
            es = es[:-1]
        else:
            cont = False

    fit_data = single_estimate_views_in_bins_with_spline(xs, ys, n_extra_bins=n_extra_bins, n=n)

    fits = []
    for i in range(n_samples):
        cur_x = []
        cur_y = []
        for x, y, e in zip(xs, ys, es):
            cur_x.append(x)
            yy = random.uniform(y - e, y + e)
            yy = yy if yy > 0.0 else 0.0
            cur_y.append(yy)
        fits.append(single_estimate_views_in_bins_with_spline(cur_x, cur_y, n_extra_bins=n_extra_bins, n=n))

    fit_bins, curves, estimate, estimate_with_0 = average_sampled_estimates(fits, zero_frac=zero_frac)
    return {
        'data': fit_data,
        'fit_bins': fit_bins,
        'curves': curves,
        'estimate': estimate,
        'estimate_with_0': estimate_with_0,
    }


def estimate_views_from_discrete_distribution(xs, ys, es, n=100, n_samples=1500, n_extra_bins=1, zero_frac=None, func=normal_distribution, force_simple_fit=False):
    # Important to make sure the distribution is properly normalized
    # Meaning, area is 1.0. So sum(y*d) = 1.0
    h = list(zip(xs, ys, es))
    h = sorted(h, key=lambda x: x[0])
    xs = [x for x, y, e in h]
    ys = [y for x, y, e in h]
    es = [e for x, y, e in h]

    data_original = {
        'x': list(xs),
        'y': list(ys),
        'es': list(es),
        'zero_frac': zero_frac,
    }

    d = xs[1] - xs[0]
    cur_area = sum([y * d for y in ys])
    ys = [y / cur_area for y in ys]
    es = [e / cur_area for e in es]

    data_normalized = {
        'x': list(xs),
        'y': list(ys),
        'es': list(es),
        'zero_frac': zero_frac,
    }

    fit_spline = estimate_views_with_splines(xs, ys, es, n_extra_bins=n_extra_bins, n=n, n_samples=n_samples, zero_frac=zero_frac)

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

    if (ys[1] < 0.66 * ys[0] and ys[0]==max(ys) and func==normal_distribution) or force_simple_fit:
        # Use the normal_distribution_tail case...
        func = normal_distribution_tail
        p0 = [2.0, ys[0]*3.0]
    elif func==normal_distribution:
        # Just use normal distribution...
        func = normal_distribution
        p0 = [1.0, 2.0, 1.0]
    else:
        func = polynomial_distribution
        p0 = P0_POLYNOMIAL_DISTRIBUTION#[1.5, 0.1, 0.0, 0.0]#, 0.0015, 0.0015]

    func_fit = lambda x, p: bin_sampled_distribution(x, p, func, n=n, n_extra_bins=n_extra_bins)

    fit_p, fit_cov = chi2_fit(func_fit, xs, fit_ys, fit_es, p0)
    fit_uncert = np.sqrt(np.diag(fit_cov))

    fit_bins, curves = bin_and_curve_estimates(xs, ys, es, fit_p, fit_cov, func, n=n, n_samples=n_samples, n_extra_bins=n_extra_bins)

    total_views = sum([fb['weighted_views_mean'] for fb in fit_bins])
    total_view_errors = [fb['weighted_views_std'] for fb in fit_bins]
    total_view_uncert = math.sqrt(sum([e*e for e in total_view_errors]))

    sampled_min_max = sample_min_max_views(xs, ys, es, n=n)
    min_max_views = rough_min_max_views(xs, ys, es)

    #for k in fit_spline['curves'].keys() if k not in ('x') and 'with_0' not in k:
    #    curves[k + '_spline'] = fit_spline['curves'][k]
    r = {
        'estimated_views': total_views,
        'estimated_views_uncert': total_view_uncert,

        'estimated_views_spline': fit_spline['estimate'][0],
        'estimated_views_spline_uncert': fit_spline['estimate'][1],

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
        'fit_bins_spline':
            [{k: v for k, v in fb.items() if 'with_0' not in k} for fb in fit_spline['fit_bins']],
        'fit_bins_spline_with_0':
            [{k.replace('_with_0', ''): v for k, v in fb.items() if '_with_0' in k or k=='bin_id'} for fb in fit_spline['fit_bins']],
        'curves': curves,
        'curves_spline': fit_spline['curves'],
        'data': data_original,
        'data_normalized': data_normalized,
    }

    r.update(min_max_views)

    estimates = {
        'Binning Uncertainty': (r['dist_binning_min_views'], r['dist_mid_views'], r['dist_binning_max_views']),
        'Rounding Uncertainty': (r['dist_rounding_min_views'], r['dist_mid_views'], r['dist_rounding_max_views']),
        'Total Uncertainty': (r['dist_total_min_views'], r['dist_mid_views'], r['dist_total_max_views']),
        'Sampled Variance': (r['min_views']-r['min_views_uncert'], r['mid_views_avg'], r['max_views']+r['max_views_uncert']),
        'Modeled (Normal)': (
            r['estimated_views'] - r['estimated_views_uncert'],
            r['estimated_views'],
            r['estimated_views'] + r['estimated_views_uncert']
        ),
        'Modeled (Spline)': (
            fit_spline['estimate'][0] - fit_spline['estimate'][1],
            fit_spline['estimate'][0],
            fit_spline['estimate'][0] + fit_spline['estimate'][1],
        )
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
        rr['curves_spline_with_0'] = {
            'x': r['curves_spline']['x'],
            'best_fit': [y*nzf for y in r['curves_spline']['best_fit']],
            'best_fit_low': [y*nzf for y in r['curves_spline']['best_fit_low']],
            'best_fit_high': [y*nzf for y in r['curves_spline']['best_fit_high']],
            'scaled_fit': [y*nzf for y in r['curves_spline']['scaled_fit']],
            'scaled_fit_low': [y*nzf for y in r['curves_spline']['scaled_fit_low']],
            'scaled_fit_high': [y*nzf for y in r['curves_spline']['scaled_fit_high']],
        }
        rr['estimates_with_0'] = dict([
            (k, [vv * nzf for vv in v]) for k, v in r['estimates'].items()
        ])
        r.update(rr)


    return r


def plot_estimation_from_discrete_distribution(estimates, fit_bins, curves, filetag, label=None, style=None):
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

    if 'views_best_fit_low' in fit['fit_bins'][0]:
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
        style=style,
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
        style=style,
        show=False,
        save="{}_Modeled_Fit.png".format(filetag),
    )
    plot.plot([
        ('bar', xs, wvses2x, {'color': 'tab:orange', 'width': 0.85, 'bottom': wvs_l, 'label': 'Model', 'alpha': 0.6})
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
    if 'views_best_fit_low' in fit['fit_bins'][0]:
        plot.plot([
            ('fill_between', xs, bfvs_l, bfvs_h, {'color': 'tab:orange', 'alpha': 0.5, 'label': 'Best Fit'}),
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
            ('plot', xs, wvs, {'color': 'tab:orange', 'label': 'Model', 'marker': 'o', 'linestyle': '-'}),
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
