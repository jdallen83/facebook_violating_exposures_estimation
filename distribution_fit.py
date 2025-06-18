import scipy as sp
import math
from jplot import jplot as jp
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
        ys_samp = [random.uniform(y-e if y-e > 0.0 else 0.0, y+e) for y, e in zip(ys, es)]
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


def estimate_views_from_discrete_distribution(xs, ys, es, n=100, n_samples=1500, n_extra_bins=1):
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

    return {
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


xs = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
]

ys = [
    11/100.0,
    11/100.0,
    6/100.0,
    2/100.0,
    1/100.0,
]

xs = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
]

ys = [
    24/100.0,
    8/100.0,
    5/100.0,
    2/100.0,
    0/100.0,
]

es = [0.5/100.0 for y in ys]


xs = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
]

cs = [
    2395918,
    895363,
    611703,
    244250,
    120142,
]

total_counts = sum(cs)

ys = [c * 1.0 / total_counts for c in cs]
es = [math.sqrt(c) / total_counts for c in cs]


# TikTok
xs = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
    5.5,
    6.5,
]

ys = [
    1.6,
    7.8,
    5.5,
    1.0,
    0.5,
    0.2,
    0.0,
]

es = [0.05 for y in ys]
ys = [y/100.0 for y in ys]
es = [e/100.0 for e in es]




xs = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
]

cs = [
    2395918,
    895363,
    611703,
    244250,
    120142,
]

total_counts = sum(cs)

ys = [c * 1.0 / total_counts for c in cs]
es = [math.sqrt(c) / total_counts for c in cs]



ys = [
    11/100.0,
    11/100.0,
    6/100.0,
    2/100.0,
    1/100.0,
]

xs = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
]

ys = [
    24/100.0,
    8/100.0,
    5/100.0,
    2/100.0,
    0/100.0,
]

ys = [
    16/100.0,
    7/100.0,
    4/100.0,
    2/100.0,
    0/100.0,
]


es = [0.5/100.0 for y in ys]





area = sum(ys) * (xs[1]-xs[0])
ys = [y / area for y in ys]
es = [e / area for e in es]
print(xs)
print(ys)
print(es)


fit = estimate_views_from_discrete_distribution(xs, ys, es, n=100, n_samples=20000, n_extra_bins=1)
fit_bins = fit.pop('fit_bins')
curves = fit.pop('curves')
total_views = fit['estimated_views']
total_views_uncert = fit['estimated_views_uncert']

##

for k, v in fit.items():
    print("{}: {}".format(k, v))

##

print(xs)
print(ys, sum(ys))
print(es)

##
for fb in fit_bins:
    print("{}\t{}\t{}\t{}".format("%3.1f" % fb['bin_id'], "%3.8f" % fb['weight_mean'], fb['weighted_views_mean'], fb['weighted_views_std']))
##

##
jp.plot_fit_with_uncert(xs, ys, es, curves['x'], curves['best_fit'], curves['best_fit_low'], curves['best_fit_high'], show=True, bar=True)
##
jp.plot_fit_with_uncert(xs, ys, es, curves['x'], curves['scaled_fit'], curves['scaled_fit_low'], curves['scaled_fit_high'], show=True, bar=True)
##
jp.plot_fit_with_uncert(xs, ys, es, xs_bf, bf_s, bf_s, bf_s, show=True, bar=True)
##


print(curves['x'])

##
for fb in fit_bins:
    rnd_b = [float("%.8f" % b) for b in fb[:-2]]
    print("\t".join("{}".format(b) for b in rnd_b))
##
print(total_views, total_view_uncert)
print(sum([b[3] for b in fit_bins]))
print(sum(ys))
print(fit_p)
##

u = -3.0
s = 2.35
print(u, s)
samps = [s for s in np.random.normal(u, s, size=10000000) if s >= 0.0]
views = np.array([round(10**s) for s in samps if int(10**s)>0])

n_bins = 5
bin_width = 1.0

sampled_dist = []
xs_example = []
for i in range(n_bins):
    xs_example.append(i * bin_width + bin_width/2.0)
    sampled_dist.append(0)

for v in views:
    assigned = False
    s = math.log10(v)
    for i, x in enumerate(xs_example):
        if s >= x - bin_width/2.0 and s < x + bin_width/2.0:
            assigned = True
            sampled_dist[i] += 1
    if not assigned:
        sampled_dist[-1] += 1

area = sum(sampled_dist) * bin_width

normed_dist = [1.0 * s / area for s in sampled_dist]
normed_dist_error = [math.sqrt(max([s, 10])) / area for s in sampled_dist]
rounded_dist = [round(s, 2) for s in normed_dist]
es_example = [max([0.005, e]) for e in normed_dist_error]

print(np.mean(samps), np.std(samps))
print(min(samps), max(samps))
print(np.mean(views), np.std(views))
print(sampled_dist)
print(xs_example)
print(normed_dist)
print(rounded_dist, sum(rounded_dist))
print(es_example)

xs = xs_example
ys = rounded_dist
es = es_example
##
print(xs)
print(ys)
print(es)

print(sum(ys) * bin_width)
total_views, total_view_uncert, fit_bins, fit_p, fit_cov, fit_uncert, (xs_bf, bf, bf_l, bf_h, x_s, bf_s, bf_s_l, bf_s_h) = estimate_views_from_discrete_distribution(xs, ys, es, n=200, n_samples=5000, n_extra_bins=7)


##

print(total_views, total_view_uncert)
print(fit_p)
print(fit_uncert)

##

ys = [y / 2.0 for y in ys]
##
print(sum(ys) * bin_width)
##
jp.plot_fit_with_uncert(xs, ys, es, xs_bf, bf_s, bf_s_l, bf_s_h, show=True, bar=True)
##
jp.plot_fit_with_uncert(xs, ys, es, xs_bf, bf, bf_l, bf_h, show=True, bar=True)
##
jp.plot_fit_with_uncert(xs, ys, es, xs_bf, bf_s, bf_s, bf_s, show=True, bar=True)
##

for fb in fit_bins:
    print(fb[:-2])
##


xs = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
]

cs = [
    2395918,
    895363,
    611703,
    244250,
    120142,
]

total_counts = sum(cs)

ys = [c * 1.0 / total_counts for c in cs]
es = [math.sqrt(c) / total_counts for c in cs]



total_views, total_view_uncert, fit_bins, fit_p, fit_cov, fit_uncert, \
    (xs_bf, bf, bf_l, bf_h, x_s, bf_s, bf_s_l, bf_s_h) \
    = estimate_views_from_discrete_distribution(xs, ys, es, n=200, n_samples=5000, n_extra_bins=2)
fit_views = total_views
fit_views_uncert = total_view_uncert

fits = []
for i in range(20):
    sampled_ys = [
        random.uniform(y-e if y-e > 0.0 else 0.0, y+e) for y, e in zip(ys, es)
    ]
    area = sum(sampled_ys) * (xs[1]-xs[0])
    sampled_ys = [y / area for y in sampled_ys]
    sampled_es = [e / area for e in es]

    total_views, total_view_uncert, fit_bins, fit_p, fit_cov, fit_uncert, \
        (xs_bf, bf, bf_l, bf_h, x_s, bf_s, bf_s_l, bf_s_h) \
        = estimate_views_from_discrete_distribution(xs, sampled_ys, sampled_es, n=200, n_samples=5000, n_extra_bins=2)
    print(round(total_views, 2), round(total_view_uncert, 2), fit_p, round(total_views*0.39, 2), round(total_view_uncert*0.39, 2))

    fits.append(total_views)
#
print(fit_views, fit_views_uncert)
print(float(np.mean(fits)), float(np.std(fits)))
min_est, mid_est, max_est = get_min_max_views(xs, ys, es)

print(min_est)
print(mid_est)
print(max_est)
