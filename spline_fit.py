import scipy as sp
import numpy as np
from facebook_violating_exposures_estimation import plot as plot
import facebook_violating_exposures_estimation.distribution_fit as dist_fit

import random

# Basic idea... use spline interpolation for everything.
# Can clamp so that derivative is 0 and the endpoints
# Can control where you make the endpoint based on n_extra bins etc...



xs_fb = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
]

ys_bullying = [
    11/100.0,
    11/100.0,
    6/100.0,
    2/100.0,
    1/100.0,
]

ys_violence = [
    24/100.0,
    8/100.0,
    5/100.0,
    2/100.0,
    0/100.0,
]

ys_hate = [
    16/100.0,
    7/100.0,
    4/100.0,
    2/100.0,
    0/100.0,
]

es_fb = [0.5/100.0 for y in xs_fb]

d = xs_fb[1] - xs_fb[0]

area_bullying = sum([y * d for y in ys_bullying])
ys_bullying = [y / area_bullying for y in ys_bullying]
es_bullying = [e / area_bullying for e in es_fb]

area_violence = sum([y * d for y in ys_violence])
ys_violence = [y / area_violence for y in ys_violence]
es_violence = [e / area_violence for e in es_fb]

area_hate = sum([y * d for y in ys_hate])
ys_hate = [y / area_hate for y in ys_hate]
es_hate = [e / area_hate for e in es_fb]



# ------------

xs = xs_fb
ys = ys_bullying
es = es_bullying
zero_frac = 1.0 - area_bullying


# ---------------


# --
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


    xs_spl.insert(0, xs_spl[0] - width / 2.0)
    ys_spl.insert(0, ys_spl[0])

    xs_spl.append(xs_spl[-1] + (n_extra_bins + 0.5) * width)
    ys_spl.append(0.0)

    spl = sp.interpolate.make_interp_spline(xs_spl, ys_spl, bc_type="clamped")

    x_spl = np.linspace(0, xs_spl[-1], num=n * (len(xs) + n_extra_bins), endpoint=False)
    y_spl = np.maximum(spl(x_spl), 0.0)

    x_spl = list(x_spl)
    y_spl = list(y_spl)

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
        views = [f['fit_bins'][i]['views'] for f in fits]

        mb = {
            'x': float(np.mean(x)),
            'y': float(np.mean(y)),
            'y_std': float(np.std(y)),
            'target_weight': float(np.mean(target_weights)),
            'target_weight_std': float(np.std(target_weights)),
            'weight': float(np.mean(weights)),
            'weight_std': float(np.std(weights)),
            'x_mean': float(np.std(x_means)),
            'x_mean_std': float(np.std(x_means)),
            'views': float(np.mean(views)),
            'views_std': float(np.std(views)),
        }

        if zero_frac is not None:
            for k in mb.keys() if k not in ('x', 'x_mean', 'x_mean_std'):
                mb[k + '_with_0'] = mb[k] * (1.0 - zero_frac)

        mean_bins.append(mb)

    pivot_bins = {k: [b[k] for b in mean_bins] for k in mean_bins[0].keys()}
    views = sum(pivot_bins['views'])
    views_uncert = float(np.sqrt(sum([v**2 for v in pivot_bins['views_std']])))

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
        for k in curves.keys() if k not in ('x'):
            curves[k + '_with_0'] = [v * (1.0 - zero_frac) for v in curves[k]]

    return mean_bins, curves, (views, views_uncert), (views * (1.0 - zero_frac) if zero_frac is not None else None, views_uncert * (1.0 - zero_frac) if zero_frac is not None else None)


def estimate_views_in_bins_with_splines(xs, ys, es, n_extra_bins=1, n=100, n_samples=10000, zero_frac=None):
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

    fit_bins, curves, estimate = average_sampled_estimates(fits, zero_frac=zero_frac)
    return {
        'data': fit_data,
        'fit_bins': fit_bins,
        'curves': curves,
        'estimate': estimate,
    }


fit_spline = estimate_views_in_bins_with_splines(xs, ys, es, n_extra_bins=0, n=100, n_samples=10000, zero_frac=zero_frac)





fit = dist_fit.estimate_views_from_discrete_distribution(xs, ys, es, n=100, n_samples=10000, n_extra_bins=1, zero_frac=0.0)

##

for k, v in fit['estimates'].items():
    print(k, v)

##

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

##

plot.plot([
        ('fill_between', x_bf, best_fit_l, best_fit_h, {'color': 'tab:orange', 'alpha': 0.5}),
        ('plot', x_bf, best_fit, {'color': 'tab:orange', 'marker': '', 'linestyle': '-'}),
        ('errorbar', xs, ys, {'color': 'tab:blue', 'marker': 'o', 'linestyle': '', 'yerr': es}),
    ],
    show=True,
    xlim=[0.0, None],
    ylim=[0.0, None],

)
##
plot.plot([
        ('fill_between', x_bf, best_fit_scaled_l, best_fit_scaled_h, {'color': 'tab:orange', 'alpha': 0.5}),
        ('plot', x_bf, best_fit_scaled, {'color': 'tab:orange', 'marker': '', 'linestyle': '-'}),
        ('errorbar', xs, ys, {'color': 'tab:blue', 'marker': 'o', 'linestyle': '', 'yerr': es}),
    ],
    show=True,
    xlim=[0.0, None],
    ylim=[0.0, None],
)

##

fit_curves = fit['curves']
plot.plot([
        ('fill_between', fit_curves['x'], fit_curves['best_fit_low'], fit_curves['best_fit_high'], {'color': 'tab:orange', 'alpha': 0.5}),
        ('plot', fit_curves['x'], fit_curves['best_fit'], {'color': 'tab:orange', 'marker': '', 'linestyle': '-'}),
        ('errorbar', xs, ys, {'color': 'tab:blue', 'marker': 'o', 'linestyle': '', 'yerr': es}),
    ],
    show=True,
    xlim=[0.0, None],
    ylim=[0.0, None],
)

##

fit_curves = fit['curves']
plot.plot([
        ('fill_between', fit_curves['x'], fit_curves['scaled_fit_low'], fit_curves['scaled_fit_high'], {'color': 'tab:orange', 'alpha': 0.5}),
        ('plot', fit_curves['x'], fit_curves['scaled_fit'], {'color': 'tab:orange', 'marker': '', 'linestyle': '-'}),
        ('errorbar', xs, ys, {'color': 'tab:blue', 'marker': 'o', 'linestyle': '', 'yerr': es}),
    ],
    show=True,
    xlim=[0.0, None],
    ylim=[0.0, None],
)


##

plot.plot([
        ('plot', x_spl, y_spl, {'color': 'tab:orange', 'marker': '', 'linestyle': '-'}),
        ('plot', xs, ys, {'color': 'tab:blue', 'linestyle': '', 'marker': 'o'})
    ],
)
plot.pl.show()

##

plot.plot([
        ('plot', x_spl, y_spl_scaled, {'color': 'tab:orange', 'marker': '', 'linestyle': '-'}),
        ('plot', xs, ys, {'color': 'tab:blue', 'linestyle': '', 'marker': 'o'})
    ],
)
plot.pl.show()

