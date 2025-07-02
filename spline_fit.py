import scipy as sp
import numpy as np
from facebook_violating_exposures_estimation import plot as plot


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



# ---------------


n_extra_bins = 0
n = 100
n_samples = 10000


# --

h = list(zip(xs, ys, es))
h = sorted(h, key=lambda x: x[0])
xs = [x for x, y, e in h]
ys = [y for x, y, e in h]
es = [e for x, y, e in h]

width = xs[1] - xs[0]
cur_area = sum([y * width for y in ys])
ys = [y / cur_area for y in ys]
es = [e / cur_area for e in es]



xs_spl = list(xs)
ys_spl = list(ys)
es_spl = list(es)


xs_spl.insert(0, xs_spl[0] - width / 2.0)
ys_spl.insert(0, ys_spl[0])
es_spl.insert(0, es_spl[0]/10000.0)

xs_spl.append(xs_spl[-1] + (n_extra_bins + 0.5) * width)
ys_spl.append(0.0)
es_spl.append(es_spl[-1]/10000.0)

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


    scale_factor = target_area / cur_area

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

for fb in fit_bins:
    print("{}\t{}\t{}\t{}\t{}".format(fb['x'], fb['y'], fb['weight'], fb['x_mean'], fb['views']))

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

