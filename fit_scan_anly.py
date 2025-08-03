import os
import json

import numpy as np

import facebook_violating_exposures_estimation.plot as plot
import facebook_violating_exposures_estimation.distribution_fit as distribution_fit


def comp_doc(doc, comp):
    c = {k: comp[k]==doc.get(k) for k in comp.keys()}
    return False not in c.values()


def curve_views(x, y):
    w = sum(y)
    return sum([(10**xx)*yy for xx, yy in zip(x, y)]) / w


def get_doc_estimates(doc):
    ests = {
        'normal': doc['fit']['estimates']['Modeled (Normal)'][1],
        'normal_uncert': doc['fit']['estimates']['Modeled (Normal)'][2] - doc['fit']['estimates']['Modeled (Normal)'][1],
        'spline': doc['fit']['estimates']['Modeled (Spline)'][1],
        'spline_uncert': doc['fit']['estimates']['Modeled (Spline)'][2] - doc['fit']['estimates']['Modeled (Spline)'][1],
    }
    doc.update(ests)
    return doc


def estimate_average_views_from_fit_distribution(fit, min_x, max_x, n=500):
    x = np.array(fit['x'])
    y = np.array(fit['best_fit'])
    e = np.array([0.0000001 for yy in y])

    fit_p, fit_cov = distribution_fit.chi2_fit(distribution_fit.normal_distribution, x, y, e, [1.0, 1.0, 1.0])

    xs = np.linspace(min_x, max_x, num=int((max_x-min_x)*n), endpoint=False)
    ys = distribution_fit.normal_distribution(xs, fit_p)

    avg_views_from_best_fit = sum([(10**xx)*yy for xx, yy in zip(x, y)]) / sum(y)
    avg_views = sum([(10**xx)*yy for xx, yy in zip(xs, ys)]) / sum(ys)
    return avg_views_from_best_fit, avg_views


def plot_est_true(docs, label, dir):
    for fit_type in ['Normal', 'Spline']:
        ests = [get_doc_estimates(d) for d in docs]
        ests = sorted(ests, key=lambda x: x['true_average'])

        average_views_from_fit_dist = []
        average_views_from_best_fit = []
        for doc in ests:
            min_x = min(doc['true_dist_x'])
            max_x = max(doc['true_dist_x']) + (doc['true_dist_x'][1] - doc['true_dist_x'][0])
            fit = doc['fit'][fit_type.lower()]['curves']

            avg_from_bf, avg_from_fit = estimate_average_views_from_fit_distribution(fit, min_x, max_x)

            average_views_from_fit_dist.append(avg_from_fit)
            average_views_from_best_fit.append(avg_from_bf)

        x = [e['true_average'] for e in ests]
        yx = [xx for xx in x]
        y_fit = [e[fit_type.lower()] for e in ests]

        plot.plot([
                ('plot', x, yx, {'color': 'tab:blue', 'marker': '', 'linestyle': '-', 'linewidth': 3.0, 'label': 'True',}),
                ('plot', x, y_fit, {'color': 'tab:orange', 'marker': 'o', 'linestyle': '', 'linewidth': 3.0, 'label': 'Estimated ({})'.format(fit_type),}),
            ],
            show=False,
            title='Estimated Avg. Views vs. True Avg. Views ({} Parameters)'.format(label),
            ylabel='Estimates Avg. Views',
            xlabel='True Avg. Views',
            xlim=[0, None],
            ylim=[0, None],
            padding=0.6,
            save=os.path.join(dir, 'est_vs_true_{}_params_{}.png'.format(label.lower().replace(' ', '_'), fit_type.lower())),
        )


        if fit_type=='Normal':
            plot.plot([
                    ('plot', x, yx, {'color': 'tab:blue', 'marker': '', 'linestyle': '-', 'linewidth': 3.0, 'label': 'True',}),
                    ('plot', x, average_views_from_fit_dist, {'color': 'tab:orange', 'marker': 'o', 'linestyle': '', 'linewidth': 3.0, 'label': 'Estimated Fit ({})'.format(fit_type), 'alpha': 0.7,}),
                    ('plot', x, average_views_from_best_fit, {'color': 'tab:red', 'marker': '.', 'linestyle': '', 'linewidth': 3.0, 'label': 'Best Fit ({})'.format(fit_type), 'alpha': 0.9,}),
                ],
                show=False,
                title='Estimated Avg. Views vs. True Avg. Views ({} Parameters)'.format(label),
                ylabel='Estimates Avg. Views',
                xlabel='True Avg. Views',
                xlim=[0, None],
                ylim=[0, None],
                padding=0.6,
                save=os.path.join(dir, 'est_from_best_fit_vs_true_{}_params_{}.png'.format(label.lower().replace(' ', '_'), fit_type.lower())),
            )


def plot_dep_key_behavior(ul, filter_doc, docs, dep_key, dep_key_label, label, fit_type, dir):
    cur_params = dict(ul)
    cur_params.update({k: v for k, v in filter_doc.items() if k!=dep_key})

    cur_docs = [d for d in docs if comp_doc(d['run_params'], cur_params)]
    cur_docs = sorted(cur_docs, key=lambda x: x['run_params'][dep_key])

    sel_params = dict(ul)
    sel_params.update(filter_doc)

    sel_docs = [d for d in docs if comp_doc(d['run_params'], sel_params)]

    x_cur = [d['run_params'][dep_key] for d in cur_docs]
    y_cur = [d['fit']['estimates']['Modeled ({})'.format(fit_type)][1] for d in cur_docs]
    e_cur = [d['fit']['estimates']['Modeled ({})'.format(fit_type)][2] - d['fit']['estimates']['Modeled ({})'.format(fit_type)][1] for d in cur_docs]

    y_true = [d['true_average'] for d in cur_docs]

    x_sel = [d['run_params'][dep_key] for d in sel_docs]
    y_sel = [d['fit']['estimates']['Modeled ({})'.format(fit_type)][1] for d in sel_docs]
    e_sel = [d['fit']['estimates']['Modeled ({})'.format(fit_type)][2] - d['fit']['estimates']['Modeled ({})'.format(fit_type)][1] for d in sel_docs]


    plot.plot([
            ('plot', x_cur, y_true, {'color': 'tab:blue', 'marker': '', 'linestyle': '-', 'linewidth': 3.0, 'label': 'True',}),
            ('errorbar', x_cur, y_cur, {'yerr': e_cur, 'color': 'tab:orange', 'marker': 'o', 'linestyle': '', 'linewidth': 1.0, 'label': 'Estimated ({})'.format(fit_type),}),
            ('errorbar', x_sel, y_sel, {'yerr': e_sel, 'color': 'tab:red', 'marker': '.', 'linestyle': '', 'linewidth': 1.0, 'label': 'Model Params Estimate ({})'.format(fit_type)}),
        ],
        show=False,
        title='Estimated Avg. Views vs. {} ({} Parameters, u={}, l={})'.format(dep_key_label, label, ul['u'], ul['l']),
        ylabel='Estimates Avg. Views',
        xlabel='{}'.format(dep_key_label),
        #xlim=[0, None],
        #ylim=[0, None],
        padding=0.5,
        save=os.path.join(dir, 'dep_key_plot_{}_{}_{}_u{}_l{}{}.png'.format(
            dep_key,
            label.lower().replace(' ', '_'),
            fit_type.lower(),
            ul['u'],
            ul['l'],
            ul.get('tag', ''),
        )),
    )


def plot_dep_key_estimates(mids, filter_doc, docs, label, plot_dir):
    for mid in mids:
        for fit_type in ('Normal', 'Spline'):
            for dep_key, dep_key_label in dep_key_label_map.items():
                plot_dep_key_behavior(mid, filter_doc, docs, dep_key, dep_key_label, label, fit_type, plot_dir)


if __name__=="__main__":
    import sys

    DIR = sys.argv[1]
    PLOT_DIR = sys.argv[2]

    INFILES = [os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith('.json')]
    print("Found", len(INFILES), "fit scan files.")

    DOCS = [json.load(open(f)) for f in INFILES]

    MIDS = [
        {'u': -3.0, 'l': 2.182,},
        {'u': 1.96, 'l': 0.7,},
        {'u': '2024Q2', 'l': 'All_Region', 'tag': 'spline_',},
        {'u': '2024Q2', 'l': 'All_Region', 'tag': 'normal_',},
    ]

    BEST = {
        'n_extra_bins': 1,
        'hist_bin_width': 0.1,
        'x_hist_max': 9.0,
        'rounding': 12,
    }

    ACTUAL_TT = {
        'n_extra_bins': 1,
        'hist_bin_width': 1.0,
        'x_hist_max': 7.0,
        'rounding': 3,
    }

    ACTUAL_YT = {
        'n_extra_bins': 1,
        'hist_bin_width': 1.0,
        'x_hist_max': 5.0,
        'rounding': 4,
    }

    ACTUAL_FB = {
        'n_extra_bins': 1,
        'hist_bin_width': 1.0,
        'x_hist_max': 5.0,
        'rounding': 2,
    }

    dep_key_label_map = {
        'x_hist_max': 'Maximum Log10(Views) Bin',
        'rounding': 'Rounding Precision',
        'n_extra_bins': 'Num Projected Bins',
        'hist_bin_width': "Width of Histogram Bins",
    }

    yt_docs = [d for d in DOCS if comp_doc(d['run_params'], ACTUAL_YT)]
    tt_docs = [d for d in DOCS if comp_doc(d['run_params'], ACTUAL_TT)]
    fb_docs = [d for d in DOCS if comp_doc(d['run_params'], ACTUAL_FB)]
    best_docs = [d for d in DOCS if comp_doc(d['run_params'], BEST)]
    mid_docs = [[d for d in DOCS if comp_doc(d['run_params'], mid)] for mid in MIDS]

    print("Plotting YouTube")
    plot_est_true(yt_docs, 'YouTube', PLOT_DIR)
    plot_dep_key_estimates(MIDS, ACTUAL_YT, DOCS, 'YouTube', PLOT_DIR)

    print("Plotting TikTok")
    plot_est_true(tt_docs, 'TikTok', PLOT_DIR)
    plot_dep_key_estimates(MIDS, ACTUAL_TT, DOCS, 'TikTok', PLOT_DIR)

    print("Plotting Facebook")
    plot_est_true(fb_docs, 'Facebook', PLOT_DIR)
    plot_dep_key_estimates(MIDS, ACTUAL_FB, DOCS, 'Facebook', PLOT_DIR)

    print("Plotting Ideal Params")
    plot_est_true(best_docs, 'Ideal', PLOT_DIR)
    plot_dep_key_estimates(MIDS, BEST, DOCS, 'Ideal', PLOT_DIR)
