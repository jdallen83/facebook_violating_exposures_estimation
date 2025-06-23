import math
import os
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np


VIEWS_BUCKET_MAP = {
    '0 views': {'l': 0, 'h': 0, 'p': 0},
    '1-10 views': {'l': 0, 'h': 1, 'p': 0.5},
    '11-100 views': {'l': 1, 'h': 2, 'p': 1.5},
    '101-1,000 views': {'l': 2, 'h': 3, 'p': 2.5},
    '1,001-10,000 views': {'l': 3, 'h': 4, 'p': 3.5},
    '>10,000 views': {'l': 4, 'h': 5, 'p': 4.5},
}

VIDEO_VIOLATIONS = [
    'Child safety',
    'Harmful or dangerous',
    'Violent or graphic',
    'Harassment and cyberbullying',
    'Nudity or sexual',
    'Promotion of violence and violent extremism',
    'Other',
    'Hateful or abusive',
    'Spam, misleading and scams',
    'Misinformation',
]
CHANNEL_VIOLATIONS = [
    'Channels Spam, misleading and scams',
    'Channels Child safety',
    'Channels Nudity and sexual',
    'Channels Misinformation',
    'Channels Harmful or dangerous',
    'Channels Hateful or abusive',
    'Channels Harassment and cyberbullying',
    'Channels Promotion of violence and violent extremism',
    'Channels Violent or graphic',
    'Channels Multiple policy violations',
]

def get_bucket_means(cur_bucket, spline, xs=None, ys=None, N=100):
    if cur_bucket['v']==0.0:
        return cur_bucket['p'], cur_bucket['p']
    if cur_bucket['p']==0.0:
        return 0, 0

    s = 0.0
    bins = []
    for i in range(N):
        x = cur_bucket['l'] + i * 1.0 / N + 1.0 / N / 2.0
        if ys is None:
            y = spline(x) / N
        else:
            y = np.interp(x, xs, ys)
        if y < 0.0:
            y = 0.0
        s += y
        bins.append((x, y))
    n = cur_bucket['v'] / s
    bins = [(x, y*n) for x, y in bins]
    exp_x = sum([(10**x)*y for x, y in bins])
    mean_x = float(sum([x*y for x, y in bins]) / cur_bucket['v'])
    r = exp_x / cur_bucket['v']
    m = math.log10(r)

    return m, mean_x


def process_histo(df_histo, method='linear'):
    view_histo = []

    for i, r in df_histo.iterrows():
        rd = dict(r)
        task = rd['name']
        bucket = VIEWS_BUCKET_MAP[task]
        d = {
            'd': rd['name'],
            'l': bucket['l'],
            'h': bucket['h'],
            'p': bucket['p'],
            'v': float(rd['pct']),
        }
        view_histo.append(d)

    view_histo = sorted(view_histo, key=lambda x: x['p'])
    view_histo_i = [v for v in view_histo]# if v['p']!=0]

    xs = [v['p'] for v in view_histo_i] + [5, 5.5]
    ys = [v['v'] for v in view_histo_i] + [0.0, 0.0]

    spl = CubicSpline(xs, ys)

    for v in view_histo:
        if method=='linear':
            v['mean_views_weighted'], v['mean_views'] = get_bucket_means(v, spl, xs=xs, ys=ys)
        elif method=='spline':
            v['mean_views_weighted'], v['mean_views'] = get_bucket_means(v, spl)
        else:
            print("Method should be either linear or spline...")
            raise ValueError

    return view_histo


def get_average_views(df_histo, method='linear'):
    view_histo = process_histo(df_histo, method=method)

    low_limit = 0.0
    high_limit = 0.0
    mid_est = 0.0
    estimate = 0.0
    non_0_s = 0.0
    for v in view_histo:
        if v['p']==0:
            continue
        if v['v']==0:
            continue
        non_0_s += v['v']

        low_limit += 10**v['l'] * v['v']
        high_limit += 10**v['h'] * v['v']
        mid_est += 10**v['p'] * v['v']
        estimate += 10**v['mean_views'] * v['v']

    return {
        'lower_bound': low_limit,
        'upper_bound': high_limit,
        'middle_estimate': mid_est,
        'estimate': estimate,
        'lower_bound_non0': low_limit / non_0_s,
        'upper_bound_non0': high_limit / non_0_s,
        'middle_estimate_non0': mid_est / non_0_s,
        'estimate_non0': estimate / non_0_s,
    }


def process_youtube_data(infile, outfile):
    df = pd.read_csv(infile, sep='\t')

    df_hist = df[df['name'].isin(VIEWS_BUCKET_MAP.keys())].copy()
    df_hist['pct'] = df_hist['value'] / 100.0

    quarters = list(df_hist.quarter.unique())

    views_data = []
    quarterly_views_data = {}
    for quarter in quarters:
        df_histo_mp = df_hist[df_hist.quarter==quarter].copy()
        if not len(df_histo_mp):
            print("Warning: No valid views historgram found for", market, period, "so won't have data then...")
            continue
        avg_views = get_average_views(df_histo_mp)
        avg_views['quarter'] = quarter
        quarterly_views_data[quarter] = avg_views

    quarterly_total_videos_removed = {}
    quarterly_total_channels_removed = {}
    quarterly_total_videos_from_channels_removed = {}
    quarterly_violation_fractions = {}
    quarterly_channel_violation_fractions = {}
    quarterly_country_videos_removed = {}

    for i, r in df.iterrows():
        d = dict(r)
        if d['name'] == 'Total videos removed':
            quarterly_total_videos_removed[d['quarter']] = int(d['value'])
        elif d['name'] == 'Total channels removed':
            quarterly_total_channels_removed[d['quarter']] = int(d['value'])
        elif d['name'] == 'Videos removed due to channel-level':
            quarterly_total_videos_from_channels_removed[d['quarter']] = int(d['value'])

        if d['quarter'] not in quarterly_violation_fractions:
            quarterly_violation_fractions[d['quarter']] = {}
        if d['quarter'] not in quarterly_channel_violation_fractions:
            quarterly_channel_violation_fractions[d['quarter']] = {}
        if d['name'] in VIDEO_VIOLATIONS:
            quarterly_violation_fractions[d['quarter']][d['name']] = int(d['value']) * 1.0 / quarterly_total_videos_removed[d['quarter']]
        elif d['name'] in CHANNEL_VIOLATIONS:
            quarterly_channel_violation_fractions[d['quarter']][d['name']] = int(d['value']) * 1.0 / quarterly_total_channels_removed[d['quarter']]

        if d['quarter'] not in quarterly_country_videos_removed:
            quarterly_country_videos_removed[d['quarter']] = {}

        if d['name'].startswith('Country-'):
            quarterly_country_videos_removed[d['quarter']][d['name'].replace('Country-', '')] = int(d['value'])

    for quarter, videos in quarterly_total_videos_removed.items():
        quarterly_country_videos_removed[quarter]['All'] = videos

    df_rows = []
    for quarter, countries in quarterly_country_videos_removed.items():
        for country, videos in countries.items():
            d = dict(quarterly_views_data[quarter])
            d.update({
                'quarter': quarter,
                'country': country,
                'violation': 'All',
                'total_videos_removed_for_quarter': videos,
                'violation_videos_removed': videos,
            })
            df_rows.append(d)
            for violation, frac in quarterly_violation_fractions[quarter].items():
                d = dict(quarterly_views_data[quarter])
                d.update({
                    'quarter': quarter,
                    'country': country,
                    'violation': violation,
                    'total_videos_removed_for_quarter': videos,
                    'violation_videos_removed': videos * frac,
                })
                df_rows.append(d)

    df_out = pd.DataFrame(df_rows)

    df_out['monthly_lower_bound_exposures'] = df_out.lower_bound * df_out.violation_videos_removed / 3.0
    df_out['monthly_upper_bound_exposures'] = df_out.upper_bound * df_out.violation_videos_removed / 3.0
    df_out['monthly_estimate_exposures'] = df_out.estimate * df_out.violation_videos_removed / 3.0
    df_out['monthly_middle_estimate_exposures'] = df_out.middle_estimate * df_out.violation_videos_removed / 3.0

    df_out = df_out[[
        'quarter',
        'country',
        'violation',
        'violation_videos_removed',
        'total_videos_removed_for_quarter',
        'monthly_lower_bound_exposures',
        'monthly_upper_bound_exposures',
        'monthly_estimate_exposures',
        'monthly_middle_estimate_exposures',
        'lower_bound',
        'upper_bound',
        'middle_estimate',
        'estimate',
        'lower_bound_non0',
        'upper_bound_non0',
        'middle_estimate_non0',
        'estimate_non0',
    ]]

    df_out.sort_values(['country', 'violation', 'quarter'], inplace=True)
    df_out.to_csv(outfile, index=False, sep='\t')

    return df_out


if __name__=='__main__':
    import sys

    if len(sys.argv)!=3:
        print("Usage: python youtube_anly.py [Path to extracted YouTube data csv file] [Path to desired output file]")
        sys.exit()

    cger_infile = sys.argv[1]
    outfile = sys.argv[2]

    if not os.path.isfile(cger_infile):
        print("Provided path the csv file", cger_infile, "is not found.")
        sys.exit()

    if os.path.isfile(outfile):
        print("Provided output file", outfile, "exists. Will not overwrite.")
        sys.exit()

    df_violating_exposures = process_youtube_data(cger_infile, outfile)
