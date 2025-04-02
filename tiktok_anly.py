import math
import os
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np

# Metric,Period type,Period,Policy type,Issue,Task type,Task,Location,Market,Result

INFILE = '/Users/jeff/work/facebook_us_violating_impressions/tiktok/2024Q2_raw_data_cger_English (2).csv'
INFILE = '/home/jeff/work/ii/analyses/violating_exposures_on_fb/tiktok/2024Q2_raw_data_cger_English (2).csv'
OUTFILE = '/home/jeff/work/ii/analyses/violating_exposures_on_fb/tiktok/2024Q2_raw_data_cger_English--Violating_exposures.tsv'
VIEWS_BUCKET_MAP = {
    '0 views': {'l': 0, 'h': 0, 'p': 0},
    '1-10 views': {'l': 0, 'h': 1, 'p': 0.5},
    '11-100 views': {'l': 1, 'h': 2, 'p': 1.5},
    '101-1,000 views': {'l': 2, 'h': 3, 'p': 2.5},
    '1,001-10,000 views': {'l': 3, 'h': 4, 'p': 3.5},
    '10,001-100,000 views': {'l': 4, 'h': 5, 'p': 4.5},
    '100,001-1,000,000 views': {'l': 5, 'h': 6, 'p': 5.5},
    '>1,000,000 views': {'l': 6, 'h': 6, 'p': 6},
}
PERIODS_MAP = {
    'Apr-Jun 2024': '2024-Q2',
    'Jan-Mar 2024': '2024-Q1',
    'Oct-Dec 2023': '2023-Q4',
}


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
        task = rd['Task']
        bucket = VIEWS_BUCKET_MAP[task]
        d = {
            'd': rd['Task'],
            'l': bucket['l'],
            'h': bucket['h'],
            'p': bucket['p'],
            'v': float(rd['Result']),
        }
        view_histo.append(d)

    view_histo = sorted(view_histo, key=lambda x: x['p'])
    view_histo_i = [v for v in view_histo if v['p']!=0]

    xs = [0] + [v['p'] for v in view_histo_i] + [7, 7.5]
    ys = [0] + [v['v'] for v in view_histo_i] + [0.0, 0.0]

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


def policy_sub_issue_map(p):
    if ' - ' in p:
        return p.split(' - ')[0].strip()
    elif p=='Misinformation':
        return 'Integrity & Authenticity'
    elif p=='Sexual Exploitation & Gender-Based Violence':
        return 'Safety & Civility'
    else:
        raise KeyError


def period_to_quarter(p):
    if 'Jan-Mar' in p:
        q = 'Q1'
    elif 'Apr-Jun' in p:
        q = 'Q2'
    elif 'Jul-Sep' in p:
        q = 'Q3'
    elif 'Oct-Dec' in p:
        q = 'Q4'
    else:
        print(p)
        raise ValueError

    y = p.strip().split()[-1]
    return "{}{}".format(y, q)


def process_tiktok_data(infile, outfile=None, market=None, period=None):
    df = pd.read_csv(infile)
    df['quarter'] = df.Period.apply(period_to_quarter)

    if market is not None:
        df = df[df.Market==market].copy()
    if period is not None:
        df = df[df.Period==period].copy()

    df_hist = df[(df['Task type']=='Share of total removals')].copy()

    markets = list(df_hist.Market.unique())
    periods = list(df_hist.Period.unique())

    df = df[(df.Market.isin(markets)) & (df.Period.isin(periods))].copy()

    views_data = []
    for market in markets:
        for period in periods:
            df_histo_mp = df_hist[(df_hist.Market==market)&(df_hist.Period==period)].copy()
            if not len(df_histo_mp):
                print("Warning: No valid views historgram found for", market, period, "so won't have data then...")
                continue
            avg_views = get_average_views(df_histo_mp)
            avg_views['Market'] = market
            avg_views['Period'] = period
            views_data.append(avg_views)

    df_avg_views = pd.DataFrame(views_data)

    # Total videos removed only exists for overall, not broken down by subtype
    df_vids = df[(df.Metric=='Total videos removed') & (df['Task type']=='All')].copy()[['Market', 'Period', 'Result']]
    # Removal rate before any views does exist by issue and subtype
    df_0view_removal_rate = df[(df.Metric=='Removal rate before any views') & (df['Task type']=='All')].copy()[['Market', 'Period', 'Issue', 'Result']]
    # Removal rate before any views from the market level histogram
    df_0view_removal_rate_histogram = df[(df['Task type']=='Share of total removals') & (df.Task=='0 views')].copy()[['Market', 'Period', 'Result']]
    print(df_0view_removal_rate_histogram)

    df_viols = df[(df.Metric=='Category share') & (df.Task=='All') & (df['Policy type']=='Policy')].copy()
    df_viols_sub = df[(df.Metric=='Category share') & (df.Task=='All') & (df['Policy type']=='Sub-policy')].copy()

    policy_map = {}
    for i, r in df_viols.iterrows():
        d = dict(r)
        if d['Location'] not in policy_map:
            policy_map[d['Location']] = {'Youth Safety & Well-Being': 0.0}
        policy_map[d['Location']][d['Issue']] = d['Result']

    df_viols['Issue_Main'] = df_viols.Issue
    df_viols['Result_Combined'] = df_viols.Result
    df_viols_sub['Issue_Main'] = df_viols_sub.Issue.apply(policy_sub_issue_map)
    df_viols_sub['Result_Combined'] = df_viols_sub.apply(
        lambda x: x['Result']*policy_map[x['Location']][x['Issue_Main']],
        axis=1
    )

    df_viols = pd.concat([df_viols, df_viols_sub])

    df_viols = pd.merge(df_viols, df_vids, on=['Market', 'Period'], how='left', suffixes=('', '_Total_videos_removed'))
    df_viols = pd.merge(df_viols, df_0view_removal_rate, on=['Market', 'Period', 'Issue'], how='left', suffixes=('', '_Removal_rate_before_any_views'))
    df_viols = pd.merge(df_viols, df_0view_removal_rate_histogram, on=['Market', 'Period'], how='left', suffixes=('', '_Removal_rate_before_any_views_from_histogram'))
    df_viols = pd.merge(df_viols, df_avg_views, on=['Market', 'Period'], how='left', suffixes=('', '_avg_views'))

    df_viols['monthly_issue_videos_removed'] = df_viols.Result_Total_videos_removed * df_viols.Result_Combined / 3.0
    df_viols['monthly_exposures_lower_bound'] = df_viols.lower_bound * df_viols.monthly_issue_videos_removed / 3.0
    df_viols['monthly_exposures_upper_bound'] = df_viols.upper_bound * df_viols.monthly_issue_videos_removed / 3.0
    df_viols['monthly_exposures_middle_estimate'] = df_viols.middle_estimate * df_viols.monthly_issue_videos_removed / 3.0
    df_viols['monthly_exposures_estimate'] = df_viols.estimate * df_viols.monthly_issue_videos_removed / 3.0

    df_viols['monthly_exposures_lower_bound_non0_corrected'] = df_viols.lower_bound_non0 * df_viols.monthly_issue_videos_removed * (1.0-df_viols.Result_Removal_rate_before_any_views) / 3.0
    df_viols['monthly_exposures_upper_bound_non0_corrected'] = df_viols.upper_bound_non0 * df_viols.monthly_issue_videos_removed * (1.0-df_viols.Result_Removal_rate_before_any_views) / 3.0
    df_viols['monthly_exposures_middle_estimate_non0_corrected'] = df_viols.middle_estimate_non0 * df_viols.monthly_issue_videos_removed * (1.0-df_viols.Result_Removal_rate_before_any_views) / 3.0
    df_viols['monthly_exposures_estimate_non0_corrected'] = df_viols.estimate_non0 * df_viols.monthly_issue_videos_removed * (1.0-df_viols.Result_Removal_rate_before_any_views) / 3.0

    df_viols.rename(columns={'Result': 'Result_Category_share', 'Result_Combined': 'Result_Category_share_overall', 'Issue_Main': 'Issue_Primary_Category'}, inplace=True)
    df_viols.sort_values(by=['Market', 'Policy type', 'Issue_Primary_Category', 'Issue', 'quarter'], inplace=True)

    if outfile is not None:
        df_viols.to_csv(outfile, sep='\t', index=False)

    return df_viols


if __name__=='__main__':
    import sys

    if len(sys.argv)!=3:
        print("Usage: python tiktok_anly.py [Path to CSER csv file] [Path to desired output file]")
        sys.exit()

    cger_infile = sys.argv[1]
    outfile = sys.argv[2]

    if not os.path.isfile(cger_infile):
        print("Provided path the CGER csv file", cger_infile, "is not found.")
        sys.exit()

    if os.path.isfile(outfile):
        print("Provided output file", outfile, "exists. Will not overwrite.")
        sys.exit()

    df_violating_exposures = process_tiktok_data(cger_infile, outfile)
