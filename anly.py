import os
import math


# This uncertainty comes from Meta rounding the percentage of impressions
# on the top 20 posts to a single significant digit. Since the % is typically
# like 0.03%, then the undertainty is 0.005%...
# Also have the same problem in the prevalence metrics.
ROUNDING_UNCERT = 0.00005

# This data was manually extracted from the Widely Viewed Content Reports up to 2024-Q2
# See this link for current data:
# https://transparency.meta.com/data/widely-viewed-content-report/
WVCR_DATA_STR = """
# Report Pct_Top_Posts Frac_Top_Posts Sum_Post_Viewers
2021Q1 0.05 0.0005 8.50E+08
2021Q2 0.1 0.001 1.02E+09
2021Q3 0.1 0.001 1.58E+09
2021Q4 0.1 0.001 1.08E+09
2022Q1 0.1 0.001 9.02E+08
2022Q2 0.1 0.001 8.89E+08
2022Q3 0.05 0.0005 8.76E+08
2022Q4 0.04 0.0004 8.32E+08
2023Q1 0.04 0.0004 7.76E+08
2023Q2 0.04 0.0004 7.59E+08
2023Q3 0.04 0.0004 8.11E+08
2023Q4 0.04 0.0004 8.06E+08
2024Q1 0.03 0.0003 6.21E+08
2024Q2 0.03 0.0003 6.24E+08
""".strip()

# Global users
# https://www.businessofapps.com/data/facebook-statistics/
# Also https://s21.q4cdn.com/399680738/files/doc_financials/2023/q4/Earnings-Presentation-Q4-2023.pdf
GLOBAL_MAU = 3065000000

# US users
# https://www.statista.com/statistics/408971/number-of-us-facebook-users/
# US and Canada is 272M https://s21.q4cdn.com/399680738/files/doc_financials/2023/q4/Earnings-Presentation-Q4-2023.pdf
# So 250M for US isn't too crazy. (Pop. of Canada is 40M, so probably 15-30M FB users there.
# Total FB users in Canada estimated to be 30M (https://www.statista.com/statistics/282364/number-of-facebook-users-in-canada/). So again, 20M MAU in Canada not crazy.
US_MAU = 250000000

GLOBAL_US_RATIO = float(GLOBAL_MAU) / float(US_MAU)


def process_wvcr_data(WVCR_DATA_STR=WVCR_DATA_STR, ROUNDING_UNCERT=ROUNDING_UNCERT, GLOBAL_US_RATIO=GLOBAL_US_RATIO):
    # Process WVCR manually extracted data...
    WVCR_DATA = {}
    for line in WVCR_DATA_STR.split('\n'):
        if line.startswith('#'):
            continue
        tokens = line.strip().split()
        WVCR_DATA[tokens[0]] = {
            'pct_top_posts': float(tokens[1]),
            'frac_top_posts': float(tokens[2]),
            'sum_post_viewers': int(float(tokens[3])),
            'estimated_total_us_imps': float(tokens[3]) / float(tokens[2]),
            'estimated_total_us_imps_uncert':
                (float(tokens[3]) / float(tokens[2]))
                * (ROUNDING_UNCERT / float(tokens[2])),
            'estimated_total_global_imps': (float(tokens[3]) / float(tokens[2])) * GLOBAL_US_RATIO,
            'estimated_total_global_imps_uncert': (float(tokens[3]) / float(tokens[2])) * GLOBAL_US_RATIO
                * (ROUNDING_UNCERT / float(tokens[2])),
        }

    return WVCR_DATA


def process_cser_data(CSER_FILE, WVCR_DATA):
    # Read in data from the CSER file.
    # You can download the file directly from Meta, and then unzip for the CSV
    # Look for the "Download (CSV)" link at this website for latest
    # https://transparency.meta.com/reports/community-standards-enforcement/
    CSER_DATA_D = {}
    with open(CSER_FILE) as f:
        for line in f:
            tokens = line.strip().split(',')
            if tokens[0] != 'Facebook': # Only WVCR data for Facebook
                continue
            if tokens[2] not in ('Lowerbound Prevalence', 'Upperbound Prevalence', 'UBP'): # Only content prevalence rows
                continue
            if tokens[3] not in WVCR_DATA.keys(): # Only WVCR quarters
                continue

            v = float(tokens[4].replace('"', '').replace('%', ''))

            tag = "{}_{}_{}".format(tokens[0], tokens[1], tokens[3])
            if tag not in CSER_DATA_D:
                CSER_DATA_D[tag] = {
                    'app': tokens[0],
                    'policy_area': tokens[1],
                    'period': tokens[3],
                }
            CSER_DATA_D[tag][tokens[2]] = v

    return CSER_DATA_D


def estimate_violating_impressions(CSER_DATA_D, WVCR_DATA, ROUNDING_UNCERT=ROUNDING_UNCERT):
    # Combine the WVCR data and the CSER data to get violating monthly impressions
    for k, d in CSER_DATA_D.items():
        prev = None
        prev_uncert = None
        if 'UBP' in d: # Dealing with an upper bound prevalence case
            prev = d['UBP'] / 100.0
            prev_uncert = 0.0
        else: # Deailing with a case where there's a proper prevalence measurement
            prev = 0.5 * (d['Upperbound Prevalence'] + d['Lowerbound Prevalence']) / 100.0
            prev_delta = 0.5 * (d['Lowerbound Prevalence'] - d['Lowerbound Prevalence']) / 100.0
            prev_uncert = math.sqrt(ROUNDING_UNCERT**2 + prev_delta**2)

        d['prevalence'] = prev
        d['prevalence_uncert'] = prev_uncert

        # Copy data from WVCR
        d['estimated_total_us_imps'] = WVCR_DATA[d['period']]['estimated_total_us_imps']
        d['estimated_total_us_imps_uncert'] = WVCR_DATA[d['period']]['estimated_total_us_imps_uncert']
        d['estimated_total_global_imps'] = WVCR_DATA[d['period']]['estimated_total_global_imps']
        d['estimated_total_global_imps_uncert'] = WVCR_DATA[d['period']]['estimated_total_global_imps_uncert']
        d['pct_top_posts'] = WVCR_DATA[d['period']]['pct_top_posts']
        d['frac_top_posts'] = WVCR_DATA[d['period']]['frac_top_posts']
        d['sum_post_viewers'] = WVCR_DATA[d['period']]['sum_post_viewers']

        # Calculate violating US imps
        d['estimated_violating_us_imps'] = prev * d['estimated_total_us_imps']
        d['estimated_violating_us_imps_uncert'] = d['estimated_violating_us_imps'] * \
            math.sqrt(
                (d['prevalence_uncert'] / d['prevalence'])**2
                + (d['estimated_total_us_imps_uncert'] / d['estimated_total_us_imps'])**2
            )

        # Calculate violating global imps
        d['estimated_violating_global_imps'] = prev * d['estimated_total_global_imps']
        d['estimated_violating_global_imps_uncert'] = d['estimated_violating_global_imps'] * \
            math.sqrt(
                (d['prevalence_uncert'] / d['prevalence'])**2
                + (d['estimated_total_global_imps_uncert'] / d['estimated_total_global_imps'])**2
            )

        # Divide quarterly numbers by 3 to get monthly numbers.
        d['estimated_monthly_violating_us_imps'] = d['estimated_violating_us_imps'] / 3.0
        d['estimated_monthly_violating_us_imps_uncert'] = d['estimated_violating_us_imps_uncert'] / 3.0
        d['estimated_monthly_violating_global_imps'] = d['estimated_violating_global_imps'] / 3.0
        d['estimated_monthly_violating_global_imps_uncert'] = d['estimated_violating_global_imps_uncert'] / 3.0

    return CSER_DATA_D


def write_estimated_violating_impressions(CSER_DATA_D, OUTFILE):
    # Write the output TSV
    COLS = [
        'app',
        'policy_area',
        'period',
        'Lowerbound Prevalence',
        'Upperbound Prevalence',
        'UBP',
        'pct_top_posts',
        'frac_top_posts',
        'sum_post_viewers',
        'prevalence',
        'prevalence_uncert',
        'estimated_total_us_imps',
        'estimated_total_us_imps_uncert',
        'estimated_violating_us_imps',
        'estimated_violating_us_imps_uncert',
        'estimated_monthly_violating_us_imps',
        'estimated_monthly_violating_us_imps_uncert',
        'estimated_total_global_imps',
        'estimated_total_global_imps_uncert',
        'estimated_violating_global_imps',
        'estimated_violating_global_imps_uncert',
        'estimated_monthly_violating_global_imps',
        'estimated_monthly_violating_global_imps_uncert',
    ]
    tsv_strings = ["\t".join(COLS)]

    CSER_DATA = sorted([d for k, d in CSER_DATA_D.items()], key=lambda x: x['policy_area']+x['period'])
    for d in CSER_DATA:
        vs = [d.get(c, '') for c in COLS]
        value_string = "\t".join(["{}".format(v) for v in vs])
        tsv_strings.append(value_string)
    tsv_string = "\n".join(tsv_strings)

    with open(OUTFILE, 'w') as f:
        f.write(tsv_string)


if __name__=='__main__':
    import sys

    if len(sys.argv)!=3:
        print("Usage: python anly.py [Path to CSER csv file] [Path to desired output file]")
        sys.exit()

    cser_infile = sys.argv[1]
    outfile = sys.argv[2]

    if not os.path.isfile(cser_infile):
        print("Provided path the CSER csv file", cser_infile, "is not found.")
        sys.exit()

    if os.path.isfile(outfile):
        print("Provided output file", outfile, "exists. Will not overwrite.")
        sys.exit()

    wvcr_data = process_wvcr_data()
    cser_data = process_cser_data(cser_infile, wvcr_data)
    violating_imps_data = estimate_violating_impressions(cser_data, wvcr_data)
    write_estimated_violating_impressions(violating_imps_data, outfile)
