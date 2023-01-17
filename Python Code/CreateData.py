import pandas as pd
import pdb
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

if __name__ == "__main__":

    ratingsFile = 'ratings.csv'
    finRatiosFile = 'full_data.csv'
    indexFile = 'capitaliq-cusip.csv'

    # https://www.programiz.com/python-programming/datetime/strptime
    custom_date_parser = lambda x: datetime.strptime(x, "%d-%b-%y")
    df_ratings = pd.read_csv(ratingsFile, delimiter=",", header='infer', infer_datetime_format=True,
                             parse_dates=['Rating Date'], date_parser=custom_date_parser, dayfirst=True, )
    df_finRatios = pd.read_csv(finRatiosFile, delimiter=",", header='infer', infer_datetime_format=True,
                     dayfirst=True, parse_dates=['Public Date'], date_parser=custom_date_parser)
    df_index = pd.read_csv(indexFile, delimiter=",", header='infer', infer_datetime_format=True, parse_dates=True,
                    dayfirst=True)

    print("Ratings df shape:", df_ratings.shape)
    print(df_ratings.columns)
    ratingType = df_ratings['Type of Rating'].unique()
    plt.hist(df_ratings['Type of Rating'], bins=len(ratingType), orientation='horizontal')
    plt.show()
    df_ratings = df_ratings[df_ratings['Type of Rating'] == 'STDLONG']
    print("Ratings df shape after STDLONG only:", df_ratings.shape)
    dates = df_ratings['Rating Date'].unique()
    print(df_ratings.loc[0])
    print("unique dates:", len(dates))
    plt.hist(df_ratings['Rating Date'], bins=len(dates), orientation='horizontal')
    plt.show()
    date_list = df_ratings['Rating Date'].to_list()
    fin_year_list = [x.year + x.month/100 for x in date_list]
    plt.hist(fin_year_list, bins=6, orientation='horizontal')
    plt.show()
    df_ratings['RateYear'] = fin_year_list
    df_ratings['company_cusip'] = 'NA'

    print("*"*100)
    df_index = df_index.drop(columns=['startdate', 'enddate'], axis=1)
    print("index df shape:", df_index.shape)
    print(df_index.columns)
    print(df_index.loc[0])
    print("unique cusips:", len(df_index['cusip'].unique()))
    print("*"*100)

    print("index df fin ratio:", df_finRatios.shape)
    print(df_finRatios.columns)
    print(df_finRatios.loc[0])

    print("unique cusips:", len(df_finRatios['CUSIP IDENTIFIER - HISTORICAL'].unique()))
    df_finRatios = df_finRatios.rename(columns={"CUSIP IDENTIFIER - HISTORICAL": "cusip",
                                                'EXCHANGE TICKER SYMBOL - HISTORICAL': 'symbolvalue'})
    date_list = df_finRatios['Public Date'].to_list()
    fin_year_list = [x.year + x.month/100 for x in date_list]
    df_finRatios['PublicYear'] = fin_year_list
    df_finRatios['RateYear'] = fin_year_list
    df_finRatios['Rating'] = "NR"
    df_finRatios['companyname'] = "NoName"

    print("*"*100)
    print("\ndf_index columns:\n", "*"*100, "\n", df_index.dtypes)
    print("\ndf_finRatios columns:\n", "*"*100, "\n", df_finRatios.dtypes)
    print("\ndf_ratings columns:\n", "*"*100, "\n", df_ratings.dtypes)
    print("*"*100)

    #df_index = df_index.astype({"cusip": str})
    #df_finRatios = df_finRatios.astype({"cusip": str})
    #df_index['cusip'] = df_index['cusip'].apply(lambda x: x.str[:6])
    #df_finRatios['cusip'] = df_finRatios['cusip'].apply(lambda x: x.str[:6])

    rat_list = df_ratings['symbolvalue'].unique()
    fin_list = df_finRatios['symbolvalue'].unique()

    common_tickers = rat_list[np.isin(rat_list, fin_list)]
    print("ratings & fin ratios unique ticker symbols: ", len(rat_list), len(fin_list))
    print("ratings & fin ratios same ticker symbols: ", len(common_tickers) ,common_tickers.shape)
    locate_ticker_finratios = df_ratings['symbolvalue'].isin(fin_list) # & pd.notna(df_ratings['symbolvalue'])
    locate_ticker_ratings = df_finRatios['symbolvalue'].isin(rat_list)
    print("num locations of ratings tickers in fin ratios tickers and vice versa: ",
          locate_ticker_finratios.sum(), locate_ticker_finratios.shape,
          locate_ticker_ratings.sum(), locate_ticker_ratings.shape)

    # create new column & insert new credit ratings for same symbolvalue obtained after fin ratio Public Date
    for ticker in common_tickers:
        all_company_ratings = df_ratings[df_ratings['symbolvalue'] == ticker]
        all_company_finRatios = df_finRatios[df_finRatios['symbolvalue'] == ticker]
        for i in all_company_finRatios.index:
            date = df_finRatios.loc[i, 'PublicYear']
            ratings_after = all_company_ratings[all_company_ratings['RateYear'] <= date]
            if len(ratings_after) > 0:
                df_finRatios.at[i, 'Rating'] = ratings_after.iloc[-1]['Rating']
                df_finRatios.at[i, 'RateYear'] = ratings_after.iloc[-1]['RateYear']
                df_finRatios.at[i, 'companyname'] = ratings_after.iloc[-1]['Entity Name']

    rated_1 = df_finRatios.loc[df_finRatios['Rating'] != 'NR'][['Rating', 'RateYear', 'PublicYear']]
    print("\n!! RATED  ", len(rated_1), ' of the ', len(df_finRatios), 'fin ratio observations')

    # Use index to find common company IDs and create database of all the cusip's
    # multiple DUPLICATE AND UNIQUE cusips for each company id exist in index database

    rat_list = df_ratings['Capital IQ Company ID'].unique()
    idx_list = df_index['companyid'].unique()
    common_ids = rat_list[np.isin(rat_list, idx_list)]
    print("ratings & index ratios unique company id: ",len(idx_list), len(rat_list))
    print("ratings & index ratios same company id: ", np.isin(rat_list, idx_list).sum(),
          np.isin(rat_list, idx_list).shape)
    #locate_ids_index = df_ratings['Capital IQ Company ID'].isin(idx_list)
    locate_rated_ids = df_index['companyid'].isin(rat_list)
    print("num locations of ratings ids in the df_index - need to check all cusip's against fin ratio cusips: ",
          #     locate_ids_index.sum(), locate_ids_index.shape,
          locate_rated_ids.sum(), locate_rated_ids.shape)

    # Use index to find common company CUSIPs

    idx_list = df_index['company_cusip'].unique()
    fin_list = df_finRatios['company_cusip'].unique()
    print("unique fin ratio cusips:", len(fin_list))
    rat_idx_list = df_index[locate_rated_ids]['company_cusip'].unique()
    common_cusips = fin_list[np.isin(fin_list, idx_list)]
    common_cusips = common_cusips[np.isin(common_cusips, rat_idx_list)]
    print("fin ratios & index & ratings same company cusips: ", len(common_cusips))

    # find which fin ratio cusip correspond to common rating_index cusips and assign rating

    for cusip in common_cusips:
        rated_company_id = df_index[df_index['company_cusip'] == cusip]['companyid'].unique()[0]
        all_company_ratings = df_ratings[df_ratings['Capital IQ Company ID'] == rated_company_id]
        all_company_finRatios = df_finRatios[df_finRatios['company_cusip'] == cusip]
        for i in all_company_finRatios.index:
            date = df_finRatios.loc[i, 'PublicYear']
            ratings_after = all_company_ratings[all_company_ratings['RateYear'] <= date]
            if len(ratings_after) > 0:
                df_finRatios.at[i, 'Rating'] = ratings_after.iloc[-1]['Rating']
                df_finRatios.at[i, 'RateYear'] = ratings_after.iloc[-1]['RateYear']
                df_finRatios.at[i, 'companyname'] = ratings_after.iloc[-1]['Entity Name']

    rated_2 = df_finRatios.loc[df_finRatios['Rating'] != 'NR'][['Rating', 'RateYear', 'PublicYear']]
    print("\n!! RATED  ", len(rated_2), ' of the ', len(df_finRatios), 'fin ratio observations')

    # Clean Up non-rated
    df_finRatios = df_finRatios[df_finRatios['Rating'] != 'NR']             # Non Rated
    df_finRatios = df_finRatios.drop(columns=['Total Debt/Total Assets.1', 'Global Company Key',
                                              'CRSP PERMNO','Forward P/E to 1-year Growth (PEG) ratio',
                                              'Forward P/E to Long-term Growth (PEG) ratio'])

    # Clean up rows where financial ratios have not changed from t to t+1
    # (due to only quarterly/semi reporting of new fin ratio data in some cases)

    fin_ratio_names = ['Net Profit Margin',
       'Operating Profit Margin Before Depreciation',
       'Operating Profit Margin After Depreciation', 'Gross Profit Margin',
       'Pre-tax Profit Margin', 'Cash Flow Margin', 'Return on Assets',
       'Return on Equity', 'Return on Capital Employed', 'Effective Tax Rate',
       'After-tax Return on Average Common Equity',
       'After-tax Return on Invested Capital',
       'After-tax Return on Total Stockholders Equity',
       'Pre-tax return on Net Operating Assets',
       'Pre-tax Return on Total Earning Assets', 'Gross Profit/Total Assets',
       'Common Equity/Invested Capital', 'Long-term Debt/Invested Capital',
       'Total Debt/Invested Capital', 'Capitalization Ratio',
       'Interest/Average Long-term Debt', 'Interest/Average Total Debt',
       'Cash Balance/Total Liabilities', 'Inventory/Current Assets',
       'Receivables/Current Assets', 'Total Debt/Total Assets',
       'Total Debt/EBITDA', 'Short-Term Debt/Total Debt',
       'Current Liabilities/Total Liabilities',
       'Long-term Debt/Total Liabilities',
       'Profit Before Depreciation/Current Liabilities',
       'Operating CF/Current Liabilities', 'Cash Flow/Total Debt',
       'Free Cash Flow/Operating Cash Flow',
       'Total Liabilities/Total Tangible Assets', 'Long-term Debt/Book Equity',
       'Total Debt/Capital', 'Total Debt/Equity',
       ]

    for cusip in fin_list:
        all_company_finRatios = df_finRatios[df_finRatios['company_cusip'] == cusip]
        fin_ratios = all_company_finRatios[fin_ratio_names].to_numpy().astype("float32")
        if len(fin_ratios) > 1:
            diff_fin_ratios = (fin_ratios[1:] - fin_ratios[0:fin_ratios.shape[0]-1]) ** 2
            diff_fin_ratios = np.where(np.isnan(diff_fin_ratios), 0, diff_fin_ratios)
            drop = diff_fin_ratios.sum(axis=1)
            drop_flag = np.where(drop > 0, False, True).tolist()
            idx_nums = np.array(all_company_finRatios.index)[1:]   # keep first row
            rows_to_drop = idx_nums[drop_flag].tolist()
            df_finRatios = df_finRatios.drop(index=rows_to_drop)
            print("dropped ", len(rows_to_drop), " from cusip ", cusip)

    headers = df_finRatios.columns
    all_fin_ratios = headers[6:76]                            # float32 financial ratios
    df_finRatios[all_fin_ratios] = df_finRatios[all_fin_ratios].replace(np.nan, 0)   # replace nan by zeros
    pdb.set_trace()
    df_finRatios.to_csv("new_data.csv", sep=',', index=False)




