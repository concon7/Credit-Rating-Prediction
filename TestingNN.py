import pandas as pd
import pdb
from matplotlib import cm
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ReLU
import pdb
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from LogisticRegression import *
from NNClassifier import *


def show_or_save(showGraphs=True, name="plt"):
    if showGraphs:
        plt.show()
    else:
        plt.savefig(name)
        plt.close()


# calc covariance matrix for all features
# drop highly correlated (cov>0.80) that have lowest number of observations
# ask user which one to keep that have equal number of observations
# print the outcome to store for future runs as a include_only list in load_clean_data() under delete=2 option

def runOnce(df, DF_train, headers, fidx, show, cov_threshold=0.80):
    xD = DF_train[headers[fidx[0]:fidx[1]]].to_numpy().astype("float32")  # financial ratios
    xF = DF_train[headers[fidx[2]:fidx[3]]].to_numpy().astype("int32")  # flag 1, if data==0 or NaN

    x = (xD - xD.mean(axis=0)) / xD.std(axis=0)
    print("shape of data: ", x.shape)

    x_cov = np.cov(x.T)
    covlist = []
    m = np.zeros((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            if (x_cov[i, j] >= cov_threshold) and (i != j):
                l = [headers[fidx[0] + i], headers[fidx[0] + j], x_cov[i, j], i, j]
                covlist.append(l)
                m[i, j] = x_cov[i, j]

    plot_confusion_matrix(x_cov, np.arange(x.shape[1]), title="Covariance Matrix", x_label="Financial Ratios",
                          y_label="Financial Ratios", anot=False)
    show_or_save(name="plt", showGraphs=show)
    plot_confusion_matrix(m, np.arange(x.shape[1]), title="Covariance Matrix > 0.80", x_label="Financial Ratios",
                          y_label="Financial Ratios", anot=False)
    show_or_save(name="plt", showGraphs=show)

    delete_list = []
    for i in range(len(covlist)):
        name_1 = covlist[i][0]
        name_2 = covlist[i][1]
        cov_12 = covlist[i][2]
        idx_1 = covlist[i][3]
        idx_2 = covlist[i][4]
        n_1 = xF[:, idx_1].sum()
        n_2 = xF[:, idx_2].sum()
        print(i, 'cov( ', name_1, '( ', idx_1, ":", n_1, ' ) , ', name_2, '( ', idx_2, ":", n_2, ' ) = ', cov_12)
        if name_1 not in delete_list and name_2 not in delete_list:
            if n_1 == n_2:
                print('Keep 1:', name_1, 'or 2:', name_2, ":")
                x = input()
                print(x)
                if x == '1':
                    delete_list.append(name_2)
                    delete_list.append("{0}_".format(name_2))
                    print("Deleting ", name_2)
                else:
                    delete_list.append(name_1)
                    delete_list.append("{0}_".format(name_1))
                    print("Deleting ", name_1)
            elif n_1 > n_2:
                print("Deleting ", name_2)
                delete_list.append("{0}_".format(name_2))
                delete_list.append(name_2)
            else:
                print("Deleting ", name_1)
                delete_list.append(name_1)
                delete_list.append("{0}_".format(name_1))

    delete_list = list(dict.fromkeys(delete_list))
    n_deleted = len(delete_list)
    fidx[1] -= int(n_deleted / 2)  # eliminate one column from data index
    fidx[2] -= int(n_deleted / 2)  # eliminate one column from data index
    fidx[3] -= n_deleted  # eliminate one column from flag index

    print(df.shape)
    print(df.columns)
    print('Data index:', fidx[0], fidx[1])
    print('Column index:', fidx[2], fidx[3])

    DF_train = df.drop(columns=delete_list)
    headers = DF_train.columns
    HD = headers[fidx[0]:fidx[1]]
    HF = headers[fidx[2]:fidx[3]]
    print(HD)
    print(HF)
    print('df shape from ', df.shape, ' to ', DF_train.shape)

    for fin_ratio_ in HF:
        DF_train = DF_train[DF_train[fin_ratio_] > 0]  # Eliminate NaN & zeros

    print('df shape after NaN/zero reduction: ', DF_train.shape)

    return DF_train, headers, fidx


def load_clean_data(file_name, fidx, show=True, delete=0):
    delete_columns = ['Shillers Cyclically Adjusted P/E Ratio', 'Book/Market',
                      'Price/Operating Earnings (Basic, Excl. EI)',
                      'Price/Operating Earnings (Diluted, Excl. EI)',
                      'P/E (Diluted, Excl. EI)', 'P/E (Diluted, Incl. EI)',
                      'Price/Sales', 'Price/Cash flow', 'Dividend Payout Ratio',
                      'Operating Profit Margin After Depreciation',
                      'Pre-tax Profit Margin', 'Return on Equity',
                      'Effective Tax Rate',
                      'After-tax Return on Invested Capital',
                      'Pre-tax return on Net Operating Assets',
                      'Pre-tax Return on Total Earning Assets',
                      'Capitalization Ratio', 'Inventory/Current Assets',
                      'Receivables/Current Assets', 'Short-Term Debt/Total Debt',
                      'Long-term Debt/Book Equity', 'Cash Conversion Cycle (Days)',
                      'Inventory Turnover', 'Asset Turnover', 'Receivables Turnover',
                      'Payables Turnover', 'Sales/Invested Capital', 'Sales/Stockholders Equity',
                      'Sales/Working Capital', 'Research and Development/Sales', 'Avertising Expenses/Sales',
                      'Labor Expenses/Sales', 'Accruals/Average Assets', 'Price/Book',
                      'Trailing P/E to Growth (PEG) ratio', 'Dividend Yield']

    delete_partial = ['Shillers Cyclically Adjusted P/E Ratio', 'Book/Market',
                      'Price/Operating Earnings (Basic, Excl. EI)', 'Effective Tax Rate',
                      'Price/Operating Earnings (Diluted, Excl. EI)',
                      'P/E (Diluted, Excl. EI)', 'P/E (Diluted, Incl. EI)',
                      'Price/Sales', 'Price/Cash flow', 'Dividend Payout Ratio',
                      'Sales/Invested Capital', 'Sales/Stockholders Equity',
                      'Sales/Working Capital', 'Research and Development/Sales', 'Avertising Expenses/Sales',
                      'Labor Expenses/Sales', 'Accruals/Average Assets', 'Price/Book',
                      'Trailing P/E to Growth (PEG) ratio', 'Dividend Yield']

    include_only = ['Enterprise Value Multiple', 'Net Profit Margin',
                    'Operating Profit Margin Before Depreciation',
                    'Gross Profit Margin', 'Cash Flow Margin', 'Return on Assets',
                    'Return on Capital Employed', 'After-tax Return on Average Common Equity',
                    'After-tax Return on Total Stockholders Equity', 'Gross Profit/Total Assets',
                    'Common Equity/Invested Capital', 'Long-term Debt/Invested Capital',
                    'Total Debt/Invested Capital', 'Interest/Average Long-term Debt',
                    'Interest/Average Total Debt', 'Cash Balance/Total Liabilities',
                    'Total Debt/Total Assets', 'Total Debt/EBITDA',
                    'Current Liabilities/Total Liabilities', 'Long-term Debt/Total Liabilities',
                    'Profit Before Depreciation/Current Liabilities', 'Operating CF/Current Liabilities',
                    'Cash Flow/Total Debt', 'Free Cash Flow/Operating Cash Flow',
                    'Total Liabilities/Total Tangible Assets', 'Total Debt/Capital',
                    'Total Debt/Equity', 'After-tax Interest Coverage',
                    'Interest Coverage Ratio', 'Cash Ratio', 'Quick Ratio (Acid Test)',
                    'Current Ratio']

    include_only2 = ['Fiscal year end', 'Fiscal quarter end', 'Public Date',
                     'Enterprise Value Multiple', 'Net Profit Margin',
                     'Operating Profit Margin Before Depreciation', 'Gross Profit Margin',
                     'Return on Assets', 'Return on Equity',
                     'After-tax Return on Average Common Equity',
                     'After-tax Return on Invested Capital',
                     'Pre-tax return on Net Operating Assets',
                     'Pre-tax Return on Total Earning Assets', 'Gross Profit/Total Assets',
                     'Common Equity/Invested Capital', 'Long-term Debt/Invested Capital',
                     'Interest/Average Long-term Debt', 'Interest/Average Total Debt',
                     'Cash Balance/Total Liabilities', 'Inventory/Current Assets',
                     'Receivables/Current Assets', 'Total Debt/EBITDA',
                     'Short-Term Debt/Total Debt', 'Current Liabilities/Total Liabilities',
                     'Long-term Debt/Total Liabilities',
                     'Profit Before Depreciation/Current Liabilities',
                     'Cash Flow/Total Debt', 'Free Cash Flow/Operating Cash Flow',
                     'Total Liabilities/Total Tangible Assets', 'Long-term Debt/Book Equity',
                     'After-tax Interest Coverage', 'Quick Ratio (Acid Test)',
                     'Cash Conversion Cycle (Days)', 'Inventory Turnover', 'Asset Turnover',
                     'Receivables Turnover', 'Payables Turnover']

    df = pd.read_csv(file_name, delimiter=",", header='infer', infer_datetime_format=True, parse_dates=True,
                     dayfirst=True)
    headers = df.columns
    print(headers)
    if delete == 0:
        delete_columns = list(dict.fromkeys(delete_columns))  # eliminate duplicate entries
        df = df.drop(columns=delete_columns)
        n_deleted = len(delete_columns)
    elif delete == 1:  # computer help to delete remainder of highly correlated
        delete_partial = list(dict.fromkeys(delete_partial))  # eliminate duplicate entries
        df = df.drop(columns=delete_partial)
        n_deleted = len(delete_partial)
    else:
        delete_extras = []
        for h in headers[fidx[0]:fidx[1]]:
            if h not in include_only:
                delete_extras.append(h)
        delete_extras = list(dict.fromkeys(delete_extras))  # eliminate duplicate entries
        df = df.drop(columns=delete_extras)
        n_deleted = len(delete_extras)

    fidx[1] -= int(n_deleted)  # eliminate one column from data index
    fidx.append(df.shape[1])
    headers = df.columns
    HD = headers[fidx[0]:fidx[1]]
    fidx.append(fidx[2] + len(HD))

    for h in HD:
        df["{0}_".format(h)] = np.where(np.array(df[h]) > 0, 1, 0).tolist()
    headers = df.columns
    HF = headers[fidx[2]:fidx[3]]
    D = df[HD].to_numpy().astype("float32")  # financial ratios
    F = df[HF].to_numpy().astype("int32")  # flag 1, if data==0 or NaN

    n = int(F.shape[0] / 1000) + 1
    fig, ax = plt.subplots(nrows=n, ncols=1, figsize=(10, 5))
    plt.suptitle("Heat Map of Financial Ratios")
    fig.text(0.04, 0.5, "# Financial Ratios", va='center', rotation='vertical')
    plt.xlabel("# Companies")
    if n == 1:
        ax.imshow(F.T, cmap=cm.coolwarm, interpolation='nearest')
    else:
        for i in range(n):
            ax[i].imshow(F[i * 1000:min((i + 1) * 1000, F.shape[0])].T, cmap=cm.coolwarm, interpolation='nearest')
    show_or_save(name="plt", showGraphs=show)

    plt.title("Distribution of Financial Ratio per Company")
    plt.xlabel("# Financial Ratios")
    plt.ylabel("# Companies")
    plt.hist(F.sum(axis=1), bins=20)
    show_or_save(name="plt", showGraphs=show)

    DF_train = pd.DataFrame(df)
    # Calculate Covariance Matrix for Data and eliminate highly correlated financial ratios

    print(headers[fidx[0]:fidx[1]])
    print("Shape Before Eliminating NaN/)s: ", DF_train.shape)
    for fin_ratio_ in headers[fidx[2]:fidx[3]]:
        if fin_ratio_ in DF_train.columns:
            DF_train = DF_train[DF_train[fin_ratio_] > 0]  # Eliminate NaN & zeros

    print("Dimensions of dataset after deleting NaN/0s: ", DF_train.shape)

    # run once to choose one financial ratio vs others for highly correlated ratios
    # then store as inlude_only list and switch model variable in __main__ to delf = 2 (delete = 2)

    if delete == 1:
        DF_train, headers, fidx = runOnce(df, DF_train, headers, fidx, show)

    HD = DF_train.columns[fidx[0]:fidx[1]]
    HF = DF_train.columns[fidx[2]:fidx[3]]
    D_train = DF_train[HD].to_numpy().astype("float32")
    F_train = DF_train[HF].to_numpy().astype("int32")

    x = (D_train - D_train.mean(axis=0)) / D_train.std(axis=0)
    x_cov = np.cov(x.T)
    plot_confusion_matrix(x_cov, np.arange(x.shape[1]), title="Reduced Data Covariance Matrix",
                          x_label="Financial Ratios",
                          y_label="Financial Ratios", anot=False)
    show_or_save(name="plt", showGraphs=show)

    # flag_train = np.where(data_train == data_train, 1, np.where(data_train == 0, 0, 1))
    print("Reduced (excluding NaN/0s) data shape:\n", D_train.shape)
    print("Reduce number to {a} companies with fin ratios between {b} - {c}".format(a=F_train.sum(axis=0).min(),
                                                                                    b=F_train.sum(axis=1).min(),
                                                                                    c=F_train.sum(axis=1).max()))

    plt.title("Distribution of Financial Ratio per Company (reduced for NaN/0s)")
    plt.xlabel("# Financial Ratios")
    plt.ylabel("# Companies")
    plt.hist(F_train.sum(axis=1), bins=20)
    show_or_save(name="plt", showGraphs=show)

    n = int(F_train.shape[0] / 1000) + 1

    fig, ax = plt.subplots(nrows=n, ncols=1, figsize=(10, 5))
    plt.suptitle("Heat Map of Financial Ratios")
    fig.text(0.04, 0.5, "# Financial Ratios", va='center', rotation='vertical')
    plt.xlabel("# Companies")
    if n == 1:
        ax.imshow(F_train.T, cmap=cm.coolwarm, interpolation='nearest')
    else:
        for i in range(n):
            ax[i].imshow(F_train[i * 1000:min((i + 1) * 1000, F_train.shape[0])].T, cmap=cm.coolwarm,
                         interpolation='nearest')

    show_or_save(name="plt", showGraphs=show)

    return DF_train, D_train, F_train, fidx


if __name__ == "__main__":

    # model params

    showG = False  # Show Graphs (True)? or save them (False)?
    enl = False  # Use equal number of each label for training sample
    delf = 2  # 0=delete features, 1=delete features correlated, 2=use only these features
    classes = 3  # Number of classes used in classification for model

    #   Load Database
    #   Delete certain columns with low relevance, then
    #   Eliminate companies from datafile that don't have full feature data, then
    #   Eliminate features that are highly correlated with other fields, then
    #   Save clean dataset for training in file data_clean.csv
    #   delf=0 just deletes finratios that have poor data quality or not applicable
    #   delf=1 check which finratios are highly correlated > 0.80 and deletes those ratios
    #   delf=2 uses a specific combination of ratios

    df, data, flag, indx = load_clean_data("new_data.csv", [3, 71], show=showG, delete=delf)  # showG
    headers = df.columns
    d = indx[1] - indx[0] + 1  # number of features  (financial ratios)
    print(headers)

    # Create one-hot representation of rating labels for full ratings and main ratings

    sortedFullRatings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
                         'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'SD', 'D']
    sortedMainRatings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC/D']
    mapRatingMain = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7])
    if classes == 2:
        lessMainRatings = ['I', 'S']
        mapRatings = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    elif classes == 3:
        lessMainRatings = ['I', 'S', 'J']
        mapRatings = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    elif classes == 4:
        lessMainRatings = ['As', 'BBB', 'BB', "J"]
        mapRatings = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3])

    Y_full = np.zeros((data.shape[0], len(sortedFullRatings))).astype("int32")
    Y_main = np.zeros((data.shape[0], len(sortedMainRatings))).astype("int32")
    Y_less = np.zeros((data.shape[0], mapRatings.max() + 1)).astype("int32")

    for i in range(Y_full.shape[0]):
        Y_full[i, sortedFullRatings.index(df["Rating"][df.index[i]])] = 1
        Y_main[i, mapRatingMain[sortedFullRatings.index(df["Rating"][df.index[i]])]] = 1
        Y_less[i, mapRatings[sortedFullRatings.index(df["Rating"][df.index[i]])]] = 1

    Y_color = np.argmax(Y_less, axis=1)  # Use Y_less, Y_main, Y_full depending on how many labels
    print("Y shape:", Y_less.shape, "\n Y one-hot rep:\n", Y_less)

    # Center all data

    data_train_center = np.log(data) - np.log(data).mean(axis=0)  # (data - data.mean(axis=0))
    print(data_train_center.mean(axis=0))

    # check PCA() outcome below against my PCA() method
    # reduced_data, idx, evecs, evals = pca_code(data_train_center)

    # Calculate the Principal Components Analysis (PCA) from all financial ratios
    #
    # Linear dimensionality reduction using Singular Value Decomposition of the data to project it
    # to a lower dimensional space. The input data is centered but not scaled for each feature before
    # applying the SVD.

    pca = PCA()  # Initialize with n_components=x parameter to only find the top eigenvectors
    z = pca.fit_transform(data_train_center)

    # z = z / z.std(axis=0)   # normalise across companies for each reduced feature
    n_pcs = pca.components_.shape[0]
    threshold = np.where(np.cumsum(pca.explained_variance_ratio_) >= .95)[0][0] + 1
    print("95% variance explained by {s} components".format(s=threshold))

    # Neural Net Classifier

    batch_size_ = 64
    epochs_ = 2000
    lr = 0.001
    reg = 'l2'

    # Centered Data - X (features), Y (labels in one-hot encoding) datasets

    #X = data_train_center
    #Y = Y_less
    #d = X.shape[1]  # number of features
    #N = Y_less.shape[0]  # number observations
    #k = Y_color.max() + 1  # label categories

    #summaryNNClass(X, Y_less, k, Y_less.shape[0], lessMainRatings,
    #               fname="data_centered", title="Neural Nets: Centered Data", show=True,
    #               batch_size=batch_size_, epochs=epochs_, lr=0.001, reg='l1', equalNoLabels=enl)

    # PCA Adjusted (features), Y (labels in one-hot encoding) datasets

    X = z
    Y = Y_less
    d = X.shape[1]  # number of features
    N = Y_less.shape[0]  # number observations
    k = Y_color.max() + 1  # label categories

    summaryNNClass(X, Y_less, k, Y_less.shape[0], lessMainRatings,
                   fname="PC_adj_NN", title="Neural Nets: PCs Adjusted Data".format(threshold), show=True,
                   batch_size=batch_size_, epochs=epochs_, lr=0.001,  reg=reg, equalNoLabels=enl)

    # PCA Reduced to threshold (features), Y (labels in one-hot encoding) datasets

    X = z[:, 0:threshold]
    Y = Y_less
    d = X.shape[1]  # number of features
    N = Y_less.shape[0]  # number observations
    k = Y_color.max() + 1  # label categories

    summaryNNClass(X, Y_less, k, Y_less.shape[0], lessMainRatings,
                   fname="PC_reduced_NN", title="Neural Nets: {0} PCs Reduced Data".format(threshold), show=True,
                   batch_size=batch_size_, epochs=epochs_, lr=0.001,  reg='l2', equalNoLabels=enl)

