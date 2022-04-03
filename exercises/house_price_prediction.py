from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    features = full_data
    # features[['date']] = features[['date']].apply(pd.to_datetime)
    # features["date"] = pd.to_datetime(features["date"], format='%Y%m%d%T')
    # features["days"] = (features['date'] - pd.to_datetime('20140101', format='%Y%m%d')).dt.days
    para_list = ["zipcode"]
    features.loc[features.yr_renovated == 0, "yr_renovated"] = features["yr_built"]
    # features["sqft_living" < 0] = 0
    features = categorail_var(features, para_list)
    features = features.dropna()
    labels = features["price"]
    features = features.drop(columns=["date", "price", "id"])
    return features, labels


def categorail_var(features , str_var_list):
    for str_var in str_var_list:
        var = features[str_var].to_list()
        pd_var = pd.Series(var)
        df_var = pd.get_dummies(pd_var)
        features = pd.concat([features, df_var], axis='columns')
    features = features.drop(columns=str_var_list)
    return features


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    Xy = pd.concat([X, y], axis='columns')
    sigma = Xy.std()
    # sigma_y = y.std()
    # XY = X.dot(y)
    covariance = X.cov()
    dfA = pd.DataFrame(sigma)
    dfB = pd.DataFrame(sigma)
    sigma_pow = dfA.dot(dfB.T)
    # sigma_pow= sigma.T.dot(sigma)
    person = covariance/sigma_pow
    person.loc[person.price < 0, "price"] = person["price"]*-1
    for i in X.columns:
        plt.scatter(Xy[i], Xy["price"])
        plt.savefig(output_path + str(i) + "_vs_price.png")
    print("1")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, cancellation_labels = load_data("../datasets/house_prices.csv")
    feature_evaluation(df, cancellation_labels, "../figs/")
    train_X, train_y, test_X, test_y = split_train_test(df,cancellation_labels)

    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    p = np.arange(10, 101)

    raise NotImplementedError()
