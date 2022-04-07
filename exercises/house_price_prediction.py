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



from sklearn import linear_model

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
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    features = full_data
    bad_rows_if_negative = ["bathrooms", "floors", "sqft_basement",
                                   "yr_renovated"]
    bad_rows_if_not_positive = ["price", "sqft_living", "sqft_lot",
                               "sqft_above", "yr_built", "sqft_living15",
                               "sqft_lot15"]
    for col in bad_rows_if_negative:
        features = features[features[col] >= 0]
    for col in bad_rows_if_not_positive:
        features = features[features[col] > 0]
    Start = pd.to_datetime(features["date"])
    End = pd.to_datetime('2014-01-01 00:00', format='%Y%m%d%T', errors='ignore')
    features["days"] = (Start - pd.to_datetime(End)).dt.days
    features = features[features["view"].isin(range(0, 5)) & features[
        "condition"].isin(range(3, 6)) & features["grade"].isin(range(5, 12))]
    features = features[features["bedrooms"] < 7]
    features = features[features["bathrooms"] < 4]
    features = features[features["sqft_living"] < 5000]
    features = features[features["floors"] < 4]
    features = features[features["sqft_lot"] < 84000]
    features = features[features["sqft_lot15"] < 1000000]
    features = features[features["sqft_above"] < 3050]
    # features = features[features["sqft_above"] < 1000]
    # para_list = ["zipcode"]
    features.loc[features.yr_renovated == 0, "yr_renovated"] = features["yr_built"]
    # features = categorail_var(features, para_list)
    features = features.dropna()
    labels = features["price"]
    features = features.drop(columns=["date", "price", "id", "lat", "long", "zipcode"])
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
    covariance = Xy.cov()
    dfA = pd.DataFrame(sigma)
    dfB = pd.DataFrame(sigma)
    sigma_pow = dfA.dot(dfB.T)
    person = covariance/sigma_pow
    person.loc[person.price < 0, "price"] = person["price"]*-1
    for i in X.columns:
        fig = px.scatter(Xy, x=i, y="price", title="the Pearson Correlation :" + str(person[i]["price"]))
        fig.write_image(output_path + str(i) + "_vs_price.png")



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, cancellation_labels = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, cancellation_labels, "../figs/")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    present = np.arange(10, 101)
    reps = np.arange(10)
    mue = []
    std = []
    confidence_up = []
    confidence_down = []
    # lin_reg = linear_model.LinearRegression()
    for p in present:
        lin_reg = LinearRegression()
        sample_loss = np.empty([0])
        for r in reps:
            train_X_change, train_y_change, test_X_change, test_y_change = split_train_test(train_X, train_y, p/100)
            lin_reg.fit(train_X_change, train_y_change)
            sample_loss = np.append(sample_loss, lin_reg.loss(test_X, test_y))
        mue.append(np.mean(sample_loss))
        std.append(np.std(sample_loss))
        confidence_up.append(mue[-1] + 2*std[-1])
        confidence_down.append(mue[-1] - 2 * std[-1])
    pan_mue = pd.Series(mue)
    pan_confidence_up = pd.Series(confidence_up)
    pan_confidence_down = pd.Series(confidence_down)
    df["mue"] = pan_mue
    df["confidence_up"] = pan_confidence_up
    df["confidence_down"] = pan_confidence_down
    fig = go.Figure([
        go.Scatter(
            name='MeanError',
            x=present,
            y=df['mue'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=present,
            y=df['confidence_up'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=present,
            y=df['confidence_down'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        yaxis_title='loss in test set',
        xaxis_title='present',
        title=' mean loss as a function of p% with confidence interval',
        hovermode="x"
    )
    # fig.update_yaxes(range=[-40000000000, 40000000000])
    # fig.update_xaxes(range=[10, 100])
    # fig.update()
    fig.write_image("../images/ex2q14graph.png")
    # fig.show()
    # fig = px.line(df, x=present, y=mue)
    # fig.write_image()
    plt.plot(present, mue)
    plt.plot(present, confidence_up)
    plt.plot(present, confidence_down)
    print(cancellation_labels.mean())
    print(cancellation_labels.var())
    print(np.sqrt(mue[-1]))
    plt.show()
