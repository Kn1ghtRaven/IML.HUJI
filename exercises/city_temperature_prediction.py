import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename,parse_dates=["Date"]).dropna().drop_duplicates()
    df= full_data
    df = pd.get_dummies(df, prefix='City_', columns=['City'])
    df.dropna()
    df = df[df["Temp"] > -50]
    df["DayOfYear"] = df["Date"].dt.dayofyear
    labels = df["Temp"]
    df = df.drop(columns=["Temp", "Date"])
    return df, labels

def group_by_country(df, labels, country):
    full_data = pd.concat([df, labels], axis='columns')
    full_data = full_data[full_data["Country"] == country]
    return full_data



if __name__ == '__main__':
    np.random.seed(0)
    # # Question 1 - Load and preprocessing of city temperature dataset
    df, labels = load_data("../datasets/City_Temperature.csv")
    full_data = group_by_country(df, labels, "Israel")
    # # Question 2 - Exploring data for specific country
    fig = px.scatter(full_data[["DayOfYear", 'Temp', 'Year']], x="DayOfYear", y="Temp", color=full_data['Year'].astype('category'))
    fig.update_layout(
        yaxis_title='Temp ',
        xaxis_title='DayOfYear',
        title='daily temperatures in Israel over the years',
        hovermode="x"
    )
    fig.write_image("../images/ex2q22graph.png")
    second_graph = full_data[['Month', 'Temp']]
    Temp_month_std = second_graph.groupby("Month").Temp.agg("std")
    fig2 = px.bar(Temp_month_std)
    fig2.update_layout(
        yaxis_title='Temp ',
        xaxis_title='Month',
        title='standard deviation of daily temperatures for each month in Israel',
        hovermode="x"
    )
    plt.show()
    fig2.write_image("../images/ex2q22graph2.png")


    # Question 3 - Exploring differences between countries
    graph_list = []
    for i in ["South Africa", "Israel", "Jordan", "The Netherlands"]:
        full_data = group_by_country(df, labels, i)
        trd_graph = full_data.groupby(['Month']).agg({'Temp': ['std', 'mean']})
        graph_list.append(go.Scatter(
        name=i,
        y=trd_graph[('Temp', 'mean')],
        error_y=dict(
            type='data',
            symmetric=True,
            array=trd_graph[('Temp', 'std')],),
        mode='lines'))
    fig3 = go.Figure(graph_list)
    fig3.write_image("../images/ex2q23graph.png")

    # Question 4 - Fitting model for different values of `k`
    data_IL = group_by_country(df, labels, "Israel")
    train_X, train_y, test_X, test_y = split_train_test(data_IL["DayOfYear"], data_IL["Temp"])
    level = range(1, 11)
    loss = np.zeros(len(level))
    min_k, min_loss = 0, np.inf
    for i, k in enumerate(level):
        ploy_reg = PolynomialFitting(k)
        ploy_reg.fit(train_X, train_y)
        loss[k-1] = np.round(ploy_reg.loss(test_X, test_y), decimals=2)
        print("the loss for degree " + str(k) + " is : ")
        print(loss[k-1])
        if loss[k-1] < min_loss:
            min_k, min_loss = k, loss[k-1]
    print("the best k is : " + str(min_k) + " with loss : " + str(min_loss))
    fig4 = px.bar(x=range(1, 11), y=loss)
    fig4.update_layout(
        yaxis_title='loss MSE ',
        xaxis_title='deg of ploy',
        title='test error recorded for each deg of ploy fit',
        hovermode="x"
    )
    fig4.write_image("../images/ex2q24graph.png")

    # Question 5 - Evaluating fitted model on different countries
    ploy_reg = PolynomialFitting(min_k)
    ploy_reg.fit(data_IL["DayOfYear"], data_IL["Temp"])
    c_loss = np.zeros(3)
    index = 0
    nations = ["South Africa", "Jordan", "The Netherlands"]
    for i in nations:
        data_country = group_by_country(df, labels, i)
        c_loss[index] = np.round(ploy_reg.loss(data_country["DayOfYear"], data_country["Temp"]), decimals=2)
        index = index + 1

    fig5 = px.bar(x=nations, y=c_loss)
    fig5.update_layout(
        yaxis_title='loss MSE ',
        xaxis_title='Country',
        title='error recorded for each country based of the data in IL',
        hovermode="x"
    )
    fig5.write_image("../images/ex2q25graph.png")