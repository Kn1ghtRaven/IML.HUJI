from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from IMLearn.learners.classifiers.perceptron import default_callback
from math import atan2, pi
pio.templates.default = "simple_white"

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")



def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)



def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    iteretions = 1000
    folder = "../datasets/"
    for n, f in [("Linearly Separable", folder + "linearly_separable.npy"), ("Linearly Inseparable", folder + "linearly_inseparable.npy")]:
        # Load dataset
        data, label = load_dataset(f)
        losses = []

        def callback(presp: Perceptron, x, y):
            loss_per_iter = presp.loss(data, label)
            losses.append(loss_per_iter)
            return loss_per_iter

        model = Perceptron(max_iter=iteretions, callback=callback)
        model.fit(data, label)
        # Plot figure
        fig = px.line(x=np.arange(1, len(losses)+1), y=losses)
        fig.update_layout(
            yaxis_title='training loss values',
            xaxis_title='iteretions',
            title=n,
        )
        fig.show()
        fig.write_image("../images/ex3q1_"+n+".png")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    folder = "../datasets/"
    for f in [ "gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data, label = load_dataset(folder + f)
        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        lda = LDA()
        gnb.fit(data, label)
        lda.fit(data, label)
        pred_gnb = gnb.predict(data)
        pred_lda = lda.predict(data)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        gnb_acc = accuracy(label, pred_gnb)
        lda_acc = accuracy(label, pred_lda)
        df = pd.DataFrame()
        symbols = np.array(["circle", "diamond", "triangle-up"])
        df["x"] = data[:, 0]
        df["y"] = data[:, 1]
        df["pred_gnb"] = pred_gnb
        df["lable"] = label
        df["pred_lda"] = pred_lda
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Naive Bayes: accuracy ="+ str(gnb_acc), "LDA : accuracy = "+str(lda_acc)))
        #plot GNB
        fig.add_trace(
            go.Scatter(x=df["x"], y=df["y"], mode="markers", marker=dict(color=df["pred_gnb"], symbol=symbols[df["lable"].astype(int)])),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers", marker=dict(color="black", symbol="x")), row=1, col=1)
        for i in range(np.shape(gnb.mu_)[0]):
            fig.add_trace(get_ellipse(gnb.mu_[i], gnb.vars_[i] * np.identity(np.shape(gnb.vars_[i].reshape(-1, 1))[0])), row=1, col=1)
        #plot LDA
        fig.add_trace(
            go.Scatter(x=df["x"], y=df["y"], mode="markers", marker=dict(color=df["pred_lda"], symbol=symbols[df["lable"].astype(int)])),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                       marker=dict(color="black", symbol="x")), row=1, col=2)
        for i in range(np.shape(lda.classes_)[0]):
            fig.add_trace(get_ellipse(lda.mu_[i, :], lda.cov_), row=1, col=2)
        fig.update_layout( title_text="Data Set : "+f, showlegend=False)
        fig.write_image("../images/ex3q2_"+f+".png")
        fig.show()
        # print("loss in {} GNB : {}".format(f, gnb.loss(data, label)))
        # print("loss in {} LDA : {} ".format(f, lda.loss(data, label)))

def quizz():
    Y = np.array([0, 0, 1, 1, 1, 1])
    X = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    gnb = GaussianNaiveBayes()
    gnb.fit(X, Y)
    Y2 = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    X2 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    X2 = X2.reshape((-1, 1))
    gnb2 = GaussianNaiveBayes()
    gnb2.fit(X2, Y2)
    print("Question 1 : ")
    print("gnb probability for class 0 is :%f" % gnb2.pi_[0])
    print("gnb mu class 1 :%f" % gnb2.mu_[1])
    print("Question 2 : ")
    print("gnb var[0, 0] :%f" % gnb.vars_[0, 0])
    print("gnb var[1, 0] :%f" % gnb.vars_[1, 0])
    print("Question 3 : ")
    print("gnb mu class 2 :%f" % gnb2.mu_[2])

    iteretions = 1000
    folder = "../datasets/"
    for n, f in [("Linearly Separable", folder + "linearly_separable.npy")]:
        # Load dataset
        data, label = load_dataset(f)
        losses = [0]
        got_to_0 = [False]
        def callback(presp: Perceptron, x, y):
            loss_per_iter = presp.loss(data, label)
            if loss_per_iter > 0 and not got_to_0[0]:
                losses[0] += 1
            else:
                got_to_0[0] = True
            return loss_per_iter

        model = Perceptron(max_iter=iteretions, callback=callback)
        model.fit(data, label)
    print("Question 4 : ")
    print("the number of iteretions to get to 0 is : %f" % losses[0])

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    # quizz()
