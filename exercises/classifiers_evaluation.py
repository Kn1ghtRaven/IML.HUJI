from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

from IMLearn.learners.classifiers.perceptron import default_callback

pio.templates.default = "simple_white"


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
    full_data = np.load(filename)
    X = full_data[:, 0:2]
    y = full_data[:, 2]
    return X, y


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
        # for iteretion in range(1, iteretions+1):
        #     losses_per_iter = []
        #     model = Perceptron(max_iter=iteretion)
        #     # Fit Perceptron and record loss in each fit iteration
        #     model.fit(data, label)
        #     for index, row in enumerate(data):
        #         losses_per_iter.append(default_callback(model, row, label[index]))
        #     losses.append(np.sum(losses_per_iter)/len(losses_per_iter)) # normalize the number from 0 to 300 to 0. to 1
        # Plot figure
        fig = px.line(x=np.arange(1, len(losses)+1), y=losses)
        fig.update_layout(
            yaxis_title='training loss values',
            xaxis_title='iteretions',
            title=n,
        )
        # fig.show()
        fig.write_image("../images/ex3q1_"+n+".png")
        print(n)
        print(losses)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        raise NotImplementedError()

        # Fit models and predict over training set
        raise NotImplementedError()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
