from __future__ import annotations
from typing import NoReturn
# from . import LinearRegression
# from ...base import BaseEstimator
from IMLearn.learners.regressors import LinearRegression
from IMLearn import BaseEstimator
from IMLearn.metrics import mean_square_error
import numpy as np


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self._k = k
        self._liniar = LinearRegression(include_intercept=True)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        X_vandar = self.__transform(X)
        self._liniar.fit(X_vandar, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self._liniar.predict(self.__transform(X))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return self._liniar.loss(self.__transform(X), y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        return np.vander(X, self._k+1)

if __name__ == '__main__':
    poly = PolynomialFitting(2)
    np.random.seed(0)
    X = 5 * np.random.random_sample([5, 2])
    y = 5 * np.random.random_sample([5, 1])
    y = y.reshape([-1, 1])
    pred = 5 * np.random.random_sample([3, 2])
    # X = np.arange(1, 10)
    # X = np.concatenate([X, X], axis=1)
    # y = np.power(X, 3) + 20
    poly.fit(X, y)
    f1 = 11
    f2 = 12
    pred = np.array([f1, f2])
    print(poly.predict(pred))
    print(f1*f1*f1)
    print(f2 * f2 * f2)