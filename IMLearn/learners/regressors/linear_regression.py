from __future__ import annotations
from typing import NoReturn
# from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv

from IMLearn import BaseEstimator
from IMLearn.metrics import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            intercept = np.ones([np.shape(X)[0], 1])
            X = np.concatenate((intercept, X), axis=1)
        U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
        sigma_inv = np.linalg.inv(np.diag(sigma))
        self.coefs_ = Vt.T @ sigma_inv @ U.T @ y

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
        if self.include_intercept_:
            intercept = np.ones([np.shape(X)[0], 1])
            X = np.concatenate((intercept, X), axis=1)
        return X @ self.coefs_

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
        theta = X @ self.coefs_
        return mean_square_error(y, theta)

if __name__ == '__main__':
    lin = LinearRegression()
    X = np.array([1, 2, 3])
    X = X.reshape([-1, 1])
    pred = np.array([4, 5, 6])
    pred = pred.reshape([-1, 1])
    y = np.array([2, 3, 4])
    y = y.reshape([-1, 1])
    lin.fit(X, y)
    print(lin.predict(pred))