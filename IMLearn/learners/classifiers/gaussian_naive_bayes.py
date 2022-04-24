from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        #find classes
        self.classes_ = np.unique(y)
        #find pi
        self.pi_ = np.zeros(shape=np.shape(self.classes_))
        for index, value in enumerate(self.classes_):
            self.pi_[index] = np.sum(y == value)
        #find mu
        y = y.reshape((-1, 1))
        full_data = np.concatenate((X, y), axis=1)
        self.mu_ = np.zeros((np.shape(self.classes_)[0], np.shape(X)[1]))
        dist_from_mu = full_data.copy()
        self.vars_ = np.zeros(shape=(np.shape(self.classes_)[0], np.shape(X)[1]))
        for index, value in enumerate(self.classes_):
            self.mu_[index] = np.sum(full_data[full_data[:, -1] == value][:, :-1], axis=0) / self.pi_[index]
            dist_from_mu[dist_from_mu[:, -1] == value][:, :-1] = (dist_from_mu[dist_from_mu[:, -1] == value][:, :-1] - self.mu_[index])/np.sqrt(self.pi_[index])  # i dont know if this will work need to debug
        #find vars
        cov = dist_from_mu[:, :-1].T @ dist_from_mu[:, :-1]
        # cov = np.sum(cov)
        self.vars_ = np.diag(cov)
        #normalize pi
        self.pi_ = self.pi_ / len(y)


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
        return self.likelihood(X).max(1)


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        number_of_fitures = np.shape(X)[1]
        likelihoods = np.zeros((np.shape(X)[0], np.shape(self.classes_)[0]))
        for index, value in enumerate(self.classes_):
            cov = np.diag(np.diag(self.vars_[index].reshape(-1, 1))) # TODO fix this line it does not work need to put the values on the diag of a matrix
            cov_inv = np.linalg.inv(cov)
            likelihoods[:, index] = np.sqrt(1 / ((2 * np.pi)**number_of_fitures * np.linalg.det(cov))) * np.exp(-1 / 2 * ((X - self.mu_[index]) @ cov_inv @(X - self.mu_[index]).T))
        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))
