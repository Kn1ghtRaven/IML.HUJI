from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_: bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def change(self, selectedmu: float, selectedsigma: float, biased_var: bool = False):
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = True, selectedmu, selectedsigma

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        if not self.biased_:
            self.var_ = np.var(X)*(len(X)/(len(X)-1))
        else:
            self.var_ = np.var(X)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        return (1/np.sqrt(2*np.pi*self.var_))*np.exp(-1*((X - self.mu_)**2)/(2*self.var_))


    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        normal = UnivariateGaussian()
        normal.change(mu, sigma)
        samples = normal.pdf(X)
        cumsamples = np.cumprod(samples)
        return np.log(cumsamples[-1])


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        mue = np.zeros(np.shape(X)[1])
        univar = UnivariateGaussian()
        self.mu_ = np.mean(X, 0)
        # for k in range(np.shape(X)[1]):
        #     mue[k] = univar.fit(X[:, k]).mu_
        self.cov_ = np.cov(X, rowvar=False)
        # self.mu_ = mue
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        invcov = np.linalg.inv(self.cov_)
        detcov = np.linalg.det(self.cov_)
        number_of_fitures = np.shape(X)[1]
        return np.sqrt(1 / ((2 * np.pi)**number_of_fitures*detcov)) * np.exp(-1 / 2 * ((X - self.mu_).T * invcov*(X - self.mu_)))

    def change(self, mu: np.ndarray, cov: np.ndarray):
        self.fitted_, self.mu_, self.cov_ = True, mu, cov

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        normal = MultivariateGaussian()
        normal.change(mu, cov)
        samples = normal.pdf(X)
        cumsamples = np.cumprod(samples)
        return np.log(cumsamples[-1])

if __name__ == '__main__':
    mu = 10
    sigma = 1
    number_of_samples = 1000
    samples = np.random.normal(mu, sigma, number_of_samples)
    banana = UnivariateGaussian()

    # print(UnivariateGaussian.log_likelihood(mu, sigma, samples))
    mues = np.zeros(int(number_of_samples/10))
    sigmas = np.zeros(int(number_of_samples/10))
    x = np.arange(10, 1001, 10)
    for i in range(1, 101):
        banana.fit(samples[0: (i*10)-1])
        mues[i-1] = banana.mu_
        sigmas[i-1] = banana.var_
    print((banana.mu_, banana.var_))
    plt.plot(x, abs(mues-mu))
    plt.title("distance from Mue")
    plt.show()
    mean = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5],
             [0.2, 2, 0, 0],
             [0, 0, 1, 0],
             [0.5, 0, 0, 1]]
    multi_samples = np.random.multivariate_normal(mean, cov, 1000)
    multi = MultivariateGaussian()
    multi.fit(multi_samples)
    print(multi.mu_)
    print(multi.cov_)
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    f2 = np.zeros(20)
    f4 = np.zeros(20)
    mean5 = [f1, 0, f3, 0]
    Index = f1
    Cols = f3
    # df = DataFrame(MultivariateGaussian.log_likelihood(mean5, cov, multi_samples), index=Index, columns=Cols)
    df = np.zeros(max(f1.shape), max(f3.shape))
    for row, i in enumerate(f1):
        for col, j in enumerate(f3):
            mean5 = [i, 0, j, 0]

    sns.heatmap(df, annot=True)