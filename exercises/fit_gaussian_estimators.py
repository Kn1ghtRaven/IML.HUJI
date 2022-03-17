from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mu = 10
    sigma = 1
    number_of_samples = 1000
    samples = np.random.normal(mu, sigma, number_of_samples)
    univar = UnivariateGaussian()
    mues = np.zeros(int(number_of_samples / 10))
    sigmas = np.zeros(int(number_of_samples / 10))
    x = np.arange(10, 1001, 10)
    for i in range(1, 101):
        univar.fit(samples[0: (i * 10) - 1])
        mues[i - 1] = univar.mu_
        sigmas[i - 1] = univar.var_
    # Question 1 - Draw samples and print fitted model
    print((univar.mu_, univar.var_))
    # Question 2 - Empirically showing sample mean is consistent
    plt.plot(x, abs(mues - mu))
    plt.title("distance from Mue")
    plt.show()
    samples.sort()
    answers = univar.pdf(samples)
    # Question 3 - Plotting Empirical PDF of fitted model
    plt.plot(samples, answers)
    plt.title("sampales pdf")
    plt.xlabel("sampales")
    plt.ylabel("PDF value")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
             [0.2, 2, 0, 0],
             [0, 0, 1, 0],
             [0.5, 0, 0, 1]])
    multi_samples = np.random.multivariate_normal(mean, cov, 1000)
    multi = MultivariateGaussian()
    multi.fit(multi_samples)
    print(multi.mu_)
    print(multi.cov_)
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    maxrow = 0
    maxcol = 0
    valmax = -1 * np.inf
    df = np.zeros((np.shape(f1)[0], np.shape(f3)[0]))
    for row, i in enumerate(f1):
        for col, j in enumerate(f3):
            mean5 = np.array([i, 0, j, 0])
            df[row, col] = MultivariateGaussian.log_likelihood(mean5, cov, multi_samples)
            if valmax < df[row, col]:
                valmax = df[row, col]
                maxcol = col
                maxrow = row
    # Question 5 - Likelihood evaluation
    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    fig, ax = plt.subplots()
    ax.imshow(df.T, origin='lower', cmap='cubehelix', aspect='auto',
              interpolation='nearest', extent=[xmin, xmax, ymin, ymax])
    ax.axis([xmin, xmax, ymin, ymax])
    plt.title("heatmap of MultivariateGaussian")
    plt.xlabel("f1")
    plt.ylabel("f3")
    plt.show()
    # Question 6 - Maximum likelihood
    print((np.round(f1[maxrow], 3), np.round(f3[maxcol], 3)))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
