from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    X = np.random.normal(loc=mu, scale=sigma, size=1000)
    fit = UnivariateGaussian().fit(X)
    print((fit.mu_, fit.var_))

    # Question 2 - Empirically showing sample mean is consistent
    mu_hat = [np.abs(mu - UnivariateGaussian().fit(X[:n]).mu_)
              for n in np.arange(1, len(X)/10, dtype=np.int)*10]
    go.Figure(go.Scatter(x=list(range(len(mu_hat))), y=mu_hat, mode="markers", marker=dict(color="black")),
              layout=dict(template="simple_white",
                          title="Deviation of Sample Mean Estimation As Function of Sample Size",
                          xaxis_title=r"$\text{Sample Size }n$",
                          yaxis_title=r"$\text{Sample Mean Estimator }\hat{\mu}_n$"))\
        .write_image("mean.deviation.over.sample.size.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = np.c_[X, fit.pdf(X)]
    pdfs = pdfs[pdfs[:, 1].argsort()]
    go.Figure(go.Scatter(x=pdfs[:, 0], y=pdfs[:, 1], mode="markers", marker=dict(color="black")),
              layout=dict(template="simple_white",
                          title="Empirical PDF Using Fitted Model",
                          xaxis_title=r"$x$",
                          yaxis_title=r"$\mathcal{N}(x;\hat{\mu},\hat{\sigma}^2)$"))\
        .write_image("empirical.pdf.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov = np.array([[1, .2, 0, .5], [.2, 2, 0, 0], [0, 0, 1, 0], [.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean=[0, 0, 4, 0], cov=cov, size=1000)
    fit = MultivariateGaussian().fit(X)
    print(np.round(fit.mu_, 3))
    print(np.round(fit.cov_, 3))
    fit.pdf(X)
    # Question 5 - Likelihood evaluation
    ll = np.zeros((200, 200))
    f_vals = np.linspace(-10, 10, ll.shape[0])
    for i, f1 in enumerate(f_vals):
        for j, f3 in enumerate(f_vals):
            ll[i, j] = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, X)

    go.Figure(go.Heatmap(x=f_vals, y=f_vals, z=ll),
              layout=dict(template="simple_white",
                          title="Log-Likelihood of Multivatiate Gaussian As Function of Expectation of Feautures 1,3",
                          xaxis_title=r"$\mu_3$",
                          yaxis_title=r"$\mu_1$"))\
        .write_image("loglikelihood.heatmap.png")

    # Question 6 - Maximum likelihood
    print("Setup of maximum likelihood (features 1 and 3):",
          np.round(f_vals[list(np.unravel_index(ll.argmax(), ll.shape))], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
