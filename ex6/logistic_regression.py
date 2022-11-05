from typing import NoReturn

import numpy as np

from IMLearn import BaseEstimator
from IMLearn.desent_methods import GradientDescent
from IMLearn.desent_methods.modules import LogisticModule, RegularizedModule, L1, L2


class LogisticRegression(BaseEstimator):

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.optimizer_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        if self.include_intercept_:
            X = np.c_[np.ones(len(X)), X]

        d = X.shape[1]
        self.coefs_ = np.random.normal(0, 1, d) / np.sqrt(d)

        if self.penalty_ == 'l1':
            model = RegularizedModule(fidelity_module=LogisticModule(weights=self.coefs_),
                                      regularization_module=L1(weights=self.coefs_[1:]),
                                      lam=self.lam_,
                                      weights=self.coefs_,
                                      include_intercept=self.include_intercept_)
        elif self.penalty_ == 'l2':
            model = RegularizedModule(fidelity_module=LogisticModule(weights=self.coefs_),
                                      regularization_module=L2(weights=self.coefs_[1:]),
                                      lam=self.lam_,
                                      weights=self.coefs_,
                                      include_intercept=self.include_intercept_)
        else:
            model = RegularizedModule(fidelity_module=LogisticModule(weights=self.coefs_),
                                      regularization_module=L2(weights=self.coefs_[1:]),
                                      lam=0,
                                      weights=self.coefs_,
                                      include_intercept=self.include_intercept_)

        self.solver_.fit(f=model, X=X, y=y)

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
            X = np.c_[np.ones(len(X)), X]

        y_hat = X @ self.coefs_
        y_hat[np.where(y_hat >= self.alpha_)] = 1
        y_hat[np.where(y_hat < self.alpha_)] = 0
        return y_hat

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        if self.include_intercept_:
            X = np.c_[np.ones(len(X)), X]

        return 1 / (1 + np.exp(-(X @ self.coefs_)))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        from ...metrics import mean_square_error
        return mean_square_error(y, self._predict(X))
