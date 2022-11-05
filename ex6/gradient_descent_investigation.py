import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from utils import *

np.random.seed(0)


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_ = []
    weights_ = []

    def callback(solver, weights, val, grad, t, eta, delta):
        values_.append(float(val))
        weights_.append(weights)

    return callback, values_, weights_


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        l1, l2 = L1(init), L2(init)
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=FixedLR(eta), callback=callback_l1).fit(f=l1, X=None, y=None)
        GradientDescent(learning_rate=FixedLR(eta), callback=callback_l2).fit(f=l2, X=None, y=None)
        plot_descent_path(module=L1, descent_path=np.array(weights_l1), title='L1 ETA = {0}'.format(eta)).show()
        plot_descent_path(module=L2, descent_path=np.array(weights_l2), title='L2 ETA = {0}'.format(eta)).show()

        norm_l1 = [np.linalg.norm(weights_l1[i]) for i in range(len(weights_l1))]
        norm_l2 = [np.linalg.norm(weights_l2[i]) for i in range(len(weights_l2))]
        go.Figure(
            [go.Scatter(x=np.arange(1, len(weights_l1) + 1), y=norm_l1, mode='markers+lines', name=r'$\text{L1}$'),
             go.Scatter(x=np.arange(1, len(weights_l2) + 1), y=norm_l2, mode='markers+lines', name=r'$\text{L2}$')]) \
            .update_layout(title=rf"$\textbf{{The Convergence Rate ETA = {eta}$").show()
        # todo find the lowest loss


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    norm = []
    for gamma in gammas:
        l1, l2 = L1(init), L2(init)
        callback, values, weights = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback).fit(f=l1, X=None, y=None)
        norm.append([np.linalg.norm(weights[i]) for i in range(len(weights))])

    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure([
        go.Scatter(x=np.arange(1, len(norm[0]) + 1), y=norm[0], mode='markers+lines', name=r'$\gamma=0.9$'),
        go.Scatter(x=np.arange(1, len(norm[1]) + 1), y=norm[1], mode='markers+lines', name=r'$\gamma=0.95$'),
        go.Scatter(x=np.arange(1, len(norm[2]) + 1), y=norm[2], mode='markers+lines', name=r'$\gamma=0.99$'),
        go.Scatter(x=np.arange(1, len(norm[3]) + 1), y=norm[3], mode='markers+lines', name=r'$\gamma=1$')])
    fig.update_layout(title=rf"$\textbf{{The Convergence Rate ETA$").show()

    # Plot descent path for gamma=0.95
    gamma = 0.95
    l1, l2 = L1(init), L2(init)
    callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
    callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
    GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback_l1).fit(f=l1, X=None, y=None)
    GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback_l2).fit(f=l2, X=None, y=None)
    plot_descent_path(module=L2, descent_path=np.array(weights_l2), title='L2 (black) VS L1 (green)'.format(eta)) \
        .add_trace(go.Scatter(x=np.array(weights_l1)[:, 0], y=np.array(weights_l1)[:, 1], mode="markers+lines")).show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def foo(solver, weights, val, grad, t, eta, delta):
    return None


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    gd = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000, out_type="last", callback=foo)
    model = LogisticRegression(include_intercept=True, solver=gd, penalty="none", lam=1, alpha=.5)
    model.fit(X=X_train.values, y=y_train.values)
    from sklearn.metrics import roc_curve, auc
    c = [custom[0], custom[-1]]
    fpr, tpr, thresholds = roc_curve(y_train.values, model.predict_proba(X_train.values))

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    alpha_star = tpr[np.argmax(tpr - fpr)]
    print(alpha_star)
    model.alpha_ = alpha_star
    loss = model.loss(X=X_test.values, y=y_test.values)
    print(loss)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    from IMLearn.model_selection.cross_validate import cross_validate
    from IMLearn.metrics import mean_square_error

    lams = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    ridge_scores, lasso_scores = np.zeros((len(lams), 2)), np.zeros((len(lams), 2))
    for i, lam in enumerate(lams):
        print(lam)
        logistic_l1 = LogisticRegression(include_intercept=True, solver=gd, penalty="l1", lam=lam, alpha=.5)
        print('done L1')
        logistic_l2 = LogisticRegression(include_intercept=True, solver=gd, penalty="l2", lam=lam, alpha=.5)
        print('done L2')
        ridge_scores[i] = cross_validate(estimator=logistic_l1, X=X_train.values,
                                         y=y_train.values, scoring=mean_square_error)
        lasso_scores[i] = cross_validate(estimator=logistic_l2, X=X_train.values,
                                         y=y_train.values, scoring=mean_square_error)

    make_subplots(1, 2, subplot_titles=[r"$\text{Ridge Regression}$", r"$\text{Lasso Regression}$"],
                  shared_xaxes=True) \
        .update_layout(title=r"$\text{Train and Validation Errors (averaged over the k-folds)}$", width=750, height=300) \
        .update_xaxes(title=r"$\lambda\text{ - Regularization parameter}$") \
        .add_traces([go.Scatter(x=lams, y=ridge_scores[:, 0], name="Ridge Train Error"),
                     go.Scatter(x=lams, y=ridge_scores[:, 1], name="Ridge Validation Error"),
                     go.Scatter(x=lams, y=lasso_scores[:, 0], name="Lasso Train Error"),
                     go.Scatter(x=lams, y=lasso_scores[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2]).show()

    l1_lam_star = lams[np.argmin(lasso_scores[:, 1])]
    l2_lam_star = lams[np.argmin(ridge_scores[:, 1])]
    print(l1_lam_star, l2_lam_star)

    logistic_l1 = LogisticRegression(include_intercept=True, solver=gd, penalty="l1", lam=l1_lam_star, alpha=.5)
    logistic_l2 = LogisticRegression(include_intercept=True, solver=gd, penalty="l2", lam=l2_lam_star, alpha=.5)
    logistic_l1.fit(X=X_train.values, y=y_train.values)
    logistic_l2.fit(X=X_train.values, y=y_train.values)
    l1_loss = logistic_l1.loss(X=X_test.values, y=y_test.values)
    l2_loss = logistic_l2.loss(X=X_test.values, y=y_test.values)
    print(l1_loss, l2_loss)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
