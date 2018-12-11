import numpy as np
import matplotlib.pyplot as plt1
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt1.figure()
    plt1.title(title)
    if ylim is not None:
        plt1.ylim(*ylim)
    plt1.xlabel("Training examples")
    plt1.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt1.grid()

    plt1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt1.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt1.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt1.legend(loc="best")
    return plt1


def plot_validation_curve(clf,X,y,param,para_range,n):
    train_scores, test_scores = validation_curve(clf, X, y, param_name=param, param_range=para_range,cv=n,scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt1.title("Validation Curve")
    plt1.xlabel(param)
    plt1.ylabel("Score")
    plt1.ylim(0.0, 1.1)
    lw = 2
    plt1.semilogx(para_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt1.fill_between(para_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt1.semilogx(para_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt1.fill_between(para_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt1.legend(loc="best")
    plt1.show()