"""Scikit-learn pipelines for price change regression and the persistence baseline."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import BASE, NEW


class PriceChangeRegressor(RegressorMixin, BaseEstimator):
    """Fit log price changes and return prices in rubles."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        if self.estimator is not None:
            self.estimator_ = clone(self.estimator).fit(X[:, :-1], np.log(y / X[:, -1]))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        price = X[:, -1]
        if self.estimator is None:
            return price.copy()
        return price * np.exp(self.estimator_.predict(X[:, :-1]))


def make_pipeline(estimator, *, extended=True, scale=False):
    if estimator is None:
        columns = ColumnTransformer([("anchor", "passthrough", ["current_price_rub"])])
    else:
        numeric = [("fill", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
        if scale:
            numeric.append(("scale", StandardScaler()))
        category = Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="не указан")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        columns = ColumnTransformer([
            ("numeric", Pipeline(numeric), [*BASE, *NEW] if extended else BASE),
            ("brand", category, ["brand"]),
            ("anchor", "passthrough", ["current_price_rub"]),
        ])
    return Pipeline([("columns", columns), ("price", PriceChangeRegressor(estimator))])
