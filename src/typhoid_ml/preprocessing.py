"""Preprocessing stages: imputation, ordinal encoding, SMOTENC, scaling/one-hot.

The ordering matters. SMOTENC requires a dense numeric matrix in which the
categorical columns are identifiable by index, so raw mixed-type features are
first imputed and ordinal-encoded. Synthetic minority samples are then generated
*inside the pipeline*, which guarantees that resampling only ever sees the
training fold and never leaks into validation or test data. Only afterwards are
the categorical columns one-hot expanded and the numeric columns standardised,
because an SVM is sensitive to feature magnitude and to spurious ordinal
relationships between category codes.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTENC
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class OrdinalPreprocessor(BaseEstimator, TransformerMixin):
    """Impute and ordinal-encode mixed raw features so SMOTENC can consume them.

    Output column order is always [numeric..., categorical...].
    """

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        self.numeric_features_ = list(self.numeric_features)
        self.categorical_features_ = list(self.categorical_features)
        self.num_imputer_ = SimpleImputer(strategy="median")
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
        self.ordinal_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        self.num_imputer_.fit(X[self.numeric_features_])
        cat_imputed = self.cat_imputer_.fit_transform(X[self.categorical_features_].astype(object))
        self.ordinal_.fit(cat_imputed)

        self.n_numeric_ = len(self.numeric_features_)
        self.n_categorical_ = len(self.categorical_features_)
        return self

    def transform(self, X):
        num = self.num_imputer_.transform(X[self.numeric_features_])
        cat = self.cat_imputer_.transform(X[self.categorical_features_].astype(object))
        cat = self.ordinal_.transform(cat)
        return np.column_stack([num, cat]).astype(float)

    @property
    def categorical_indices_(self):
        return list(range(self.n_numeric_, self.n_numeric_ + self.n_categorical_))


class SmoteNCResampler(BaseEstimator):
    """SMOTENC wrapper that knows where the categorical columns are.

    Implemented as an imblearn-compatible sampler so it participates in the
    pipeline and is applied to training folds only.
    """

    _estimator_type = "sampler"
    _sampling_type = "over-sampling"

    def __init__(self, n_numeric, n_categorical, random_state=42, k_neighbors=5):
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def _sampler(self):
        return SMOTENC(
            categorical_features=list(range(self.n_numeric, self.n_numeric + self.n_categorical)),
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )

    def fit_resample(self, X, y):
        return self._sampler().fit_resample(X, y)

    def fit(self, X, y=None):
        return self


def smotenc_resample(X, y, n_numeric, n_categorical, random_state=42, k_neighbors=5):
    """Functional form, used by scripts and tests."""
    return SmoteNCResampler(n_numeric, n_categorical, random_state, k_neighbors).fit_resample(X, y)


class PostSMOTEEncoder(BaseEstimator, TransformerMixin):
    """Standardise numeric columns and one-hot encode the ordinal category codes."""

    def __init__(self, n_numeric, n_categorical, categorical_names=None):
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.categorical_names = categorical_names

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.scaler_.fit(X[:, : self.n_numeric])
        self.ohe_.fit(X[:, self.n_numeric :])
        return self

    def transform(self, X):
        num = self.scaler_.transform(X[:, : self.n_numeric])
        cat = self.ohe_.transform(X[:, self.n_numeric :])
        return sparse.hstack([sparse.csr_matrix(num), cat], format="csr")

    def get_feature_names_out(self, input_features=None):
        num_names = list(input_features[: self.n_numeric]) if input_features is not None else [
            f"num_{i}" for i in range(self.n_numeric)
        ]
        cat_base = (
            list(self.categorical_names)
            if self.categorical_names is not None
            else [f"cat_{i}" for i in range(self.n_categorical)]
        )
        return np.array(num_names + list(self.ohe_.get_feature_names_out(cat_base)))
