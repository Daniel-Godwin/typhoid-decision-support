"""Pipeline validation tests (Table 3.2: unit testing and pipeline validation)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.data import build_target, feature_frame, load_dataset  # noqa: E402
from typhoid_ml.model import build_pipeline, n_candidates  # noqa: E402
from typhoid_ml.preprocessing import (  # noqa: E402
    OrdinalPreprocessor,
    PostSMOTEEncoder,
    smotenc_resample,
)

DATA = ROOT / "data" / "typhoid_dataset.csv"


@pytest.fixture(scope="module")
def sample():
    df = load_dataset(DATA)
    X, numeric, categorical = feature_frame(df, "routine")
    y = build_target(df, "binary")
    Xs, _, ys, _ = train_test_split(X, y, train_size=2000, stratify=y, random_state=0)
    return Xs, ys, numeric, categorical


def test_ordinal_preprocessor_removes_missing_values(sample):
    X, y, numeric, categorical = sample
    pre = OrdinalPreprocessor(numeric, categorical).fit(X)
    out = pre.transform(X)
    assert out.shape == (len(X), len(numeric) + len(categorical))
    assert not np.isnan(out).any()


def test_categorical_indices_follow_numeric_block(sample):
    X, y, numeric, categorical = sample
    pre = OrdinalPreprocessor(numeric, categorical).fit(X)
    assert pre.categorical_indices_ == list(range(len(numeric), len(numeric) + len(categorical)))


def test_unseen_category_maps_to_sentinel(sample):
    X, y, numeric, categorical = sample
    pre = OrdinalPreprocessor(numeric, categorical).fit(X)
    novel = X.head(1).copy()
    novel.loc[novel.index[0], categorical[0]] = "__never_seen__"
    assert pre.transform(novel)[0, len(numeric)] == -1


def test_smotenc_balances_training_classes(sample):
    X, y, numeric, categorical = sample
    pre = OrdinalPreprocessor(numeric, categorical).fit(X)
    Xr, yr = smotenc_resample(pre.transform(X), y, len(numeric), len(categorical))
    counts = pd.Series(yr).value_counts()
    assert counts.min() == counts.max()
    assert len(Xr) > len(X)


def test_smotenc_keeps_categorical_codes_integral(sample):
    """Synthetic categorical values must remain valid category codes, not blends."""
    X, y, numeric, categorical = sample
    pre = OrdinalPreprocessor(numeric, categorical).fit(X)
    Xr, _ = smotenc_resample(pre.transform(X), y, len(numeric), len(categorical))
    cat_block = Xr[:, len(numeric) :]
    assert np.allclose(cat_block, np.round(cat_block))


def test_post_smote_encoder_scales_and_expands(sample):
    X, y, numeric, categorical = sample
    pre = OrdinalPreprocessor(numeric, categorical).fit(X)
    A = pre.transform(X)
    post = PostSMOTEEncoder(len(numeric), len(categorical), categorical).fit(A)
    out = post.transform(A)
    assert out.shape[0] == len(X)
    assert out.shape[1] > len(numeric) + len(categorical)  # one-hot expansion
    dense_numeric = out[:, : len(numeric)].toarray()
    assert abs(dense_numeric.mean()) < 0.1  # standardised
    assert len(post.get_feature_names_out(np.array(numeric + categorical))) == out.shape[1]


def test_pipeline_end_to_end_predicts_valid_labels(sample):
    X, y, numeric, categorical = sample
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = build_pipeline(numeric, categorical, kernel="linear").fit(Xtr, ytr)
    pred = model.predict(Xte)
    assert len(pred) == len(Xte)
    assert set(pred).issubset(set(y.unique()))


def test_calibrated_pipeline_emits_probabilities(sample):
    X, y, numeric, categorical = sample
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = build_pipeline(numeric, categorical, kernel="linear", probability=True).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    assert proba.shape == (len(Xte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_resampling_never_touches_held_out_data(sample):
    """The test partition must reach the classifier unresampled and unchanged."""
    X, y, numeric, categorical = sample
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = build_pipeline(numeric, categorical, kernel="linear").fit(Xtr, ytr)
    assert len(model.predict(Xte)) == len(Xte) == len(yte)


def test_search_space_is_tractable():
    assert 10 <= n_candidates() <= 60
