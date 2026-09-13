"""Gradient-boosted-tree baselines on the Rule Set 2 feature set.

``azimuth_gbt`` is a faithful reproduction of the Rule Set 2 model: sklearn's
``GradientBoostingRegressor`` with exactly the hyperparameters Azimuth hardcodes in
``models/ensembles.adaboost_on_fold`` + ``model_comparison.adaboost_setup``
(squared-error loss, lr 0.1, 100 trees, depth 3, no subsampling, no feature
subsampling). ``lightgbm_gbt`` is a modern GBT on the same features, included
because the spec asks for XGBoost/LightGBM and because it shows how much of any
gap to the CNN is "better trees" rather than "better representation".
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Azimuth's hardcoded Rule Set 2 settings. 'ls' was renamed 'squared_error' in
# modern scikit-learn; it is the same loss.
AZIMUTH_GBT_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.1,
    n_estimators=100,
    alpha=0.5,
    subsample=1.0,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    init=None,
    max_features=None,
    max_leaf_nodes=None,
    warm_start=False,
)


def azimuth_gbt(seed: int = 0) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=seed, **AZIMUTH_GBT_PARAMS)


def lightgbm_gbt(seed: int = 0):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


def fit_predict(model, X_train, y_train, X_test) -> np.ndarray:
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=np.float64)
