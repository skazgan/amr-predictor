"""
Train and save an AMR classifier.

Usage:
    python src/train.py --X data/processed/X.csv \
                        --y data/processed/y.csv \
                        --model models/rf.pkl
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted soft-voting ensemble of XGBoost and Random Forest.

    Instead of hard majority vote, we average the predicted probabilities —
    XGBoost gets 60% weight (it consistently outperforms RF on this dataset),
    RF gets 40% weight. This reduces variance while keeping XGBoost's signal.
    """

    def __init__(self, xgb_weight: float = 0.6, n_estimators: int = 300):
        self.xgb_weight  = xgb_weight
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.xgb_ = CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=self.n_estimators, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric="logloss", verbosity=0,
            ),
            method="isotonic", cv=3,
        )
        self.rf_ = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=self.n_estimators, max_features="sqrt",
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        )
        self.xgb_.fit(X, y)
        self.rf_.fit(X, y)
        return self

    def predict_proba(self, X):
        p_xgb = self.xgb_.predict_proba(X)
        p_rf  = self.rf_.predict_proba(X)
        rf_w  = 1.0 - self.xgb_weight
        return self.xgb_weight * p_xgb + rf_w * p_rf

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def feature_importances_(self):
        """Average feature importances from both base estimators."""
        def get_fi(cal_clf):
            fis = [est.estimator.feature_importances_
                   for est in cal_clf.calibrated_classifiers_]
            return np.mean(fis, axis=0)
        fi_xgb = get_fi(self.xgb_)
        fi_rf  = get_fi(self.rf_)
        return self.xgb_weight * fi_xgb + (1 - self.xgb_weight) * fi_rf


def build_pipeline(model: str = "xgb", n_features: int = 512) -> Pipeline:
    """
    Build a pipeline with:
      1. Feature selection  – keep the top n_features k-mers by mutual information
      2. Scaling            – standardise counts
      3. Classifier         – RF, XGBoost, or ensemble (soft-voting XGB+RF)
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)

    if model == "ensemble":
        clf = SoftVotingEnsemble(xgb_weight=0.6)
    elif model == "xgb":
        base_clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    else:
        base_clf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)

    return Pipeline([
        ("select", selector),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def tune_n_features(X: pd.DataFrame, y: pd.Series, model: str = "xgb") -> int:
    """Grid-search over number of features; return best value."""
    print(f"  Tuning number of features for [{model.upper()}] ...")
    candidates = [64, 128, 256, 512, 1024]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_k, best_auc = candidates[0], 0.0
    for k in candidates:
        pipe = build_pipeline(model, n_features=k)
        scores = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc")
        auc = scores["test_score"].mean()
        print(f"    k={k:5d}  →  AUC {auc:.3f}")
        if auc > best_auc:
            best_auc, best_k = auc, k
    print(f"  Best: k={best_k} (AUC {best_auc:.3f})")
    return best_k


def train(X: pd.DataFrame, y: pd.Series, model: str = "rf",
          n_features: int | None = None) -> tuple:
    if n_features is None:
        n_features = tune_n_features(X, y, model)

    pipe = build_pipeline(model, n_features=n_features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        return_train_score=False,
    )
    print(f"\n--- Cross-validation results (5-fold) [{model.upper()}, k={n_features}] ---")
    for metric, vals in scores.items():
        if metric.startswith("test_"):
            name = metric.replace("test_", "")
            print(f"  {name:12s}: {vals.mean():.3f} ± {vals.std():.3f}")

    pipe.fit(X, y)
    return pipe, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--model", default="models/rf.pkl")
    args = parser.parse_args()

    X = pd.read_csv(args.X, index_col=0)
    y = pd.read_csv(args.y, index_col=0).squeeze()

    print(f"Loaded {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution:\n{y.value_counts().to_string()}")

    model, _ = train(X, y)

    with open(args.model, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {args.model}")


if __name__ == "__main__":
    main()
