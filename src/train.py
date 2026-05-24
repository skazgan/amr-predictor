"""
Train and save an AMR classifier.

Usage:
    python src/train.py --X data/processed/X.csv \
                        --y data/processed/y.csv \
                        --model models/rf.pkl
"""

import argparse
import pickle
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_pipeline(model: str = "rf", n_features: int = 512) -> Pipeline:
    """
    Build a pipeline with:
      1. Feature selection  – keep the top n_features k-mers by mutual information
      2. Scaling            – standardise counts
      3. Classifier         – RF or XGBoost
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)

    if model == "xgb":
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
    else:
        base_clf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    # Wrap with isotonic calibration so predicted probabilities are reliable.
    # cv=3 uses internal cross-fitting — the calibrator learns on held-out folds
    # so the model can't overfit its own predictions.
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
