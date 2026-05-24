"""
Evaluate a saved model and plot ROC curve + confusion matrix.

Usage:
    python src/evaluate.py --model models/rf.pkl \
                           --X data/processed/X.csv \
                           --y data/processed/y.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def plot_roc(y_true, y_prob, save_path="models/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="steelblue", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"ROC curve saved to {save_path}")


def plot_confusion(y_true, y_pred, save_path="models/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Susceptible", "Resistant"],
        yticklabels=["Susceptible", "Resistant"],
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--X", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    X = pd.read_csv(args.X, index_col=0)
    y = pd.read_csv(args.y, index_col=0).squeeze()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Susceptible", "Resistant"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    plot_roc(y_test, y_prob)
    plot_confusion(y_test, y_pred)


if __name__ == "__main__":
    main()
