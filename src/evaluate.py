"""
Model Evaluation & Comparison
===============================
Evaluates all trained models, generates metrics, confusion matrices,
ROC curves, and a comparison summary.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n📊 {model_name}")
    print(f"   Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"   Precision: {metrics['Precision']:.4f}")
    print(f"   Recall:    {metrics['Recall']:.4f}")
    print(f"   F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"   ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    return metrics


def plot_confusion_matrices(models: dict, X_test, y_test):
    """Generate confusion matrix plots for all models."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["On-Time", "Delayed"],
            yticklabels=["On-Time", "Delayed"],
        )
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices – Shipment Delay Prediction", fontsize=15, fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📈 Confusion matrices saved: {filepath}")


def plot_roc_curves(models: dict, X_test, y_test):
    """Generate ROC curve comparison plot."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    colors = ["#3b82f6", "#22c55e", "#f97316"]

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC={auc_score:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves – Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(OUTPUTS_DIR, "roc_curves.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 ROC curves saved: {filepath}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    """Plot feature importance for tree-based models."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15 features

        plt.figure(figsize=(10, 7))
        plt.barh(
            range(len(indices)),
            importances[indices],
            color="#6366f1",
            edgecolor="#4338ca",
        )
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
        plt.xlabel("Feature Importance", fontsize=12)
        plt.title(f"Top 15 Features – {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")
        filepath = os.path.join(OUTPUTS_DIR, f"feature_importance_{safe_name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Feature importance saved: {filepath}")


def evaluate_all_models(models: dict, X_test, y_test, feature_names: list = None) -> pd.DataFrame:
    """
    Evaluate all models, generate plots, and return comparison table.
    """
    print("=" * 60)
    print("📊 MODEL EVALUATION & COMPARISON")
    print("=" * 60)

    all_metrics = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

    # Comparison table
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.set_index("Model")

    print("\n" + "=" * 60)
    print("📋 COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string())

    # Identify best model
    best_model_name = comparison_df["ROC-AUC"].idxmax()
    best_auc = comparison_df.loc[best_model_name, "ROC-AUC"]
    print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_auc:.4f})")

    # Generate plots
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)

    # Feature importance for tree-based models
    if feature_names:
        for name, model in models.items():
            plot_feature_importance(model, feature_names, name)

    # Save metrics to JSON
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n💾 Metrics saved: {metrics_path}")

    # Save comparison CSV
    comparison_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path)
    print(f"💾 Comparison saved: {comparison_path}")

    return comparison_df


if __name__ == "__main__":
    from preprocessing import load_data, prepare_data

    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # Load saved models
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
        "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib")),
        "XGBoost": joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib")),
    }

    evaluate_all_models(models, X_test, y_test, feature_names)
