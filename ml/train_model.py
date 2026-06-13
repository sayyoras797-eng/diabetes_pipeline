"""
ml/train_model.py
-----------------
Trains a diabetes risk prediction model using data from PostgreSQL.
Tracks every experiment with MLflow and saves the best model artifact to S3/MinIO.

Usage:
    python train_model.py
    python train_model.py --experiment "v2-feature-test"
"""

import os
import json
import argparse
import warnings
import psycopg2
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "database": os.getenv("POSTGRES_DB", "diabetes_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "password"),
}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "/app/models")

FEATURE_COLS = [
    "age", "bmi", "glucose", "blood_pressure",
    "hba1c", "insulin", "skin_thickness", "pregnancies"
]
TARGET_COL = "label"


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_data_from_postgres() -> pd.DataFrame:
    print("📦 Loading data from PostgreSQL...")
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    query = f"""
        SELECT {', '.join(FEATURE_COLS)}, gender, {TARGET_COL}
        FROM patient_records
        ORDER BY recorded_at DESC
        LIMIT 5000
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"   ✅ Loaded {len(df)} records")
    return df


def preprocess(df: pd.DataFrame):
    """Encode gender, split features / target."""
    le = LabelEncoder()
    df["gender_enc"] = le.fit_transform(df["gender"])

    feature_cols = FEATURE_COLS + ["gender_enc"]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Fill any NaN with column median
    X = X.fillna(X.median())
    return X, y, feature_cols


# ──────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────
def get_models():
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    }


# ──────────────────────────────────────────────
# Plots (logged as MLflow artifacts)
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    path = f"/tmp/cm_{model_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names: list, model_name: str) -> str | None:
    """Works for tree-based models only."""
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df_imp["feature"], df_imp["importance"], color="#4C72B0")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    path = f"/tmp/fi_{model_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_comparison(results: dict, y_test, X_test) -> str:
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, info in results.items():
        y_prob = info["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    path = "/tmp/roc_comparison.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_and_log(experiment_name: str = "diabetes-risk-prediction"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    df = load_data_from_postgres()
    X, y, feature_cols = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Dataset split: train={len(X_train)}, test={len(X_test)}")
    print(f"   Class balance (train): {dict(y_train.value_counts())}\n")

    models = get_models()
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, pipeline in models.items():
        print(f"🔄 Training: {model_name} ...")
        with mlflow.start_run(run_name=model_name):
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train,
                                        cv=cv, scoring="roc_auc", n_jobs=-1)

            # Fit on full train set
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "cv_roc_auc_mean": cv_scores.mean(),
                "cv_roc_auc_std": cv_scores.std(),
            }

            # Log params
            clf = pipeline.named_steps["clf"]
            mlflow.log_params({
                "model_type": model_name,
                "train_size": len(X_train),
                "test_size": len(X_test),
                **{k: v for k, v in clf.get_params().items()
                   if not isinstance(v, object.__class__)},
            })

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log plots as artifacts
            cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
            mlflow.log_artifact(cm_path, "plots")

            fi_path = plot_feature_importance(pipeline, feature_cols, model_name)
            if fi_path:
                mlflow.log_artifact(fi_path, "plots")

            # Log model
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"diabetes_{model_name.lower()}",
            )

            results[model_name] = {"model": pipeline, "metrics": metrics}

            print(f"   Accuracy={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1_score']:.4f}  "
                  f"ROC-AUC={metrics['roc_auc']:.4f}  "
                  f"CV-AUC={metrics['cv_roc_auc_mean']:.4f}±{metrics['cv_roc_auc_std']:.4f}")

    # ── ROC comparison plot (logged under a separate parent run) ──
    with mlflow.start_run(run_name="model_comparison"):
        roc_path = plot_roc_comparison(results, y_test, X_test)
        mlflow.log_artifact(roc_path, "comparison")

    # ── Pick & save best model by accuracy ──
    best_name = max(results, key=lambda k: results[k]["metrics"]["accuracy"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)

    meta_path = os.path.join(MODEL_OUTPUT_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_name": best_name,
            "feature_cols": feature_cols,
            "metrics": best_metrics,
        }, f, indent=2)

    print(f"\n🏆 Best model: {best_name}")
    print(f"   ROC-AUC : {best_metrics['roc_auc']:.4f}")
    print(f"   F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"\n💾 Saved to: {model_path}")
    print(f"💾 Metadata saved to: {meta_path}")

    # Summary table
    print("\n📋 Model Comparison Summary:")
    print(f"{'Model':<22} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>9}")
    print("─" * 52)
    for name, info in results.items():
        m = info["metrics"]
        marker = " ← best" if name == best_name else ""
        print(f"{name:<22} {m['accuracy']:>9.4f} {m['f1_score']:>8.4f} "
              f"{m['roc_auc']:>9.4f}{marker}")

    return best_model, best_name, results


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="diabetes-risk-prediction",
                        help="MLflow experiment name")
    args = parser.parse_args()
    train_and_log(experiment_name=args.experiment)
