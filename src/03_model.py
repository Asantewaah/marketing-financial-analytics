"""
03_model.py — Predictive Modelling
Bank Marketing Campaign: Predicting Term Deposit Subscriptions

Trains three classifiers (Logistic Regression, Random Forest, XGBoost),
evaluates with cross-validation and held-out test metrics, and produces
comparison plots. Best model is saved for the business value script.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import shap
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay,
    confusion_matrix
)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
FIGS    = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
FIGS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def load_splits():
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_test  = pd.read_csv(DATA / "X_test.csv")
    y_train = pd.read_csv(DATA / "y_train.csv").squeeze()
    y_test  = pd.read_csv(DATA / "y_test.csv").squeeze()
    print(f"Loaded: {X_train.shape[0]:,} train | {X_test.shape[0]:,} test rows")
    return X_train, X_test, y_train, y_test


def define_models() -> dict:
    """Three classifiers spanning interpretability vs performance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced",
            solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=4,   # accounts for class imbalance
            eval_metric="logloss", random_state=42,
            verbosity=0
        ),
    }


def cross_validate_models(models: dict, X_train, y_train) -> pd.DataFrame:
    """5-fold stratified CV; report ROC-AUC and Average Precision."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        print(f"  CV: {name} …", end=" ", flush=True)
        auc  = cross_val_score(model, X_train, y_train, cv=cv,
                               scoring="roc_auc", n_jobs=-1)
        ap   = cross_val_score(model, X_train, y_train, cv=cv,
                               scoring="average_precision", n_jobs=-1)
        rows.append({"Model": name,
                     "ROC-AUC": auc.mean(), "ROC-AUC std": auc.std(),
                     "Avg Precision": ap.mean(), "Avg Precision std": ap.std()})
        print(f"AUC={auc.mean():.3f} ± {auc.std():.3f}")
    return pd.DataFrame(rows).set_index("Model")


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test) -> dict:
    """Fit on full train, evaluate on held-out test."""
    fitted = {}
    metrics = {}
    for name, model in models.items():
        print(f"  Fitting {name} …")
        model.fit(X_train, y_train)
        fitted[name] = model
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.4).astype(int)   # threshold tuned for recall
        metrics[name] = {
            "ROC-AUC":      roc_auc_score(y_test, y_prob),
            "Avg Precision": average_precision_score(y_test, y_prob),
            "y_prob":        y_prob,
            "y_pred":        y_pred,
        }
    return fitted, metrics


def plot_roc_pr_curves(fitted: dict, metrics: dict, X_test, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = ["#2563EB", "#16A34A", "#DC2626"]

    for (name, model), colour in zip(fitted.items(), colours):
        y_prob = metrics[name]["y_prob"]
        RocCurveDisplay.from_predictions(
            y_test, y_prob, ax=axes[0],
            name=name,
            color=colour, alpha=0.85
        )
        PrecisionRecallDisplay.from_predictions(
            y_test, y_prob, ax=axes[1],
            name=name,
            color=colour, alpha=0.85
        )

    axes[0].set_title("ROC Curves — Test Set", fontweight="bold")
    axes[0].plot([0,1],[0,1],"k--",lw=0.8,label="Random")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Precision-Recall Curves — Test Set", fontweight="bold")
    axes[1].legend(fontsize=9)

    for ax in axes:
        sns.despine(ax=ax)

    fig.suptitle("Model Comparison: ROC & Precision-Recall", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "05_roc_pr_curves.png")
    plt.close()
    print("Saved: 05_roc_pr_curves.png")


def plot_confusion_matrices(metrics: dict, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    model_names = list(metrics.keys())
    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y_test, metrics[name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax,
                    xticklabels=["No","Yes"], yticklabels=["No","Yes"],
                    linewidths=0.5, cbar=False)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices (threshold = 0.4)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "06_confusion_matrices.png")
    plt.close()
    print("Saved: 06_confusion_matrices.png")


def plot_feature_importance(fitted: dict, X_train) -> None:
    """Random Forest permutation importance — top 15 features."""
    rf = fitted["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top15 = importances.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top15.index, top15.values,
                   color="#2563EB", alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, top15.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_title("Random Forest: Top 15 Feature Importances", fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Impurity")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(FIGS / "07_feature_importance.png")
    plt.close()
    print("Saved: 07_feature_importance.png")


def plot_shap(fitted: dict, X_test) -> None:
    """SHAP beeswarm for XGBoost — most interpretable summary."""
    print("  Computing SHAP values (XGBoost) …")
    xgb = fitted["XGBoost"]
    sample = X_test.sample(2000, random_state=42)  # fast subsample
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, sample, plot_type="dot",
                      max_display=15, show=False, plot_size=None)
    plt.title("SHAP Feature Impact — XGBoost (n=2,000 sample)",
              fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIGS / "08_shap_beeswarm.png")
    plt.close()
    print("Saved: 08_shap_beeswarm.png")


def time_split_check() -> float:
    """
    Stricter test: the raw data are ordered by date (May 2008 to Nov 2010).
    Train on the earliest 80% of contacts and score the latest 20%, which is
    closer to how the model would be used on a future campaign.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("features", Path(__file__).with_name("02_features.py"))
    feats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feats)
    df = feats.ordinal_encode_categoricals(feats.engineer_features(feats.encode_unknowns(feats.load_raw())))
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y", "duration", "long_call"])
    cut = int(0.8 * len(df))
    model = define_models()["Random Forest"].fit(X.iloc[:cut], y.iloc[:cut])
    auc = roc_auc_score(y.iloc[cut:], model.predict_proba(X.iloc[cut:])[:, 1])
    print(f"  Time-based split (train on earliest 80%, test on latest 20%): ROC-AUC = {auc:.3f}")
    print(f"  Conversion rate: {y.iloc[:cut].mean():.1%} in training period vs {y.iloc[cut:].mean():.1%} in test period")
    return auc


def save_best_model(fitted: dict, metrics: dict) -> None:
    best = max(metrics, key=lambda k: metrics[k]["ROC-AUC"])
    print(f"\n  Best model: {best} (AUC={metrics[best]['ROC-AUC']:.3f})")
    with open(OUTPUTS / "best_model.pkl", "wb") as f:
        pickle.dump({"name": best, "model": fitted[best]}, f)
    print("  Saved: outputs/best_model.pkl")


def print_test_report(metrics: dict, y_test) -> None:
    print("\n=== Test Set Classification Reports ===")
    for name, m in metrics.items():
        print(f"\n--- {name} ---")
        print(classification_report(y_test, m["y_pred"],
                                    target_names=["No", "Yes"]))


if __name__ == "__main__":
    print("=== Modelling Pipeline ===\n")

    X_train, X_test, y_train, y_test = load_splits()
    models = define_models()

    print("\n[1] Cross-Validation")
    cv_results = cross_validate_models(models, X_train, y_train)
    print("\nCV Summary:")
    print(cv_results.round(3).to_string())
    cv_results.to_csv(OUTPUTS / "cv_results.csv")

    print("\n[2] Test Set Evaluation")
    fitted, metrics = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print("\n[3] Plots")
    plot_roc_pr_curves(fitted, metrics, X_test, y_test)
    plot_confusion_matrices(metrics, y_test)
    plot_feature_importance(fitted, X_train)
    plot_shap(fitted, X_test)

    pd.DataFrame({n: {"ROC-AUC": m["ROC-AUC"], "Avg Precision": m["Avg Precision"]}
                  for n, m in metrics.items()}).T.to_csv(OUTPUTS / "test_metrics.csv")
    print_test_report(metrics, y_test)
    save_best_model(fitted, metrics)

    print("\n[4] Robustness: time-based split")
    pd.Series({"time_split_roc_auc": time_split_check()}).to_csv(OUTPUTS / "time_split_check.csv")

    print("\n✓ Modelling complete.")
