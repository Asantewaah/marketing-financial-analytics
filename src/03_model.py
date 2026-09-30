"""
03_model.py — Predictive Modelling
Bank Marketing Campaign: Predicting Term Deposit Subscriptions

Trains three classifiers (Logistic Regression, Random Forest, XGBoost),
evaluates with cross-validation and held-out test metrics, and produces
comparison plots. Best model is saved for the business value script.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve, confusion_matrix
)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
FIGS    = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
FIGS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import INK, GOLD, GREY, LIGHT, DARK_GOLD, SUBTLE, SEQ, LOW_HIGH, apply_style, frame, save

apply_style()
MODEL_COLOURS = {"Random Forest": INK, "XGBoost": GOLD, "Logistic Regression": GREY}
ECONOMIC = {"econ_score", "nr.employed", "euribor3m", "emp.var.rate", "high_rates",
            "cons.conf.idx", "stable_economy", "cons.price.idx"}


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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(top=0.78, wspace=0.3)
    order = ["Logistic Regression", "XGBoost", "Random Forest"]   # draw the leaders last
    for ax in axes:
        ax.grid(axis="y", color=LIGHT, lw=1)
    for name in order:
        y_prob, col = metrics[name]["y_prob"], MODEL_COLOURS[name]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        lw = 1.5 if name == "Logistic Regression" else 2
        axes[0].plot(fpr, tpr, color=col, lw=lw)
        axes[1].plot(rec, prec, color=col, lw=lw)
    axes[0].plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    axes[0].text(0.62, 0.55, "random guessing", color=GREY, fontsize=8.5, rotation=33)
    axes[1].axhline(y_test.mean(), color=GREY, lw=1, ls="--")
    axes[1].text(0.02, y_test.mean() - 0.015, f"random guessing ({y_test.mean():.0%})", color=GREY,
                 fontsize=8.5, ha="left", va="top")
    # direct labels, one line per model, in the empty lower-right of the ROC panel
    for i, name in enumerate(["Random Forest", "XGBoost", "Logistic Regression"]):
        m = metrics[name]
        axes[0].text(0.98, 0.26 - 0.09 * i, f"{name}  AUC {m['ROC-AUC']:.3f}", transform=axes[0].transAxes,
                     ha="right", fontsize=9, fontweight="bold",
                     color=DARK_GOLD if name == "XGBoost" else MODEL_COLOURS[name])
        axes[1].text(0.98, 0.95 - 0.09 * i, f"{name}  AP {m['Avg Precision']:.3f}", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=9, fontweight="bold",
                     color=DARK_GOLD if name == "XGBoost" else MODEL_COLOURS[name])
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(xlabel="Recall (share of subscribers found)", ylabel="Precision (share of calls that convert)",
                xlim=(0, 1), ylim=(0, 1.01))
    axes[0].set_title("ROC curve")
    axes[1].set_title("Precision-recall curve")
    frame(fig, "Random Forest and XGBoost tie, and both beat logistic regression",
          f"Held-out test set of {len(y_test):,} clients, using only information available before the call.")
    save(fig, FIGS / "05_roc_pr_curves.png")


def plot_confusion_matrices(metrics: dict, y_test) -> None:
    names = ["Logistic Regression", "Random Forest", "XGBoost"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    fig.subplots_adjust(top=0.68, wspace=0.35)
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_test, metrics[name]["y_pred"])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=SEQ, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            share = v / cm[i].sum()
            ax.text(j, i, f"{v:,}\n{share:.0%} of row", ha="center", va="center", fontsize=9,
                    color="white" if share > 0.55 else INK)
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_yticks([0, 1], ["No", "Yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if name == names[0] else "")
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        recall, precision = cm[1, 1] / cm[1].sum(), cm[1, 1] / cm[:, 1].sum()
        ax.set_title(f"{name}\n\n", fontsize=10.5)
        ax.text(0, 1.03, f"finds {recall:.0%} of subscribers\n{precision:.0%} of flagged clients convert",
                transform=ax.transAxes, fontsize=8.5, color=SUBTLE)
    cm = confusion_matrix(y_test, metrics["XGBoost"]["y_pred"])
    frame(fig, f"At a 0.4 cut-off, XGBoost finds {cm[1, 1] / cm[1].sum():.0%} of subscribers, "
               f"but most flagged clients still say no",
          "Test-set predictions with each model's score cut at 0.4. Shading shows the share of each actual "
          "outcome (row).")
    save(fig, FIGS / "06_confusion_matrices.png")


def plot_feature_importance(fitted: dict, X_train) -> None:
    """Random Forest impurity importance: top 15 features, economic indicators highlighted."""
    rf = fitted["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    econ_share = importances[importances.index.isin(ECONOMIC)].sum() / importances.sum()
    top15 = importances.nlargest(15).sort_values()
    colours = [GOLD if f in ECONOMIC else INK for f in top15.index]

    fig, ax = plt.subplots(figsize=(8, 5.6))
    fig.subplots_adjust(top=0.82, left=0.27)
    ax.barh(top15.index, top15.values, color=colours, height=0.7)
    for i, (f, val) in enumerate(top15.items()):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=8.5,
                color=DARK_GOLD if f in ECONOMIC else INK)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_xlim(0, top15.max() * 1.15)
    ax.text(0.98, 0.3, "\u25a0 economic indicator", transform=ax.transAxes, ha="right",
            color=DARK_GOLD, fontsize=9, fontweight="bold")
    ax.text(0.98, 0.23, "\u25a0 client or campaign", transform=ax.transAxes, ha="right",
            color=INK, fontsize=9, fontweight="bold")
    frame(fig, f"The forest leans mainly on economic indicators ({econ_share:.0%} of its importance)",
          "Top 15 features by random forest importance. The indicators mostly record when a call was "
          "made,\nso part of the model's skill is knowing which period it is in.")
    save(fig, FIGS / "07_feature_importance.png")


def plot_shap(fitted: dict, X_test) -> None:
    """SHAP beeswarm for XGBoost."""
    print("  Computing SHAP values (XGBoost) …")
    xgb = fitted["XGBoost"]
    sample = X_test.sample(2000, random_state=42)  # fast subsample
    shap_vals = shap.TreeExplainer(xgb).shap_values(sample)

    fig = plt.figure(figsize=(9, 6.6))
    shap.summary_plot(shap_vals, sample, plot_type="dot", max_display=15, show=False,
                      plot_size=None, cmap=LOW_HIGH, color_bar_label="Feature value")
    fig.subplots_adjust(top=0.84)
    ax = fig.axes[0]
    ax.grid(axis="x", color=LIGHT, lw=1)
    ax.set_xlabel("SHAP value: push towards subscribing (right) or not (left)", color=INK, fontsize=10)
    ax.tick_params(axis="y", labelsize=10, labelcolor=INK)
    ax.tick_params(axis="x", labelsize=9.5, labelcolor=INK)
    frame(fig, "Economic context and contact history drive XGBoost's predictions",
          "Each dot is one of 2,000 test clients. Marigold = high feature value, blue = low. Low employment "
          "and\ninterest rates push predictions up; many contacts this campaign push them down.")
    save(fig, FIGS / "08_shap_beeswarm.png")


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
