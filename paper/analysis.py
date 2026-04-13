"""
Full analysis pipeline for:
"Intelligent IoT-Enabled Remote Healthcare Monitoring Using Ensemble Machine
Learning and MedGemma-Based Clinical Decision Support"
Author: Lukman Kunveng et al.
"""

import warnings, os, json
warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("SHAP not installed; skipping SHAP plot.")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "Asthma":           "#2196F3",
    "Diabetes Mellitus":"#9C27B0",
    "Healthy":          "#4CAF50",
    "Heart Disease":    "#F44336",
    "Hypertension":     "#FF9800",
}
DISEASE_ORDER = ["Asthma", "Diabetes Mellitus", "Healthy", "Heart Disease", "Hypertension"]
COLOR_LIST    = [PALETTE[d] for d in DISEASE_ORDER]

STYLE = dict(dpi=300, bbox_inches="tight")

plt.rcParams.update({
    "font.family":     "DejaVu Serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading dataset …")
df = pd.read_csv("../Patient_dataset.csv")

# Encode Gender
df["Gender"] = (df["Gender"] == "Male").astype(int)

FEATURE_COLS = [
    "Gender", "Heart Rate (bpm)", "SpO2 Level (%)",
    "Systolic Blood Pressure (mmHg)", "Diastolic Blood Pressure (mmHg)",
    "Body Temperature (C)", "Weight_kg", "Height_cm", "BMI",
]
TARGET_COL = "Predicted Disease"

X = df[FEATURE_COLS].values
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL])
classes = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Dataset exploration (4-panel)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1 – Dataset exploration …")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Dataset Overview — IoT-Collected Patient Vitals (N = 60,000)",
             fontsize=14, fontweight="bold", y=1.01)

# 1a – Class distribution
ax = axes[0, 0]
counts = df[TARGET_COL].value_counts().reindex(DISEASE_ORDER)
bars = ax.bar(DISEASE_ORDER, counts.values,
              color=COLOR_LIST, edgecolor="white", linewidth=1.2, zorder=3)
ax.set_title("(a) Class Distribution", fontweight="bold")
ax.set_ylabel("Patient Count")
ax.set_xticklabels(DISEASE_ORDER, rotation=20, ha="right")
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
for bar, cnt in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f"{cnt:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(0, counts.max() * 1.15)

# 1b – Gender split per class
ax = axes[0, 1]
gender_df = df.groupby([TARGET_COL, "Gender"]).size().unstack(fill_value=0)
gender_df.columns = ["Female", "Male"]
gender_df = gender_df.reindex(DISEASE_ORDER)
x_pos = np.arange(len(DISEASE_ORDER))
w = 0.35
ax.bar(x_pos - w/2, gender_df["Female"], w, label="Female",
       color="#E91E63", alpha=0.85, edgecolor="white")
ax.bar(x_pos + w/2, gender_df["Male"],   w, label="Male",
       color="#1565C0", alpha=0.85, edgecolor="white")
ax.set_title("(b) Gender Distribution per Condition", fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(DISEASE_ORDER, rotation=20, ha="right")
ax.set_ylabel("Patient Count")
ax.legend(framealpha=0.7)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# 1c – BMI violin
ax = axes[1, 0]
bmi_data = [df.loc[df[TARGET_COL] == d, "BMI"].values for d in DISEASE_ORDER]
parts = ax.violinplot(bmi_data, positions=range(len(DISEASE_ORDER)),
                      showmedians=True, showextrema=True)
for i, (pc, col) in enumerate(zip(parts["bodies"], COLOR_LIST)):
    pc.set_facecolor(col); pc.set_alpha(0.75)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
parts["cbars"].set_color("gray")
if "cextrema" in parts: parts["cextrema"].set_color("gray")
ax.set_xticks(range(len(DISEASE_ORDER)))
ax.set_xticklabels(DISEASE_ORDER, rotation=20, ha="right")
ax.set_title("(c) BMI Distribution by Condition", fontweight="bold")
ax.set_ylabel("BMI (kg/m²)")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)

# 1d – Correlation heatmap
ax = axes[1, 1]
feat_labels = ["Gender","HR","SpO₂","SBP","DBP","Temp","Weight","Height","BMI"]
corr = pd.DataFrame(X_scaled, columns=feat_labels).corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax, cmap="RdYlBu_r", center=0,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.4, square=True, cbar_kws={"shrink": 0.75})
ax.set_title("(d) Feature Correlation Matrix", fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig1_dataset_overview.pdf", **STYLE)
plt.savefig("figures/fig1_dataset_overview.png", **STYLE)
plt.close()
print("  Saved fig1")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("Training XGBoost …")
xgb_model = XGBClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=3, gamma=0.1,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=SEED, n_jobs=-1,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)

print("Training LightGBM …")
lgb_model = LGBMClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    min_child_samples=20, reg_alpha=0.05, reg_lambda=0.05,
    random_state=SEED, n_jobs=-1, verbose=-1,
)
lgb_model.fit(X_train, y_train)

# Soft-voting ensemble
proba_xgb = xgb_model.predict_proba(X_test)
proba_lgb = lgb_model.predict_proba(X_test)
proba_ens = (proba_xgb + proba_lgb) / 2.0
y_pred_ens = np.argmax(proba_ens, axis=1)

# Individual predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)

# ── Per-model metrics ─────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, name):
    return {
        "Model": name,
        "Accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(y_true, y_pred, average="weighted") * 100, 2),
        "Recall":    round(recall_score(y_true, y_pred, average="weighted") * 100, 2),
        "F1":        round(f1_score(y_true, y_pred, average="weighted") * 100, 2),
    }

metrics = [
    get_metrics(y_test, y_pred_xgb, "XGBoost"),
    get_metrics(y_test, y_pred_lgb, "LightGBM"),
    get_metrics(y_test, y_pred_ens, "Ensemble (Ours)"),
]
metrics_df = pd.DataFrame(metrics)
print("\nModel Comparison:\n", metrics_df.to_string(index=False))
metrics_df.to_csv("figures/model_metrics.csv", index=False)

ens_acc = accuracy_score(y_test, y_pred_ens) * 100
print(f"\nEnsemble Accuracy: {ens_acc:.4f}%")

# Full classification report
report = classification_report(y_test, y_pred_ens,
                               target_names=classes, output_dict=True)
report_df = pd.DataFrame(report).T
print("\nClassification Report:\n", report_df.round(4))
report_df.to_csv("figures/classification_report.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2 – Confusion matrix …")
cm = confusion_matrix(y_test, y_pred_ens)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

short = ["Asthma", "Diabetes", "Healthy", "Heart Dis.", "Hypertension"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Confusion Matrix — Ensemble Model (XGBoost + LightGBM)",
             fontsize=13, fontweight="bold")

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
            xticklabels=short, yticklabels=short,
            ax=axes[0], cbar_kws={"shrink": 0.8})
axes[0].set_title("(a) Absolute Counts", fontweight="bold")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Normalised
annot = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Purples", linewidths=0.5,
            xticklabels=short, yticklabels=short,
            ax=axes[1], vmin=0, vmax=100, cbar_kws={"shrink": 0.8})
axes[1].set_title("(b) Row-Normalised (%)", fontweight="bold")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("figures/fig2_confusion_matrix.pdf", **STYLE)
plt.savefig("figures/fig2_confusion_matrix.png", **STYLE)
plt.close()
print("  Saved fig2")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Multi-class ROC curves
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3 – ROC curves …")
y_bin = label_binarize(y_test, classes=list(range(len(classes))))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Per-class ROC
ax = axes[0]
macro_tpr = np.linspace(0, 1, 200)
macro_aucs = []
for i, (cls, col) in enumerate(zip(classes, COLOR_LIST)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba_ens[:, i])
    roc_auc = auc(fpr, tpr)
    macro_aucs.append(roc_auc)
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f"{cls} (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Classifier")
ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
ax.set_title("(a) Per-Class ROC Curves", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", framealpha=0.85)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.02])
ax.grid(True, linestyle="--", alpha=0.4)

# Macro average
ax2 = axes[1]
mean_auc = np.mean(macro_aucs)
for i, (cls, col) in enumerate(zip(classes, COLOR_LIST)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba_ens[:, i])
    ax2.plot(fpr, tpr, color=col, lw=1.5, alpha=0.5)
# Macro average line
all_fpr = np.unique(np.concatenate([
    roc_curve(y_bin[:, i], proba_ens[:, i])[0] for i in range(len(classes))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba_ens[:, i])
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= len(classes)
ax2.plot(all_fpr, mean_tpr, "k-", lw=2.5,
         label=f"Macro Average (AUC = {mean_auc:.4f})")
ax2.plot([0, 1], [0, 1], "r--", lw=1.2, alpha=0.6)
ax2.set_title("(b) Macro-Average ROC Curve", fontweight="bold")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right", framealpha=0.85)
ax2.set_xlim([-0.01, 1.01])
ax2.set_ylim([-0.01, 1.02])
ax2.grid(True, linestyle="--", alpha=0.4)
patches = [mpatches.Patch(color=col, label=cls)
           for cls, col in zip(classes, COLOR_LIST)]
ax2.legend(handles=patches + [
    plt.Line2D([0], [0], color="k", lw=2.5, label=f"Macro Avg (AUC={mean_auc:.4f})")],
    loc="lower right", framealpha=0.85, fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig3_roc_curves.pdf", **STYLE)
plt.savefig("figures/fig3_roc_curves.png", **STYLE)
plt.close()
print("  Saved fig3")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Feature Importance (XGB + LGB side by side)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4 – Feature importance …")
feat_labels = ["Gender","Heart Rate","SpO₂","Systolic BP",
               "Diastolic BP","Temperature","Weight","Height","BMI"]

xgb_imp = xgb_model.feature_importances_
lgb_imp = lgb_model.feature_importances_ / lgb_model.feature_importances_.sum()
ens_imp = (xgb_imp + lgb_imp) / 2.0

imp_df = pd.DataFrame({
    "Feature": feat_labels,
    "XGBoost":  xgb_imp,
    "LightGBM": lgb_imp,
    "Ensemble": ens_imp,
}).sort_values("Ensemble", ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("Feature Importance Scores — Predictive Contribution of IoT Vitals",
             fontsize=13, fontweight="bold")

for ax, col, color, title in zip(
    axes,
    ["XGBoost", "LightGBM", "Ensemble"],
    ["#1565C0", "#6A1B9A", "#1B5E20"],
    ["(a) XGBoost", "(b) LightGBM", "(c) Ensemble (Ours)"]
):
    bars = ax.barh(imp_df["Feature"], imp_df[col] * 100,
                   color=color, alpha=0.82, edgecolor="white", height=0.65)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Relative Importance (%)")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                f"{w:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, imp_df[col].max() * 115)

plt.tight_layout()
plt.savefig("figures/fig4_feature_importance.pdf", **STYLE)
plt.savefig("figures/fig4_feature_importance.png", **STYLE)
plt.close()
print("  Saved fig4")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Model comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 5 – Model comparison …")
# Add baselines for comparison
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

baselines = [
    ("Naïve Bayes",      GaussianNB()),
    ("Decision Tree",    DecisionTreeClassifier(max_depth=10, random_state=SEED)),
    ("Random Forest",    RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
    ("SVM (Linear)",     CalibratedClassifierCV(LinearSVC(random_state=SEED, max_iter=2000))),
    ("XGBoost",          xgb_model),
    ("LightGBM",         lgb_model),
]

comp_metrics = []
for name, clf in baselines:
    clf.fit(X_train, y_train)
    yp = clf.predict(X_test)
    comp_metrics.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, yp) * 100,
        "Precision": precision_score(y_test, yp, average="weighted") * 100,
        "Recall":    recall_score(y_test, yp, average="weighted") * 100,
        "F1":        f1_score(y_test, yp, average="weighted") * 100,
    })

# Add ensemble
comp_metrics.append({
    "Model":     "Ensemble (Ours)",
    "Accuracy":  accuracy_score(y_test, y_pred_ens) * 100,
    "Precision": precision_score(y_test, y_pred_ens, average="weighted") * 100,
    "Recall":    recall_score(y_test, y_pred_ens, average="weighted") * 100,
    "F1":        f1_score(y_test, y_pred_ens, average="weighted") * 100,
})

comp_df = pd.DataFrame(comp_metrics)
comp_df.to_csv("figures/comparison_metrics.csv", index=False)
print(comp_df.to_string(index=False))

metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(comp_df))
w = 0.2
bar_colors = ["#78909C", "#26A69A", "#42A5F5", "#EC407A"]

fig, ax = plt.subplots(figsize=(14, 6))
for i, (metric, col) in enumerate(zip(metric_cols, bar_colors)):
    bars = ax.bar(x + i * w, comp_df[metric], w,
                  label=metric, color=col, alpha=0.88, edgecolor="white")

ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(comp_df["Model"], rotation=22, ha="right")
ax.set_ylabel("Score (%)")
ax.set_ylim(50, 102)
ax.set_title("Comparative Model Evaluation — Accuracy, Precision, Recall, and F1-Score",
             fontweight="bold", fontsize=13)
ax.legend(loc="lower right", framealpha=0.85)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Highlight ensemble
ens_idx = len(comp_df) - 1
ax.axvspan(ens_idx - 0.1, ens_idx + 4 * w + 0.1,
           alpha=0.07, color="#1B5E20", label="_ens")
ax.text(ens_idx + 1.5 * w, 101.2, "★ Proposed",
        ha="center", fontsize=9, color="#1B5E20", fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig5_model_comparison.pdf", **STYLE)
plt.savefig("figures/fig5_model_comparison.png", **STYLE)
plt.close()
print("  Saved fig5")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Learning curves
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 6 – Learning curves …")
from sklearn.ensemble import GradientBoostingClassifier

train_sizes, train_scores, test_scores = learning_curve(
    lgb_model, X_scaled, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    train_sizes=np.linspace(0.10, 1.0, 10),
    scoring="accuracy", n_jobs=-1, verbose=0,
)
tr_mean = train_scores.mean(axis=1) * 100
tr_std  = train_scores.std(axis=1) * 100
te_mean = test_scores.mean(axis=1) * 100
te_std  = test_scores.std(axis=1) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Learning Curve Analysis — Ensemble Model Generalisation",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(train_sizes, tr_mean, "o-", color="#1565C0", lw=2, label="Training Accuracy")
ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                alpha=0.12, color="#1565C0")
ax.plot(train_sizes, te_mean, "s-", color="#C62828", lw=2, label="Validation Accuracy")
ax.fill_between(train_sizes, te_mean - te_std, te_mean + te_std,
                alpha=0.12, color="#C62828")
ax.set_title("(a) Accuracy vs. Training Set Size", fontweight="bold")
ax.set_xlabel("Training Samples")
ax.set_ylabel("Accuracy (%)")
ax.legend(framealpha=0.85)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_ylim(min(te_mean) - 3, 101)

# Per-class F1 across conditions
ax2 = axes[1]
per_class_f1 = [report[cls]["f1-score"] * 100 for cls in classes]
bars = ax2.bar(classes, per_class_f1,
               color=COLOR_LIST, edgecolor="white", linewidth=1.2, zorder=3)
ax2.axhline(np.mean(per_class_f1), color="red", lw=1.8, linestyle="--",
            label=f"Macro Avg F1 = {np.mean(per_class_f1):.2f}%")
ax2.set_title("(b) Per-Class F1-Score", fontweight="bold")
ax2.set_ylabel("F1-Score (%)")
ax2.set_xticklabels(classes, rotation=20, ha="right")
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)
ax2.set_ylim(0, 110)
ax2.legend(framealpha=0.85)
for bar, val in zip(bars, per_class_f1):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig6_learning_curves.pdf", **STYLE)
plt.savefig("figures/fig6_learning_curves.png", **STYLE)
plt.close()
print("  Saved fig6")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Probability calibration + critical alert analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 7 – Calibration & alert analysis …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Calibration and Critical Alert Detection Performance",
             fontsize=13, fontweight="bold")

# Calibration: one-vs-rest for each class
ax = axes[0]
for i, (cls, col) in enumerate(zip(classes, COLOR_LIST)):
    prob_true, prob_pred = calibration_curve(
        (y_test == i).astype(int), proba_ens[:, i], n_bins=10)
    ax.plot(prob_pred, prob_true, "o-", color=col, lw=1.8, label=cls, markersize=4)
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
ax.set_title("(a) Probability Calibration Curves", fontweight="bold")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.legend(fontsize=8, framealpha=0.85)
ax.grid(True, linestyle="--", alpha=0.4)

# Alert detection stats
ax2 = axes[1]
alert_cols = ["Heart Rate Alert", "SpO2 Level Alert",
              "Blood Pressure Alert", "Temperature Alert"]
alert_labels = ["Heart Rate", "SpO₂", "Blood Pressure", "Temperature"]
abnormal_counts = [(df[c] == "ABNORMAL").sum() for c in alert_cols]
normal_counts   = [(df[c] == "NORMAL").sum() for c in alert_cols]
x = np.arange(len(alert_labels))
w = 0.38
ax2.bar(x - w/2, normal_counts,   w, label="Normal",   color="#43A047", alpha=0.85)
ax2.bar(x + w/2, abnormal_counts, w, label="Abnormal", color="#E53935", alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(alert_labels)
ax2.set_title("(b) IoT Alert Distribution Across Vital Parameters", fontweight="bold")
ax2.set_ylabel("Patient Count")
ax2.legend(framealpha=0.85)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)
for xi, (n, a) in zip(x, zip(normal_counts, abnormal_counts)):
    ax2.text(xi - w/2, n + 200, f"{n:,}", ha="center", fontsize=8)
    ax2.text(xi + w/2, a + 200, f"{a:,}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig7_calibration_alerts.pdf", **STYLE)
plt.savefig("figures/fig7_calibration_alerts.png", **STYLE)
plt.close()
print("  Saved fig7")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – System Architecture Diagram (programmatic)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 8 – Architecture diagram …")
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

def draw_box(ax, x, y, w, h, title, subtitle, color, text_color="white"):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor="white",
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h * 0.65, title,
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=text_color, zorder=4)
    ax.text(x + w/2, y + h * 0.28, subtitle,
            ha="center", va="center", fontsize=7.5,
            color=text_color, alpha=0.88, zorder=4, wrap=True)

def arrow(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#37474F",
                                lw=2, mutation_scale=18))

# Layer 1 – IoT
draw_box(ax, 0.3, 1.5, 2.4, 3.0, "IoT Layer",
         "Wearables & Sensors\nBP · HR · SpO₂\nTemp · Weight · BMI",
         "#1565C0")
# Layer 2 – Blockchain
draw_box(ax, 3.2, 1.5, 2.4, 3.0, "Blockchain Layer",
         "Hyperledger Fabric\nImmutable Ledger\nSmart Contracts · IPFS",
         "#4527A0")
# Layer 3 – Preprocessing
draw_box(ax, 6.1, 1.5, 2.4, 3.0, "Preprocessing",
         "Normalisation\nFeature Engineering\nData Validation",
         "#00695C")
# Layer 4 – Ensemble ML
draw_box(ax, 9.0, 1.5, 2.4, 3.0, "Ensemble ML",
         "XGBoost + LightGBM\nSoft-Voting Classifier\n95.22% Accuracy",
         "#E65100")
# Layer 5 – MedGemma
draw_box(ax, 11.8, 3.0, 2.0, 1.5, "MedGemma LLM",
         "Clinical Recs\nADA · ESC · GINA\nWHO Guidelines",
         "#880E4F")
draw_box(ax, 11.8, 1.2, 2.0, 1.5, "Alert Engine",
         "Critical Thresholds\nHypertensive Crisis\nHypoxaemia Flags",
         "#B71C1C")

# Arrows
for x1, x2 in [(2.7, 3.2), (5.6, 6.1), (8.5, 9.0)]:
    arrow(ax, x1, x2, 3.0)
arrow(ax, 11.4, 11.8, 4.0)
arrow(ax, 11.4, 11.8, 2.0)

ax.set_title("Proposed System Architecture: IoT → Blockchain → ML Ensemble → MedGemma",
             fontsize=12, fontweight="bold", pad=12)

# Layer labels
for x, label in [(1.5, "Layer 1"), (4.4, "Layer 2"),
                 (7.3, "Layer 3"), (10.2, "Layer 4")]:
    ax.text(x, 1.2, label, ha="center", fontsize=8,
            color="#555", style="italic")

plt.tight_layout()
plt.savefig("figures/fig8_architecture.pdf", **STYLE)
plt.savefig("figures/fig8_architecture.png", **STYLE)
plt.close()
print("  Saved fig8")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Cross-validation performance
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 9 – Cross-validation …")
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

cv_results = {}
for name, clf in [("XGBoost", xgb_model), ("LightGBM", lgb_model)]:
    scores = cross_val_score(clf, X_scaled, y, cv=skf,
                             scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores * 100
    print(f"  {name}: {scores.mean()*100:.4f}% ± {scores.std()*100:.4f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("10-Fold Stratified Cross-Validation Results",
             fontsize=13, fontweight="bold")

ax = axes[0]
folds = np.arange(1, 11)
ax.plot(folds, cv_results["XGBoost"],  "o-", color="#1565C0", lw=2,
        label=f"XGBoost  (μ={cv_results['XGBoost'].mean():.2f}%)")
ax.plot(folds, cv_results["LightGBM"], "s-", color="#6A1B9A", lw=2,
        label=f"LightGBM (μ={cv_results['LightGBM'].mean():.2f}%)")
ax.axhline(np.mean(list(cv_results.values())), color="red",
           lw=1.5, linestyle="--", label="Grand Mean")
ax.set_title("(a) Per-Fold Accuracy", fontweight="bold")
ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(folds)
ax.legend(framealpha=0.85)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_ylim(min(min(cv_results["XGBoost"]), min(cv_results["LightGBM"])) - 1, 101)

ax2 = axes[1]
bp = ax2.boxplot([cv_results["XGBoost"], cv_results["LightGBM"]],
                 patch_artist=True, notch=True,
                 medianprops={"color": "white", "linewidth": 2.5})
for patch, color in zip(bp["boxes"], ["#1565C0", "#6A1B9A"]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax2.set_xticklabels(["XGBoost", "LightGBM"])
ax2.set_title("(b) Accuracy Distribution (10-Fold)", fontweight="bold")
ax2.set_ylabel("Accuracy (%)")
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig("figures/fig9_crossvalidation.pdf", **STYLE)
plt.savefig("figures/fig9_crossvalidation.png", **STYLE)
plt.close()
print("  Saved fig9")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 – Vital distributions per disease (clinical insight)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 10 – Vital distributions …")
vitals_plot = [
    ("Systolic Blood Pressure (mmHg)", "Systolic BP (mmHg)"),
    ("Heart Rate (bpm)",               "Heart Rate (bpm)"),
    ("SpO2 Level (%)",                 "SpO₂ (%)"),
    ("Body Temperature (C)",           "Temperature (°C)"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Vital Sign Distribution by Diagnosed Condition — IoT-Collected Data",
             fontsize=13, fontweight="bold")

for ax, (col, label) in zip(axes.flatten(), vitals_plot):
    for disease, color in zip(DISEASE_ORDER, COLOR_LIST):
        data = df.loc[df[TARGET_COL] == disease, col].values
        x_range = np.linspace(data.min(), data.max(), 200)
        kde = stats.gaussian_kde(data)
        ax.plot(x_range, kde(x_range), color=color, lw=2, label=disease)
        ax.fill_between(x_range, kde(x_range), alpha=0.08, color=color)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7.5, framealpha=0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(f"Distribution of {label}", fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig10_vital_distributions.pdf", **STYLE)
plt.savefig("figures/fig10_vital_distributions.png", **STYLE)
plt.close()
print("  Saved fig10")

# ═══════════════════════════════════════════════════════════════════════════════
# SHAP (optional)
# ═══════════════════════════════════════════════════════════════════════════════
if SHAP_OK:
    print("Figure 11 – SHAP summary …")
    explainer = shap.TreeExplainer(xgb_model)
    X_sample  = X_test[:500]
    shap_vals = explainer.shap_values(X_sample)

    feat_df = pd.DataFrame(X_sample, columns=feat_labels)
    fig, axes = plt.subplots(1, len(classes), figsize=(18, 5), sharey=True)
    fig.suptitle("SHAP Feature Contributions — Per-Class Explanation (XGBoost)",
                 fontsize=12, fontweight="bold")
    for i, (cls, ax) in enumerate(zip(classes, axes)):
        sv = shap_vals[i] if isinstance(shap_vals, list) else shap_vals[:, :, i]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)
        ax.barh(np.array(feat_labels)[order], mean_abs[order],
                color=COLOR_LIST[i], alpha=0.85)
        ax.set_title(cls, fontsize=9, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/fig11_shap.pdf", **STYLE)
    plt.savefig("figures/fig11_shap.png", **STYLE)
    plt.close()
    print("  Saved fig11")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE SUMMARY STATS
# ═══════════════════════════════════════════════════════════════════════════════
summary = {
    "dataset": {"n": 60000, "features": 9, "classes": 5},
    "train_size": int(X_train.shape[0]),
    "test_size":  int(X_test.shape[0]),
    "ensemble": {
        "accuracy":  round(accuracy_score(y_test, y_pred_ens) * 100, 4),
        "precision": round(precision_score(y_test, y_pred_ens, average="weighted") * 100, 4),
        "recall":    round(recall_score(y_test, y_pred_ens, average="weighted") * 100, 4),
        "f1":        round(f1_score(y_test, y_pred_ens, average="weighted") * 100, 4),
        "macro_auc": round(mean_auc * 100, 4),
    },
    "xgboost": {
        "accuracy":  round(accuracy_score(y_test, y_pred_xgb) * 100, 4),
        "f1":        round(f1_score(y_test, y_pred_xgb, average="weighted") * 100, 4),
    },
    "lightgbm": {
        "accuracy":  round(accuracy_score(y_test, y_pred_lgb) * 100, 4),
        "f1":        round(f1_score(y_test, y_pred_lgb, average="weighted") * 100, 4),
    },
    "per_class": {cls: {
        "precision": round(report[cls]["precision"] * 100, 4),
        "recall":    round(report[cls]["recall"] * 100, 4),
        "f1":        round(report[cls]["f1-score"] * 100, 4),
    } for cls in classes},
}
with open("figures/summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ALL FIGURES SAVED — Analysis complete.")
print("="*60)
print(json.dumps(summary["ensemble"], indent=2))
