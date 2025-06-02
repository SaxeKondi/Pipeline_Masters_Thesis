import numpy as np
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree

# --- Config ---
FEATURES_PATH = "texture_analysis/feature_analysis/feature_vectors.npz"
MODEL_PATH = "texture_analysis/models/rf_texture_model.joblib"
FEATURE_TXT_PATH = "texture_analysis/models/rf_texture_features.txt"
PLOT_PATH = "texture_analysis/models/rf_feature_importances.png"
TREE_PLOT_PATH = "texture_analysis/models/shallow_tree_example.png"
CLASS_REPORT_PATH = "texture_analysis/models/rf_classification_report.txt"
CM_PLOT_PATH = "texture_analysis/models/rf_confusion_matrix.png"
CV_RESULTS_PATH = "texture_analysis/models/grid_search_results.csv"
FEATURE_NAMES = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# --- Load features ---
data = np.load(FEATURES_PATH)
X_list, y_list = [], []

for class_name in sorted(data.files):
    feats = data[class_name]
    X_list.append(feats)
    y_list.extend([class_name] * len(feats))

X = np.vstack(X_list)
y = np.array(y_list)

# --- Encode class labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Cross-validation setup ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Optimal hyperparameter Grid Search ---
use_grid_search = True

if use_grid_search:
    print("Running Grid Search for Random Forest hyperparameters...")

    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [6, 8, 10, 12],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }

    base_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )

    grid = GridSearchCV(
        base_rf,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y_encoded)

    rf = grid.best_estimator_

    print("Best parameters from Grid Search:", grid.best_params_)
    print(f"Best cross-validated accuracy: {grid.best_score_:.3f}")

    # Save full grid search results
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(CV_RESULTS_PATH, index=False)
    print(f"Saved Grid Search results to: {CV_RESULTS_PATH}")
else:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    scores = cross_val_score(rf, X, y_encoded, cv=cv)
    print(f"Cross-validation accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

# --- Final training ---
rf.fit(X, y_encoded)
joblib.dump(rf, MODEL_PATH)
print(f"Saved RF model to: {MODEL_PATH}")

# --- Save feature list ---
with open(FEATURE_TXT_PATH, "w") as f:
    for name in FEATURE_NAMES:
        f.write(f"{name}\n")

# --- Plot feature importance ---
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[sorted_idx], align="center")
plt.xticks(range(len(importances)), np.array(FEATURE_NAMES)[sorted_idx], rotation=45)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()
print(f"Saved feature importances plot to: {PLOT_PATH}")

# --- Predict and Evaluate ---
if not use_grid_search:
    y_pred = cross_val_predict(rf, X, y_encoded, cv=cv)
else:
    y_pred = rf.predict(X)

report = classification_report(y_encoded, y_pred, target_names=le.classes_)
print("Classification Report:\n", report)

# Save report
with open(CLASS_REPORT_PATH, "w") as f:
    f.write("Classification Report (Final Model)\n")
    f.write(report)
print(f"Saved classification report to: {CLASS_REPORT_PATH}")

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_PLOT_PATH)
plt.close()
print(f"Saved confusion matrix plot to: {CM_PLOT_PATH}")
