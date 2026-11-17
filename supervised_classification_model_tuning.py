"""Supervised classification, model comparison and tuning."""

# If required, run:
# %pip install -U pandas numpy matplotlib seaborn scikit-learn joblib

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
print("scikit-learn:", sklearn.__version__)

DATA_PATH = Path("customer-churn-training.csv")
TARGET = "churned"
ID_COLUMNS = ["customer_id"]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Place customer-churn-training.csv beside this notebook "
        "or update DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
display(df.head())
display(df.isna().sum().to_frame("missing_values"))

if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")
if df[TARGET].isna().any():
    raise ValueError("Target contains missing values.")
if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("This notebook expects binary target values 0 and 1.")

df = df.drop_duplicates().copy()
df[TARGET] = df[TARGET].astype(int)

print("Target counts:")
display(df[TARGET].value_counts().sort_index().to_frame("count"))

# Exclude customer_id: identifiers should not be learned as predictors.
drop_cols = [TARGET] + [c for c in ID_COLUMNS if c in df.columns]
X = df.drop(columns=drop_cols).copy()
y = df[TARGET].copy()

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

class_counts = y.value_counts()
if len(class_counts) != 2 or class_counts.min() < 2:
    raise ValueError("Need at least two observations in each class.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train size:", len(X_train), "| Test size:", len(X_test))
print("Training class counts:")
display(y_train.value_counts())

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)

model_grids = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "param_grid": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear"]
        }
    },
    "Support Vector Machine": {
        "estimator": SVC(
            probability=True, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "param_grid": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["rbf"],
            "classifier__gamma": ["scale", "auto"]
        }
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 3],
            "classifier__min_samples_leaf": [1, 2]
        }
    },
    "Extra Trees": {
        "estimator": ExtraTreesClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 3],
            "classifier__min_samples_leaf": [1, 2]
        }
    }
}

# Keep folds feasible for the smallest class in the training partition.
min_train_class = int(y_train.value_counts().min())
n_splits = min(2, min_train_class)
if n_splits < 2:
    raise ValueError("Insufficient training examples for stratified CV.")

cv = StratifiedKFold(
    n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
)
print(f"Using {n_splits}-fold StratifiedKFold.")

searches = {}
comparison_rows = []

for model_name, config in model_grids.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", config["estimator"])
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["param_grid"],
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    search.fit(X_train, y_train)
    searches[model_name] = search

    comparison_rows.append({
        "Model": model_name,
        "Best CV F1": search.best_score_,
        "Best Parameters": search.best_params_
    })

    print(f"{model_name}: best CV F1 = {search.best_score_:.4f}")
    print("Best parameters:", search.best_params_)

cv_results = pd.DataFrame(comparison_rows).sort_values(
    "Best CV F1", ascending=False
).reset_index(drop=True)

display(cv_results)

test_rows = []
test_predictions = {}

for model_name, search in searches.items():
    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    y_score = best_pipeline.predict_proba(X_test)[:, 1]

    test_predictions[model_name] = (y_pred, y_score)

    test_rows.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": (
            roc_auc_score(y_test, y_score)
            if y_test.nunique() == 2 else np.nan
        ),
        "CV F1": search.best_score_
    })

model_comparison = pd.DataFrame(test_rows).sort_values(
    "CV F1", ascending=False
).reset_index(drop=True)

display(model_comparison.round(4))
model_comparison.to_csv("model_comparison.csv", index=False)

plt.figure(figsize=(8, 6))
ax = plt.gca()

for model_name, (y_pred, y_score) in test_predictions.items():
    if y_test.nunique() == 2:
        RocCurveDisplay.from_predictions(
            y_test, y_score, name=model_name, ax=ax
        )

plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
plt.title("ROC Curves - Tuned Classifiers")
plt.tight_layout()
plt.show()

for model_name, (y_pred, y_score) in test_predictions.items():
    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Retained", "Churned"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

champion_name = max(
    searches,
    key=lambda name: searches[name].best_score_
)
champion_search = searches[champion_name]
champion_model = champion_search.best_estimator_

print("Champion model:", champion_name)
print("Champion mean CV F1:", round(champion_search.best_score_, 4))
print("Champion parameters:", champion_search.best_params_)

joblib.dump(champion_model, "champion_customer_churn_model.joblib")
print("Saved: champion_customer_churn_model.joblib")

loaded_champion = joblib.load("champion_customer_churn_model.joblib")

new_customer = pd.DataFrame([{
    "tenure_months": 5,
    "support_tickets": 3,
    "monthly_spend_inr": 799,
    "last_login_days": 15,
    "plan_type": "Basic"
}]).reindex(columns=X.columns)

prediction = loaded_champion.predict(new_customer)[0]
probability = loaded_champion.predict_proba(new_customer)[0, 1]

print("Prediction:", "Churned" if prediction == 1 else "Retained")
print(f"Predicted churn probability: {probability:.4f}")

