from comet_ml import Experiment
import pandas as pd
import numpy as np
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------
# COMET EXPERIMENT
# ---------------------------------------------

# Create experiment
experiment = Experiment(
    api_key="RRookOInE54DzbqU4ird7iTk0",
    project_name="ML Leader",
    workspace="sakshi-kannor"
)


# ---------------------------------------------
# LOAD DATASET
# ---------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Log dataset info
experiment.log_dataset_hash(X)

# ---------------------------------------------
# DATA SPLIT
# ---------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# MODELS
# ---------------------------------------------

models = {

    "LogisticRegression": LogisticRegression(max_iter=500),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10
    ),

    "DecisionTree": DecisionTreeClassifier(),

    "GradientBoosting": GradientBoostingClassifier(),

    "SVM": SVC(probability=True)

}

# ---------------------------------------------
# RESULTS STORAGE
# ---------------------------------------------

results = []

# ---------------------------------------------
# TRAIN MODELS
# ---------------------------------------------

for name, model in models.items():

    print("\nTraining:", name)

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    train_time = end - start

    pred = model.predict(X_test)

    # ---------------------------------------------
    # METRICS
    # ---------------------------------------------

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([name, acc, prec, rec, f1, train_time])

    # ---------------------------------------------
    # LOG METRICS
    # ---------------------------------------------

    experiment.log_metric(f"{name}_accuracy", acc)
    experiment.log_metric(f"{name}_precision", prec)
    experiment.log_metric(f"{name}_recall", rec)
    experiment.log_metric(f"{name}_f1", f1)
    experiment.log_metric(f"{name}_train_time", train_time)

    # ---------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------

    cm = confusion_matrix(y_test, pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")

    experiment.log_figure(figure_name=f"{name}_confusion_matrix")

    plt.close()

    # ---------------------------------------------
    # ROC CURVE
    # ---------------------------------------------

    probs = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend()

    experiment.log_figure(figure_name=f"{name}_roc_curve")

    plt.close()

    # ---------------------------------------------
    # FEATURE IMPORTANCE (TREE MODELS)
    # ---------------------------------------------

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        plt.figure(figsize=(10,6))

        sns.barplot(
            x=importance,
            y=X.columns
        )

        plt.title(f"{name} Feature Importance")

        experiment.log_figure(figure_name=f"{name}_feature_importance")

        plt.close()

    # ---------------------------------------------
    # LOG MODEL
    # ---------------------------------------------

    experiment.log_model(name, model)


# ---------------------------------------------
# RESULTS DATAFRAME
# ---------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Training Time"
    ]
)

print("\nModel Comparison\n")
print(results_df)

# ---------------------------------------------
# MODEL COMPARISON GRAPH
# ---------------------------------------------

plt.figure(figsize=(10,6))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=results_df
)

plt.title("Model Accuracy Comparison")

experiment.log_figure(figure_name="model_accuracy_comparison")

plt.close()


# ---------------------------------------------
# SAVE RESULTS
# ---------------------------------------------

results_df.to_csv("model_results.csv", index=False)

experiment.log_asset("model_results.csv")

experiment.end()