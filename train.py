import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from config import DATA_PROCESSED, FEATURE_COLUMNS, MOOD_LABELS, MODELS_DIR, NUM_MOODS, RANDOM_SEED, RESULTS_DIR

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED, "test.csv"))
    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df["mood_code"]
    y_test = test_df["mood_code"]
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm


def cross_validate_model(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
    return scores, f1_scores


def save_confusion_matrix(cm, model_name, results_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    mood_names = [MOOD_LABELS[i] for i in range(NUM_MOODS)]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=mood_names,
        yticklabels=mood_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(results_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    train_df = pd.read_csv(os.path.join(DATA_PROCESSED, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED, "test.csv"))
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    X_full = full_df[FEATURE_COLUMNS]
    y_full = full_df["mood_code"]

    models = get_models()
    results = {}

    print("=" * 60)
    print("5-Fold Stratified Cross-Validation")
    print("=" * 60)

    for name, model in models.items():
        cv_scores, cv_f1 = cross_validate_model(model, X_full, y_full, cv=5)
        print(f"\n{name}:")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  CV F1 (weighted): {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")

    print(f"\n{'=' * 60}")
    print("Train/Test Evaluation (80/20 stratified split)")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc, report, cm = evaluate_model(model, X_test, y_test)
        results[name] = {"accuracy": acc, "report": report, "cm": cm, "model": model}

        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODELS_DIR, filename))

        print(f"\n{'=' * 40}")
        print(f"{name} — Test Accuracy: {acc:.4f}")
        print(f"\nPer-class metrics:")
        print(report)

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_acc = results[best_name]["accuracy"]

    joblib.dump(results[best_name]["model"], os.path.join(MODELS_DIR, "best_model.pkl"))
    save_confusion_matrix(results[best_name]["cm"], best_name, RESULTS_DIR)

    print(f"\n{'Model':<20}{'Test Accuracy':<16}{'CV Mean':<16}{'CV Std'}")
    print("-" * 55)

    cv_results = {}
    for name in ["Decision Tree", "Random Forest", "KNN"]:
        model = models[name]
        cv_s, cv_f1 = cross_validate_model(model, X_full, y_full, cv=5)
        cv_results[name] = cv_s
        print(f"{name:<20}{results[name]['accuracy']:<16.4f}{cv_s.mean():<16.4f}{cv_s.std():.4f}")

    print(f"\nBest (test accuracy): {best_name} ({best_acc:.4f})")
    print(f"\nNote: CV accuracy may differ from test accuracy due to")
    print(f"  class imbalance (romantic={sum(y_full == 4)}, calm={sum(y_full == 2)},")
    print(f"  happy={sum(y_full == 5)}, energetic={sum(y_full == 7)})")


if __name__ == "__main__":
    main()
