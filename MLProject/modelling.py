import os
import glob
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

# 0. ARGUMEN CLI (untuk MLflow Projects)
parser = argparse.ArgumentParser()
parser.add_argument("--test_size",    type=float, default=0.2)
parser.add_argument("--random_state", type=int,   default=42)
args, _ = parser.parse_known_args()

# 1. KONFIGURASI
EXPERIMENT_NAME = "Titanic Survival Prediction"
MODEL_NAME      = "titanic-survival-v1"
TARGET_COLUMN   = "Survived"
TEST_SIZE       = args.test_size
RANDOM_STATE    = args.random_state

# 2. LOAD DATA
def load_preprocessed_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for pattern in ["*preprocessed*.csv", "*preprocessing*/*.csv",
                    "*preprocessing*.csv", "*.csv"]:
        candidates.extend(glob.glob(os.path.join(script_dir, pattern)))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"Tidak ditemukan file CSV di: {script_dir}")
    csv_path = candidates[0]
    print(f"\n[INFO] File data : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Shape     : {df.shape}")
    print(f"[INFO] Kolom     : {list(df.columns)}")
    return df

# 3. ENCODE KATEGORIKAL
def encode_if_needed(df):
    le = LabelEncoder()
    encoded = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
        encoded.append(col)
    if encoded:
        print(f"[INFO] Encoded   : {encoded}")
    return df

# 4. TRAINING + LOGGING
def train_model(X_train, X_test, y_train, y_test,
                model, model_name, feature_names):

    with mlflow.start_run(run_name=model_name, nested=True):

        mlflow.sklearn.autolog(
            log_input_examples   = False,
            log_model_signatures = False,
            log_models           = False,
            log_datasets         = True,
            silent               = True,
            disable_for_unsupported_versions = True,
        )

        # Training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ── Metrics test manual ───────────────────────────────────────
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_metric("test_accuracy",  acc)
        mlflow.log_metric("test_f1_score",  f1)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall",    rec)

        # ── Cross-validation ──────────────────────────────────────────
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        mlflow.log_metric("cv_mean_accuracy", float(cv.mean()))
        mlflow.log_metric("cv_std_accuracy",  float(cv.std()))

        # ── Tags ──────────────────────────────────────────────────────
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("dataset",    "titanic_preprocessed")
        mlflow.set_tag("target_col", TARGET_COLUMN)
        mlflow.set_tag("n_features", str(len(feature_names)))

        # ── ARTIFACT 1: confusion matrix + classification report ──────
        cm = confusion_matrix(y_test, y_pred)
        cm_fname = f"confusion_matrix_{model_name.replace(' ', '_')}.txt"
        with open(cm_fname, "w") as f:
            f.write(f"=== {model_name} ===\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))
        mlflow.log_artifact(cm_fname)
        os.remove(cm_fname)

        # ── ARTIFACT 2: feature importance CSV (RF & GB saja) ─────────
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature":    feature_names,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            fi_fname = f"feature_importance_{model_name.replace(' ', '_')}.csv"
            fi.to_csv(fi_fname, index=False)
            mlflow.log_artifact(fi_fname)
            os.remove(fi_fname)

        # ── LOG MODEL ─────────────────────────────────────────────────
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(tmp_dir, "model")

        signature     = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]

        mlflow.sklearn.save_model(
            sk_model      = model,
            path          = model_dir,
            signature     = signature,
            input_example = input_example,
            pip_requirements = ["scikit-learn", "pandas", "numpy"],
        )

        mlflow.log_artifacts(model_dir, artifact_path="model")

        mlflow.sklearn.log_model(
            sk_model              = model,
            artifact_path         = "model",
            signature             = signature,
            input_example         = input_example,
            registered_model_name = MODEL_NAME,
        )

        shutil.rmtree(tmp_dir)

        print(f"\n  ✔  {model_name}")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     F1-Score  : {f1:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     CV Acc    : {cv.mean():.4f} ± {cv.std():.4f}")
        return acc

# 5. MAIN
def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"[INFO] Tracking  : sqlite:///mlflow.db")
    print(f"[INFO] Experiment: {EXPERIMENT_NAME}")
    print(f"[INFO] test_size : {TEST_SIZE} | random_state: {RANDOM_STATE}")

    df = load_preprocessed_data()
    df = encode_if_needed(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom '{TARGET_COLUMN}' tidak ditemukan!\n"
            f"Tersedia: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = list(X.columns)

    print(f"[INFO] Target    : '{TARGET_COLUMN}' | Kelas: {sorted(y.unique())}")
    print(f"[INFO] Fitur ({len(feature_names)}) : {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")

    models = {
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }

    print("\n" + "=" * 55)
    print("   TRAINING  —  MLflow Autolog")
    print("=" * 55)

    results = {}
    for name, mdl in models.items():
        results[name] = train_model(
            X_train, X_test, y_train, y_test, mdl, name, feature_names
        )

    print("\n" + "=" * 55)
    print("   RINGKASAN HASIL")
    print("=" * 55)
    best = max(results, key=results.get)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        tag = "  ← BEST ★" if name == best else ""
        print(f"   {name:<22} Accuracy: {acc:.4f}{tag}")

    best_model_obj = models[best]
    joblib.dump(best_model_obj, "model.pkl")
    print(f"\n   Model terbaik ({best}) disimpan sebagai: model.pkl")

    print(f"\n✅ Training selesai!")
    print(f"   Data tersimpan di: mlflow.db")
    print(f"\n   Jalankan MLflow UI di terminal:")
    print(f"   > mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"   Buka browser → http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
