import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc


def main():
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №4")
    print("Бэггинг и бустинг. Метод случайного леса, AdaBoost и градиентный бустинг")
    print("=" * 70)

    # 1. Загрузка датасета
    df = pd.read_csv("data/housing.csv")

    print("\nПервые 5 строк датасета:")
    print(df.head())

    print("\nРазмер датасета:", df.shape)

    print("\nНазвания столбцов:")
    print(df.columns.tolist())

    print("\nПропущенные значения:")
    print(df.isnull().sum())

    # 2. Очистка данных
    df = df.dropna()

    if "ocean_proximity" in df.columns:
        df = df.drop("ocean_proximity", axis=1)

    print("\nРазмер датасета после очистки:", df.shape)

    # 3. Бинарная классификация: дорогой дом или нет
    median_value = df["median_house_value"].median()
    df["expensive_house"] = (df["median_house_value"] > median_value).astype(int)

    X = df.drop(["median_house_value", "expensive_house"], axis=1)
    y = df["expensive_house"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # 4. СЛУЧАЙНЫЙ ЛЕС

    print("\n" + "=" * 70)
    print("1. RANDOM FOREST CLASSIFIER")
    print("=" * 70)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)

    print(f"\nAccuracy (Random Forest): {acc_rf:.4f}")
    print(f"OOB Accuracy: {rf_model.oob_score_:.4f}")
    print(f"OOB Error: {1 - rf_model.oob_score_:.4f}")

    print("\nМатрица ошибок (Random Forest):")
    print(cm_rf)

    print("\nClassification Report (Random Forest):")
    print(report_rf)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
    plt.title("Матрица ошибок — Random Forest")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    plt.show()


    # 5. ADABOOST

    print("\n" + "=" * 70)
    print("2. ADABOOST CLASSIFIER")
    print("=" * 70)

    ada_model = AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    )
    ada_model.fit(X_train, y_train)

    y_pred_ada = ada_model.predict(X_test)
    y_proba_ada = ada_model.predict_proba(X_test)[:, 1]

    acc_ada = accuracy_score(y_test, y_pred_ada)
    cm_ada = confusion_matrix(y_test, y_pred_ada)
    report_ada = classification_report(y_test, y_pred_ada)

    print(f"\nAccuracy (AdaBoost): {acc_ada:.4f}")

    print("\nМатрица ошибок (AdaBoost):")
    print(cm_ada)

    print("\nClassification Report (AdaBoost):")
    print(report_ada)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_ada, annot=True, fmt="d", cmap="Greens")
    plt.title("Матрица ошибок — AdaBoost")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    plt.show()


    # 6. GRADIENT BOOSTING

    print("\n" + "=" * 70)
    print("3. GRADIENT BOOSTING CLASSIFIER")
    print("=" * 70)

    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    y_pred_gb = gb_model.predict(X_test)
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]

    acc_gb = accuracy_score(y_test, y_pred_gb)
    cm_gb = confusion_matrix(y_test, y_pred_gb)
    report_gb = classification_report(y_test, y_pred_gb)

    print(f"\nAccuracy (Gradient Boosting): {acc_gb:.4f}")

    print("\nМатрица ошибок (Gradient Boosting):")
    print(cm_gb)

    print("\nClassification Report (Gradient Boosting):")
    print(report_gb)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Oranges")
    plt.title("Матрица ошибок — Gradient Boosting")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    plt.show()


    # 7. ROC-КРИВЫЕ

    print("\n" + "=" * 70)
    print("4. ROC-КРИВЫЕ")
    print("=" * 70)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    fpr_ada, tpr_ada, _ = roc_curve(y_test, y_proba_ada)
    auc_ada = auc(fpr_ada, tpr_ada)

    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
    auc_gb = auc(fpr_gb, tpr_gb)

    print(f"ROC-AUC Random Forest: {auc_rf:.4f}")
    print(f"ROC-AUC AdaBoost: {auc_ada:.4f}")
    print(f"ROC-AUC Gradient Boosting: {auc_gb:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.4f})")
    plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost (AUC = {auc_ada:.4f})")
    plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {auc_gb:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Случайная модель")

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC-кривые для ансамблевых моделей")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # 8. СРАВНЕНИЕ МОДЕЛЕЙ
  
    print("\n" + "=" * 70)
    print("5. СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)

    results = pd.DataFrame({
        "Модель": ["Random Forest", "AdaBoost", "Gradient Boosting"],
        "Accuracy": [acc_rf, acc_ada, acc_gb],
        "ROC-AUC": [auc_rf, auc_ada, auc_gb]
    })

    print(results)


if __name__ == "__main__":
    main()