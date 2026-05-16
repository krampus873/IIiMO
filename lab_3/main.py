import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


def main():
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №3")
    print("Деревья решений в задачах классификации и регрессии. ROC-кривая")
    print("=" * 60)

    # 1. Загрузка
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

    # 3. Регрессия

    print("\n" + "=" * 60)
    print("1. ЗАДАЧА РЕГРЕССИИ — DecisionTreeRegressor")
    print("=" * 60)

    target_reg = "median_house_value"

    X_reg = df.drop(target_reg, axis=1)
    y_reg = df[target_reg]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)

    y_pred_reg = reg_model.predict(X_test_reg)

    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    print("\nРезультаты регрессии:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    plt.xlabel("Реальные значения")
    plt.ylabel("Предсказанные значения")
    plt.title("Decision Tree Regressor: реальные и предсказанные значения")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    plot_tree(
        reg_model,
        feature_names=X_reg.columns,
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title("Дерево решений для регрессии")
    plt.tight_layout()
    plt.show()


    # 4. ЗАДАЧА КЛАССИФИКАЦИИ

    print("\n" + "=" * 60)
    print("2. ЗАДАЧА КЛАССИФИКАЦИИ — DecisionTreeClassifier")
    print("=" * 60)

    # Бинарная классификация: дорогой дом или нет
    median_value = df["median_house_value"].median()
    df["expensive_house"] = (df["median_house_value"] > median_value).astype(int)

    X_clf = df.drop(["median_house_value", "expensive_house"], axis=1)
    y_clf = df["expensive_house"]

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    clf_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf_model.fit(X_train_clf, y_train_clf)

    y_pred_clf = clf_model.predict(X_test_clf)
    y_proba_clf = clf_model.predict_proba(X_test_clf)

    acc = accuracy_score(y_test_clf, y_pred_clf)
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    report = classification_report(y_test_clf, y_pred_clf)

    print("\nРезультаты классификации:")
    print(f"Accuracy: {acc:.4f}")

    print("\nМатрица ошибок:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Матрица ошибок")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    plot_tree(
        clf_model,
        feature_names=X_clf.columns,
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title("Дерево решений для классификации")
    plt.tight_layout()
    plt.show()


    # 5. ROC-кривая

    print("\n" + "=" * 60)
    print("3. ROC-КРИВАЯ")
    print("=" * 60)

    fpr, tpr, thresholds = roc_curve(y_test_clf, y_proba_clf[:, 1])
    roc_auc = auc(fpr, tpr)

    print(f"ROC-AUC: {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='o', label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Случайная модель")
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC-кривая")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()