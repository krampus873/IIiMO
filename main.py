import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def price_category(price):
    if price < 150000:
        return 0
    elif price < 300000:
        return 1
    else:
        return 2


def main():
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("Задачи регрессии и классификации")
    print("=" * 60)

    # 1. Загрузка датасета
    df = pd.read_csv("data/housing.csv")

    print("\nПервые 5 строк датасета:")
    print(df.head())

    print("\nРазмер датасета:", df.shape)

    print("\nНазвания столбцов:")
    print(df.columns.tolist())

    print("\nПропущенные значения:")
    print(df.isnull().sum())

    # 2. Предобработка данных
    df = df.dropna()

    if "ocean_proximity" in df.columns:
        df = df.drop("ocean_proximity", axis=1)

    print("\nРазмер датасета после очистки:", df.shape)

    # =========================
    # 3. ЗАДАЧА РЕГРЕССИИ
    # =========================
    print("\n" + "=" * 60)
    print("1. ЗАДАЧА РЕГРЕССИИ")
    print("=" * 60)

    target_reg = "median_house_value"

    X_reg = df.drop(target_reg, axis=1)
    y_reg = df[target_reg]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print("\nРазмер обучающей выборки:", X_train_reg.shape)
    print("Размер тестовой выборки:", X_test_reg.shape)

    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)

    reg_model = LinearRegression()
    reg_model.fit(X_train_reg_scaled, y_train_reg)

    y_pred_reg = reg_model.predict(X_test_reg_scaled)

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
    plt.title("Линейная регрессия: реальные и предсказанные значения")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =========================
    # 4. ЗАДАЧА КЛАССИФИКАЦИИ
    # =========================
    print("\n" + "=" * 60)
    print("2. ЗАДАЧА КЛАССИФИКАЦИИ")
    print("=" * 60)

    df["price_category"] = df["median_house_value"].apply(price_category)

    X_clf = df.drop(["median_house_value", "price_category"], axis=1)
    y_clf = df["price_category"]

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    print("\nРазмер обучающей выборки:", X_train_clf.shape)
    print("Размер тестовой выборки:", X_test_clf.shape)

    scaler_clf = StandardScaler()
    X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler_clf.transform(X_test_clf)

    clf_model = LogisticRegression(max_iter=2000)
    clf_model.fit(X_train_clf_scaled, y_train_clf)

    y_pred_clf = clf_model.predict(X_test_clf_scaled)

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


if __name__ == "__main__":
    main()