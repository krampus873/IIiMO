import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def generate_data():
    """Генерация тестовых данных для задачи 'Выборы президента'"""
    print("Генерация тестовых данных...")
    
    np.random.seed(42)
    
    # 500 примеров, 12 бинарных признаков (ответы избирателей)
    X = np.random.randint(0, 2, size=(500, 12))
    
    # 2 класса: [1, 0] - правящая партия, [0, 1] - оппозиция
    # Делаем это зависимым от входных данных для осмысленности
    # Если сумма ответов > 6, то правящая партия побеждает
    Y = np.zeros((500, 2))
    for i in range(500):
        if X[i].sum() > 6:
            Y[i] = [1, 0]  # Правящая партия
        else:
            Y[i] = [0, 1]  # Оппозиция
    
    # Сохраняем в файлы
    np.savetxt('data/dataIn.txt', X.T, fmt='%d')  # Транспонируем для формата 12×N
    np.savetxt('data/dataOut.txt', Y.T, fmt='%d')
    
    print(f"Данные сгенерированы: {X.shape[0]} примеров")
    print(f"Входные данные (12 признаков): data/dataIn.txt")
    print(f"Выходные данные (2 класса): data/dataOut.txt\n")
    
    return X, Y


def load_data():
    """Загрузка данных из файлов"""
    print("Загрузка данных...")
    
    X = np.loadtxt('data/dataIn.txt').T  # Транспонируем обратно в N×12
    Y = np.loadtxt('data/dataOut.txt').T
    
    print(f"Размер входных данных: {X.shape}")
    print(f"Размер выходных данных: {Y.shape}\n")
    
    return X, Y


def create_mlp_model(input_dim, hidden_neurons=16):
    """Создание MLP с одним скрытым слоем и сигмоидой"""
    model = keras.Sequential([
        keras.layers.Dense(hidden_neurons, activation='sigmoid', input_shape=(input_dim,)),  # Скрытый слой с logsig
        keras.layers.Dense(2, activation='softmax')  # Выходной слой для 2 классов
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №5")
    print("Нейронные сети. Многослойный перцептрон (MLP)")
    print("Задание: 'Выборы президента'")
    print("=" * 70)
    print()
    
    # 1. Генерация или загрузка данных
    import os
    if not os.path.exists('data/dataIn.txt'):
        os.makedirs('data', exist_ok=True)
        X, Y = generate_data()
    else:
        X, Y = load_data()
    
    # 2. Разделение на train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    
    print(f"Обучающая выборка: {X_train.shape[0]} примеров")
    print(f"Тестовая выборка: {X_test.shape[0]} примеров\n")
    
    # 3. Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Создание MLP
    print("=" * 70)
    print("СОЗДАНИЕ И ОБУЧЕНИЕ MLP")
    print("=" * 70)
    
    model = create_mlp_model(input_dim=12, hidden_neurons=16)
    
    print("\nАрхитектура модели:")
    model.summary()
    
    # 5. Обучение модели
    print("\nОбучение модели...")
    history = model.fit(
        X_train_scaled, Y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test_scaled, Y_test),
        verbose=0
    )
    
    print("Обучение завершено!\n")
    
    # 6. Оценка модели
    print("=" * 70)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 70)
    
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = np.argmax(Y_test, axis=1)
    
    accuracy = accuracy_score(y_test_labels, y_pred)
    cm = confusion_matrix(y_test_labels, y_pred)
    report = classification_report(y_test_labels, y_pred, target_names=['Правящая партия', 'Оппозиция'])
    
    print(f"\nAccuracy MLP: {accuracy:.4f}")
    print("\nМатрица ошибок:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # 7. Визуализация матрицы ошибок
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Правящая партия', 'Оппозиция'],
                yticklabels=['Правящая партия', 'Оппозиция'])
    plt.title('Матрица ошибок — MLP')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    plt.show()
    
    # 8. Графики обучения
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Изменение функции потерь')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Изменение точности')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 9. Сравнение с классическими алгоритмами
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ С КЛАССИЧЕСКИМИ АЛГОРИТМАМИ")
    print("=" * 70)
    
    # Преобразуем Y обратно в одномерные метки для scikit-learn
    y_train_1d = np.argmax(Y_train, axis=1)
    y_test_1d = np.argmax(Y_test, axis=1)
    
    # Логистическая регрессия
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train_1d)
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test_1d, y_pred_lr)
    
    print(f"\nLogistic Regression Accuracy: {acc_lr:.4f}")
    
    # Случайный лес
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train_1d)  # RF не требует нормализации
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test_1d, y_pred_rf)
    
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    
    # Сравнительная таблица
    print("\n" + "=" * 70)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 70)
    
    import pandas as pd
    results = pd.DataFrame({
        'Модель': ['MLP (Neural Network)', 'Logistic Regression', 'Random Forest'],
        'Accuracy': [accuracy, acc_lr, acc_rf]
    })
    
    print("\n", results)
    
    # График сравнения
    plt.figure(figsize=(8, 5))
    plt.bar(results['Модель'], results['Accuracy'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.ylabel('Accuracy')
    plt.title('Сравнение моделей')
    plt.ylim(0, 1.1)
    for i, v in enumerate(results['Accuracy']):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    main()