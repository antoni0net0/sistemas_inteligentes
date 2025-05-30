import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from dataset import dataset, rotulo

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(dataset)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(rotulo)
y = to_categorical(y_encoded)

# Validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=18)
cv_scores = []

# Resultados
y_true_total = []
y_pred_total = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"\nTreinando Fold {fold + 1}")
    model.fit(X_train_cv, y_train_cv, epochs=100, batch_size=8, verbose=0)
    
    loss, accuracy = model.evaluate(X_val_cv, y_val_cv, verbose=0)
    print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}")
    cv_scores.append(accuracy)

    # Predições
    y_val_pred_probs = model.predict(X_val_cv)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    y_val_true = np.argmax(y_val_cv, axis=1)

    y_true_total.extend(y_val_true)
    y_pred_total.extend(y_val_pred)

# Estatísticas
media = np.mean(cv_scores)
desvio = np.std(cv_scores)
print(f"\nMédia de acurácia nos folds: {media:.4f}")
print(f"Desvio padrão das acurácias: {desvio:.4f}")

# Resultados finais
target_names = [str(c) for c in label_encoder.classes_]
print("\nMatriz de Confusão:\n", confusion_matrix(y_true_total, y_pred_total))
print("\nRelatório de Classificação:\n", classification_report(y_true_total, y_pred_total, target_names=target_names, digits=2))
