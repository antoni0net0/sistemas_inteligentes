import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from dataset import dataset, rotulo

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(dataset)
y = np.array(rotulo)

# Validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = []
y_true_total = []
y_pred_total = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Fold {fold + 1} - Acurácia: {acc:.2f}")
    acc_scores.append(acc)

    y_true_total.extend(y_test)
    y_pred_total.extend(y_pred)

# Relatório final
media = np.mean(acc_scores)
desvio = np.std(acc_scores)

print(f"\nMédia de acurácia nos folds: {media:.4f}")
print(f"Desvio padrão das acurácias: {desvio:.4f}")
print("\nRelatório de classificação acumulado (todos os folds):\n")
print(confusion_matrix(y_true_total, y_pred_total))
print(classification_report(y_true_total, y_pred_total, digits=2))
