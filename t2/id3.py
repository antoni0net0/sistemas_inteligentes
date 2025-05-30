from dataset import rotulo, dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

X = dataset
y = rotulo

# Treino/teste
X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X, y, random_state=400, test_size=0.4, stratify=y
)

# ID3
clf1 = DecisionTreeClassifier(criterion='entropy', random_state=400)

# Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=400)
cv_scores = cross_val_score(clf1, X, y, cv=cv, scoring='accuracy')

print("=== Validação Cruzada (5-fold) ===")
print("Acurácias em cada fold:", cv_scores)
print("Acurácia média:", np.mean(cv_scores))
print("Desvio padrão:", np.std(cv_scores))
print()

clf1.fit(X_train, y_clf_train)

# Avaliação do conjunto de teste
y_pred_clf1 = clf1.predict(X_test)
y_train_pred = clf1.predict(X_train)

print("=== Árvore de Decisão ===")
print("=== Avaliação com Test ===")
print(confusion_matrix(y_clf_test, y_pred_clf1))
print(classification_report(y_clf_test, y_pred_clf1))
print("=== Avaliação com Treino ===")
print(confusion_matrix(y_clf_train, y_train_pred))
print(classification_report(y_clf_train, y_train_pred))