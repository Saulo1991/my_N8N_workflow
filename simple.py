from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Dados
iris = load_iris()
X = iris.data
y = iris.target

# Modelo
model = DecisionTreeClassifier()
model.fit(X, y)

# Previsão
print(model.predict([X[0]]))
