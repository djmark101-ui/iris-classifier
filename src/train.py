from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

print("Iris Flower Classification")
print("--------------------------")
print(f"Accuracy: {accuracy:.2f}")
print()
print("Confusion Matrix:")
print(matrix)
print()
print("Classification Report:")
print(report)
