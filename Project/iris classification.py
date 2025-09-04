from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

print("--- Dataset Information ---")
print(f"Features (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")
print(f"Species: {iris.target_names}\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Data Splitting ---")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples\n")

print("========== K-Nearest Neighbors (KNN) ==========")
knn_classifier = KNeighborsClassifier(n_neighbors=3)

print("Training the KNN model...")
knn_classifier.fit(X_train, y_train)
print("KNN model trained successfully.\n")

y_pred_knn = knn_classifier.predict(X_test)

print("--- KNN Model Evaluation ---")
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy: {accuracy_knn:.2f} ({accuracy_knn * 100:.2f}%)")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

print("\n========== Decision Tree Classifier ==========")
dt_classifier = DecisionTreeClassifier(random_state=42)

print("Training the Decision Tree model...")
dt_classifier.fit(X_train, y_train)
print("Decision Tree model trained successfully.\n")

y_pred_dt = dt_classifier.predict(X_test)

print("--- Decision Tree Model Evaluation ---")
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {accuracy_dt:.2f} ({accuracy_dt * 100:.2f}%)")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

print("\n--- Making a New Prediction ---")
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction_knn = knn_classifier.predict(new_flower)
prediction_dt = dt_classifier.predict(new_flower)

print(f"New flower features: {new_flower[0]}")
print(f"KNN Prediction: {iris.target_names[prediction_knn[0]]}")
print(f"Decision Tree Prediction: {iris.target_names[prediction_dt[0]]}")

new_flower_2 = np.array([[6.7, 3.0, 5.2, 2.3]])
prediction_knn_2 = knn_classifier.predict(new_flower_2)
prediction_dt_2 = dt_classifier.predict(new_flower_2)

print(f"\nNew flower features: {new_flower_2[0]}")
print(f"KNN Prediction: {iris.target_names[prediction_knn_2[0]]}")
print(f"Decision Tree Prediction: {iris.target_names[prediction_dt_2[0]]}")