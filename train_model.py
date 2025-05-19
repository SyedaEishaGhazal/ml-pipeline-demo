from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test = load_and_preprocess_data()

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "iris_model.pkl")
