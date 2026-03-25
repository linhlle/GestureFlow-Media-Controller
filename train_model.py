import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    df = pd.read_csv('gesture_data.csv')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} frames")
    print(f"Testing on {len(X_test)} frames")

    # n_estimators=100: We create 100 'Decision Trees' that will vote on the gesture.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*30)
    print(f"FINAL ACCURACY: {accuracy * 100:.2f}%")
    print("="*30)

    print("\nDetailed Breakdown:")
    print(classification_report(y_test, y_pred))

    with open('gesture_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("\nSuccess: 'gesture_classifier.pkl' is ready for action!")

if __name__ == "__main__":
    train_model()