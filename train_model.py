import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    csv_path = 'gesture_data.csv'
    if not os.path.exists(csv_path):
        print("Error csv file")
        return
    df = pd.read_csv(csv_path)


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

    print("\nSuccess")

    target_names = ["Neutral", "L-Shape", "High-Five", "2-Finger"]
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Gesture Recognition Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    train_model()