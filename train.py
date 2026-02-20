import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("AIvsHuman.csv", encoding="latin-1")

# Select columns
X = df["text"]
y = df["generated"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text → numbers
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM model
model = LinearSVC()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Results
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Report:\n", classification_report(y_test, y_pred))
# Test custom input
while True:
    user_text = input("\nEnter text to test (or type exit): ")

    if user_text.lower() == "exit":
        break

    text_vec = vectorizer.transform([user_text])
    prediction = model.predict(text_vec)[0]

    if prediction == 1:
        print(" Human written text")
    else:
        print(" AI generated text")