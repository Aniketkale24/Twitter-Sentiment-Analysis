import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. LOAD DATA
# -------------------------------
data = pd.read_csv("data/train.csv")
data = data.dropna(subset=['text'])

# keep only required columns
data = data[['text', 'sentiment']]

# -------------------------------
# 2. PREPROCESS FUNCTION
# -------------------------------
def preprocess(text):
    text = str(text)   # 🔥 IMPORTANT FIX
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Apply preprocessing
data['cleaned'] = data['text'].apply(preprocess)

# -------------------------------
# 3. FEATURES & LABELS
# -------------------------------
X = data['cleaned']
y = data['sentiment']

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# -------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. TRAIN MODEL
# -------------------------------
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

# -------------------------------
# 6. EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. USER INPUT
# -------------------------------
while True:
    user_input = input("\nEnter a tweet (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        break

    cleaned = preprocess(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)

    print("Predicted Sentiment:", prediction[0])
