from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Better dataset (~40 samples)
texts = [
    "i love this product", "this is amazing", "very happy with this",
    "excellent service", "i am feeling great", "this is fantastic",
    "absolutely wonderful", "i enjoyed it", "so good", "best experience",

    "i hate this", "this is bad", "very disappointing",
    "worst experience ever", "i am sad", "this is terrible",
    "not good at all", "very poor service", "awful product", "i regret this",

    "this is okay", "not bad", "average experience",
    "it is fine", "nothing special", "could be better",
    "i feel neutral", "just okay", "neither good nor bad", "normal experience"
]

labels = [
    "positive","positive","positive","positive","positive","positive","positive","positive","positive","positive",
    "negative","negative","negative","negative","negative","negative","negative","negative","negative","negative",
    "neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral"
]

# Clean data
texts = [clean_text(t) for t in texts]

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Evaluate
predictions = model.predict(X)
print("Model Accuracy:", accuracy_score(labels, predictions))

# CLI
print("\nSentiment Analyzer (type 'exit' to quit)\n")

while True:
    user_input = input("Enter text: ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    cleaned = clean_text(user_input)
    X_input = vectorizer.transform([cleaned])
    prediction = model.predict(X_input)

    print("Predicted Sentiment:", prediction[0])
    print("-" * 40)