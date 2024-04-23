import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

reviews = [
    ("This movie is fantastic!", "positive"),
    ("I didn't like this movie at all.", "negative"),
    ("The plot was engaging and the acting was superb.", "positive"),
    ("It was a waste of time.", "negative"),
    ("Highly recommend this film.", "positive")
]

texts, labels = zip(*reviews)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

y_pred = classifier.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

custom_reviews = [
    input("Enter Your Reviews:")
]

custom_reviews_counts = vectorizer.transform(custom_reviews)
custom_predictions = classifier.predict(custom_reviews_counts)

for review, prediction in zip(custom_reviews, custom_predictions):
    print(f"Review: {review} | Predicted Sentiment: {prediction}")
