import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("IMDB Dataset.csv")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df)
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer(stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
X_test_counts = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_counts)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
def classify_review(review):
    review_counts = vectorizer.transform([review])
    return 'positive' if clf.predict(review_counts)[0] == 1 else 'negative'
review = "the movie was phenomenal"
print(f'Review: {review}\nSentiment: {classify_review(review)}')
review2= "the movie was a torture "
print(f'Review2: {review2}\nSentiment: {classify_review(review2)}')
