# Fake Job Postings Detection (Simple Python Demo)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 1. Load dataset (download 'fake_job_postings.csv' from Kaggle)
data = pd.read_csv("fake_job_postings.csv")

# Use only 'description' and 'fraudulent' columns
df = data[['description', 'fraudulent']].dropna()

# 2. Split data
X = df['description']
y = df['fraudulent']   # 0 = Real, 1 = Fake

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Convert text → numeric (Bag of Words)
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train a Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 6. Test with a custom job description
sample_job = ["Looking for freshers, pay deposit before joining, easy online work."]
sample_vec = vectorizer.transform(sample_job)
print("Prediction:", "Fake" if model.predict(sample_vec)[0] == 1 else "Real")
