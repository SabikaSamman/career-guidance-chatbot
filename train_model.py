import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load the dataset
df=pd.read_csv('Career20Dataset.csv')
df.sample(5)

# Preprocess the data
X=df['question']
y=df['role']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer= TfidfVectorizer(lowercase=True, stop_words='english')
X_train_vectorized= vectorizer.fit_transform(X_train)
X_test_vectorized= vectorizer.transform(X_test)

# Train the model
model= MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred= model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Accuracy: {accuracy}, F1 Score: {f1}')

# Save the model and vectorizer
joblib.dump(model, 'intent_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')