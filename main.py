import pandas as pd
import regex
from sklearn.model_selection import train_test_split
df = pd.read_csv('IMDB_dataset.csv')

#joblib is used to save the model and vectorizer for future use
import joblib


print(df.head())
print(df.shape)

#cleaning the br tags from the review column
df['review'] = df['review'].apply(lambda x: regex.sub('<br />','',x))
#changing the sentiment column to binary values
def sentiment_to_binary(sentiment):
    if sentiment == 'positive':
        return 1
    else:
        return 0


df['label'] = df['sentiment'].map({'positive':1 , 'negative' : 0})

print(df.head())


X =df['review']
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(X_train_vec.shape)
print(X_test_vec.shape)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vec, Y_train)


from sklearn.metrics import accuracy_score

predictions = model.predict(X_test_vec)
accuracy = accuracy_score(Y_test, predictions)
print(accuracy)


results = pd.DataFrame({
    'review': X_test,
    'true_label': Y_test,
    'predicted_label': predictions
})

wrong = results[results['true_label'] != results['predicted_label']]
print(wrong.shape)
print(wrong.head(10))

joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')

