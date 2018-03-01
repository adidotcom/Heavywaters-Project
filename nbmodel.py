import pandas as pd
# For splitting into training and testing data
from sklearn.model_selection import train_test_split
# For term frequency - inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
# For Naive Bayes Classification Modeling
from sklearn.naive_bayes import MultinomialNB

# For saving the model
from sklearn.externals import joblib

# Loading the dataset
df = pd.read_csv('/Users/adi/Downloads/shuffled-full-set-hashed.csv', names=["doctype", "details"])
y = df.doctype

# Count the instances of each document type
df['doctype'].value_counts()

# Create training and testing variables
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',encoding='latin-1', stop_words='english')
features = tfidf.fit_transform(X_train.details.values.astype(str))

X_test_idf = tfidf.transform(X_test.details.values.astype(str))

nbmodel = MultinomialNB().fit(features, y_train)

# Save the model
joblib.dump(nbmodel, 'nbmodel.pkl')