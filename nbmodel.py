# ---------------------Importing all the required modules----------------------

import pandas as pd
# For splitting into training and testing data
from sklearn.model_selection import train_test_split
# For term frequency - inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
# For Naive Bayes Classification Modeling
from sklearn.naive_bayes import MultinomialNB
import pickle
# For saving the model
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

# ------------------- Data Exploration and Manipulation------------------------

# Reading the dataset from local machine
df = pd.read_csv('/Users/adi/Downloads/shuffled-full-set-hashed.csv', names=["doctype", "details"])
y = df.doctype

# Count the instances of each document type
df['doctype'].value_counts()

# Create training and testing variables
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.15)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# Vectorization of the training data
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2')
features = tfidf.fit_transform(X_train.details.values.astype(str))

joblib.dump(tfidf, 'vec_count.joblib')
print ("Success")

# X_test_idf = tfidf.transform(X_test.details.values.astype(str))

# ---------------- Building a multinomial Naive Bayes model----------------------
# 0.7932697460079305 accuracy
nbmodel = MultinomialNB(alpha=0.1).fit(features, y_train)

# Save the model
pickle.dump(nbmodel, open("nbmodel.pkl", "wb"))
