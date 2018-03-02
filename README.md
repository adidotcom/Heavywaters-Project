# Heavywaters-Project

The idea of the project is to be able to successfully use machine learning concepts and build a document classification model. A document labeled dataset is provided, that has 62204 entries of documents. Each entry has a document label and an OCR output of the document. All the words of the document are obscured using fixed length hashing. 

The dataset looks something like this:

CANCELLATION NOTICE, 586242498a88 21e314d3afcc 818a7ff3bf29 4e43b72d46c0 578830762b27........ 43565b1afa44 5f6653c869fc

### Step 1: Understanding the data
The very first approach was to look at the data and understand what they mean at a very high level. The document can be labelled into 14 different categories and the following number of records:
```
List of categorical variable and their occurance the in the dataset

BILL                       18968                                                                                             
POLICY CHANGE              10627
CANCELLATION NOTICE         9731
BINDER                      8973
DELETION OF INTEREST        4826
REINSTATEMENT NOTICE        4368
DECLARATION                  968
CHANGE ENDORSEMENT           889
RETURNED CHECK               749
EXPIRATION NOTICE            734
NON-RENEWAL NOTICE           624
BILL BINDER                  289
INTENT TO CANCEL NOTICE      229
APPLICATION                  229
```
A machine learning model (classifier) is to be built that can learn from this dataset as an input. It will be used later to predict the category of a document when an unseen document input is given to it. 

### Step 2: Train-test split
The dataset is split into two parts: one for training the model and the other set will be used as a test dataset in order to make predictions. The *train_test_split* method from the *scikit-learn library* was used to split the dataset in the ratio of 80:20, where 80% of the data was used for training the classifier and 20% of the data was reserved for making model prediction.

```
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
```
It was nececcary to built a *features* set from each given input text entry inorder to train the model. This is done by transforming the obscured text into numeric vectors. *Bag of words* which is one of the widely known models, helps to assign numeric values to the words, creating a list of numbers. Scikit provides a vectorizer called TfidfVectorizer which transforms the text based on the bag-of-words, additionally, it computes term frequencies and evaluate each word using the tf-idf weighting scheme.

```
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',encoding='latin-1', stop_words='english')
features = tfidf.fit_transform(X_train.details.values.astype(str))
```

### Step 3: Builing the model
From the various supervised machine learning algorithms available at hand, this project is built using the Multinomial Naive Bayes model, as it provided the best result and faster processing. 

```
nbmodel = MultinomialNB().fit(features, y_train)
```
The Naive Bayes Classifier has an accuracy of approximately 72% 

### Running AWS ML instance
The AWS ML capability was also used to train and test the classifier. AWS ML built and evaluated a classifier with an average F-1 score of 0.72

### Deploying the model on AWS cloud platform
This was my first experience deploying a model on AWS/Azure platform. Interface wise, I found Azure ML much more user-friendly. Finally, I decided to go with AWS, as Heavywaters rely majorly on this platform. The first step was to upload the model on the AWS S3 bucket and then deploy the model on a *serverless* AWS Lambda. 

The model was then attempted to be tested on the local machine using Flask API. A virtual environment was created, and the flask application was implemented under it. A python executable *predictions.py* is attached in the repository. The program currently sends a POST request to the model in the S3 bucket and gives a 405 METHOD NOT ALLOWED error. Many attempts have been made to connect to the S3 bucket and run the model on the local machine. Due to time restrictions and as I am still learning AWS and getting better at it, the project is still under *production* and is expected to be completed by Tuesday 6th March.
