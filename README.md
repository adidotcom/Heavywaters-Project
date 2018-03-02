# Heavywaters-Project

The idea of the project is to be able to successfully use machine learning cocepts and build a document classification model. A document labeled dataset is provided, that has 62204 entries of documents. Each entry has a document label and an OCR output of the document. All the words of the document was obscured using fixed length hashing. 

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

