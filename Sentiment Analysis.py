
import pandas as pd
import numpy as np

from xgboost import XGBClassifier


import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import ngrams
nltk.download('stopwords')

import textblob
from newspaper import Article
from textblob import TextBlob

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix



import keras
from keras.models import Sequential
from keras.layers import Dense

nltk.download('punkt')


countries=pd.read_csv("Countries.csv") 
customers=pd.read_csv("Customer.csv") 
training=pd.read_csv("Training.csv") 
geographies=pd.read_csv("Geographies.csv") 
orders=pd.read_csv("Orders.csv") 
products=pd.read_csv("Products.csv")
sales=pd.read_csv("Sales.csv")  
feedback=pd.read_csv("Feedback.csv") 



#Making Testing Dataframe
testingData={"Comment":range(1,feedback.shape[0]+1),"Comment.1":feedback["Feedback Received"],"Category":feedback["Category"]}
testing=pd.DataFrame(testingData,columns=["Comment","Comment.1","Category"])


#Labels were made by different people so making sure styling is correct by remove spaces and making all labels consistent
training["Category"]=training["Category"].str.lower()
testing["Category"]=testing["Category"].str.lower()


training["Category"]=training["Category"].str.replace(" ","")
testing["Category"]=testing["Category"].str.replace(" ","")


''' Step 1: Assigning Sentiment to Training '''
def sentimentAssignent(df):
    comments=df["Comment.1"]
    sentimentList=[]
 
    
    for item in comments:
    
        obj = TextBlob(item)
        sentiment = obj.sentiment.polarity
        sentimentList.append(sentiment)
        
        
    
    df["Sentiment"]=sentimentList
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
    df["Sentiment"]=scaler.fit_transform(df["Sentiment"].values.reshape(-1,1))
    

sentimentAssignent(training)
sentimentAssignent(testing)

    





''' Step 3: Data Preprocessing '''

def preProcessing(text):
    #split into words
    tokens=word_tokenize(text)
    #convert to lowercase
    tokens= [w.lower() for w in tokens]
    #Filter out punctuation
    words = [word for word in tokens if word.isalpha()]
    #Filter out stop words
    stop_words = stopwords.words('english')
    words=[w for w in words if w not in stop_words]
    #Reducing to stem
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    return(stemmed)

#Applying pre-processing to the training data
training["Comment.1"]=training["Comment.1"].apply(preProcessing)    
testing["Comment.1"]=testing["Comment.1"].apply(preProcessing)    



''' Step 4: Feature Engineering (using N-grams) '''


def Features(df):
    features=[]
    #Going up to five-grams
    numbers=[1,2,3,4,5]
    for number in numbers:
        
        #for comment in df["Comment.1"]:
        for comment in training["Comment.1"]:
            gram=ngrams(comment, number) 
            gram=list(gram)
            for item in gram:
                if item not in features:
                    features.append(item)

    return features
                
def FeatureFiller(df,features):                
    #Getting names of all columns

    #Making a column for each feature
    for feature in features:
        df[feature]=0    
    #Iterating through each comment and filling out the rows    
    def FeatureFillerRow(row):
        numbers=[1,2,3,4,5]
        comment=row["Comment.1"]
        index=row["Comment"]-1        
        for number in numbers:
            gram=ngrams(comment,number)
            gram=list(gram)
            for item in gram:
                if item in features:
                    df[item][index]=df[item][index]+1
              
    df=df.apply(lambda row: FeatureFillerRow(row), axis=1)
listFeatures=Features(training)
FeatureFiller(training,listFeatures)
FeatureFiller(testing,listFeatures)









''' Step 5: Setting up X and y Variables '''
le= preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()


X_train = training.iloc[:, 3:].values
y_train = training.iloc[:, 2].values
le.fit(y_train)
y_train=le.transform(y_train)
lb.fit(y_train)
y_train=lb.transform(y_train)

X_test = testing.iloc[:, 3:].values
y_test = testing.iloc[:, 2].values
y_test=le.transform(y_test)


'''Using Random Forest to get important features'''
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

features=list(training.iloc[:, 3:])
importances=list(classifier.feature_importances_)

print(len(features), len(importances))
#FI={"features":features,"importances":importances}
dfFI=pd.DataFrame()
dfFI["features"]=features
dfFI["importances"]=importances
dfFI.sort_values(by=["importances"])
    
dfFI=dfFI.iloc[0:1000,:]

topFeatures=list(dfFI["features"])
training=training[training.columns.intersection(topFeatures)]
testing=testing[testing.columns.intersection(topFeatures)]



#Adjusting X Variables
X_train = training.iloc[:, :].values
X_test = testing.iloc[:, :].values

# Feature Scaling

sc = StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test = sc.transform(X_test)






''' Step 6: Setting up ANN '''
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = int(X_test.shape[1]/2), init = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))

#Adding second hidden layer
classifier.add(Dense(output_dim = int(X_test.shape[1]/5), init = 'uniform', activation = 'relu', input_dim = int(X_test.shape[1]/2)))


# Adding the output layer
classifier.add(Dense(output_dim =8, init = 'uniform', activation = 'softmax'))


# Compiling the ANN
classifier.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
 
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size =32, epochs = 5)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred2 = classifier.predict(X_test)
predictions = np.argmax(y_pred2, axis=-1) 
ac2=accuracy_score(y_test, predictions) #66% Accuraacy




predictions=list(le.inverse_transform(predictions))
testing["Comment_Category"]=predictions
testing["Week Posted"]=feedback["Week Posted"]
testing["Order_Number"]=feedback["Order_Number"]
testing["Category"]=feedback["Category"]


testing.to_csv('Feedback1.csv')









































