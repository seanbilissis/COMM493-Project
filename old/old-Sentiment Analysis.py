import pandas as pd
import numpy as np

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

nltk.download('punkt')


countries=pd.read_csv("Countries.csv") 
customers=pd.read_csv("Customer.csv") 
training=pd.read_csv("Training.csv") 
geographies=pd.read_csv("Geographies.csv") 
orders=pd.read_csv("Orders.csv") 
products=pd.read_csv("Products.csv")
sales=pd.read_csv("Sales.csv")  
feedback=pd.read_csv("Feedback.csv") 

training=training

testingData={"Comment":range(1,feedback.shape[0]+1),"Comment.1":feedback["Feedback Received"]}
testing=pd.DataFrame(testingData,columns=["Comment","Comment.1"])


''' Step 1: Assigning Sentiment to Training '''
def sentimentAssignent(df):
    comments=df["Comment.1"]
    sentimentList=[]
    roundedSentiment=[]
    
    
    def Rounder(sentiment):
        if (sentiment>=-1) and (sentiment<-0.25):
            sentiment="Disappointed"
            
        elif (sentiment>=-0.25) and (sentiment<0):
            sentiment="Mixed"
            
        else:
            sentiment="Happy"
        return sentiment
    
    
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
    #Going up to tri-grams
    numbers=[1,2,3]
    for number in numbers:
        grams=[]
        
        for comment in df["Comment.1"]:
            gram=ngrams(comment, number) 
            gram=list(gram)
            for item in gram:
                if item not in features:
                    features.append(item)
                    
                    

    #Making a column for each feature
    for feature in features:
        df[feature]=0    
    #Iterating through each comment and filling out the rows    
    def FeatureFiller(row):
        numbers=[1,2,3]
        comment=row["Comment.1"]
        index=row["Comment"]-1        
        for number in numbers:
            gram=ngrams(comment,number)
            gram=list(gram)
            print("GRAM:",gram)        
            for item in gram:
                df[item][index]=df[item][index]+1
      
              
         
        
    df.apply(lambda row: FeatureFiller(row), axis=1)

Features(training)
Features(testing)



''' Step 5: Setting up X and y Variables '''
X_train = dataset.iloc[:, 3:].values
y_train = dataset.iloc[:, 2].values

X_test = dataset.iloc[:, 3:].values
y_test = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


''' Step 6: Setting up ANN '''
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)


'''Step 6: Combining y_pred and Feedback '''
feedback["Sentiment"]=testing["Sentiment"]

feedback.to_csv("Analysis.csv",index=False)

