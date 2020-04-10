'''
Name of tables created:
    dfAverage - This dataframe tells the average sentiment per week.
    dfSentimentandSales- This dataframe tells the average sentiment per week and sales per week.
    dfCountry - This dataframe provides Sentiment by country
    dfCustomers - This provides a list of customers with their sentiment not sorted (dfCustomerSortedLeastToMost and dfCustomerSortedMostToLeast are sorted)
    dfAges - Sentiment by age group
    dfGender - Sentiment by gender
    dfProduct - Sentiment for each product
    dfProductType - This provides an overview of what sentiment is across different product types (hat, sweater, shirt....)
    dfProductStyle - This provides an overview of what sentiment is across different product types (funny and bold, party prices, contemporary pieces)
    dfCustomerSentimentChange- This provides information on what customer's sentiment has changed the most over time
    dfComplaints - Shows complaints and their frequency
    dfPraise - shows praises and their frequency
     
'''





#Importing Libraries
import pandas as pd
import numpy as np


#Importing Dataframes
countries=pd.read_csv("Countries.csv") 
customers=pd.read_csv("Customer.csv") 
geographies=pd.read_csv("Geographies.csv") 
orders=pd.read_csv("Orders.csv") 
products=pd.read_csv("Products.csv")
sales=pd.read_csv("Sales.csv")  
feedback=pd.read_csv("Feedback1.csv") 
orderItems=pd.read_csv("Order_Items.csv")



'''
dfAverage
This dataframe tells the average sentiment per week.
Because it provides a more higher-level overview it is better suited for the CEO
'''
#Average Sentiment and Sales over time
latestComment=feedback["Week Posted"].max()
earliestComment=feedback["Week Posted"].min()

weeks=range(earliestComment,latestComment+1)
averages=[]
for item in weeks:
    count=0
    totalSum=0
    for index,row in feedback.iterrows():
        if row["Week Posted"]==item:
            count=count+1
            totalSum=totalSum+row["Sentiment"]
    average=totalSum/count
    averages.append(average)

Data={"Week":weeks,"Average":averages}
dfAverage=pd.DataFrame(data=Data)
#dfAverage.to_csv('dfAverage.csv')
            



'''
dfSentimentandSales
This dataframe tells the average sentiment per week and sales per week.
Because it provides a more higher-level overview it is better suited for the CEO
'''
Data={"Week":weeks,"Average":averages,}
dfSentimentandSales=pd.DataFrame(data=Data)
dfSentimentandSales["Sales"]=sales["Net Sales"][earliestComment-2:latestComment-1]

#dfSentimentandSales.to_csv('dfSentimentandSales.csv')



'''
dfCountry
This dataframe provides Sentiment by country
'''

orderNumbers=feedback["Order_Number"].tolist()
Alpha3=[]
Sentiment=[]
Count=[]
Average=[]


for item in orderNumbers:
    
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    countryID=customers.loc[customers['Customer ID']==customerID,'Country'].iloc[0]
    countryAlpha3=countries.loc[countries['Text Code']==countryID,'Alpha-3 code'].iloc[0]
    
    if countryAlpha3 not in Alpha3:
        Alpha3.append(countryAlpha3)
        Sentiment.append(sentiment)
        Count.append(1)
    elif countryAlpha3 in Alpha3:
        CountryIndex=Alpha3.index(countryAlpha3)
        Sentiment[CountryIndex]=Sentiment[CountryIndex]+sentiment
        Count[CountryIndex]=Count[CountryIndex]+1

for item in Alpha3:
    CountryIndex=Alpha3.index(item)
    Average.append(Sentiment[CountryIndex]/Count[CountryIndex])

country_data={"Country":Alpha3, "Sentiment":Average}
dfCountry=pd.DataFrame(data=country_data)
#dfCountry.to_csv('dfCountry.csv')
    
    
'''
dfCustomers
This provides a list of customers with their sentiment sorted from highest to least
This is a more lower level view of what customers are feeling and is better suited for the marketing manager
'''    
#Sentiment by Customer (Most and Least) 
Customers=[]
Sentiment=[]
Count=[]
Average=[]
for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    
    if customerID not in Customers:
        Customers.append(customerID)
        Sentiment.append(sentiment)
        Count.append(1)
    else:
        CustomerIndex=Customers.index(customerID)
        Sentiment[CustomerIndex]=Sentiment[CustomerIndex]+sentiment
        Count[CustomerIndex]=Count[CustomerIndex]+1
for item in Customers:
    CustomerIndex=Customers.index(customerID)
    Average.append(Sentiment[CustomerIndex]/Count[CustomerIndex])

dfCustomers=pd.DataFrame(data={"Customer":Customers,"Average Sentiment":Sentiment})
dfCustomerSortedLeastToMost=dfCustomers.sort_values(by="Average Sentiment",ascending=True)
dfCustomerSortedMostToLeast=dfCustomers.sort_values(by="Average Sentiment",ascending=False)

#dfCustomerSortedLeastToMost.to_csv('dfCustomerSortedLeastToMost.csv')
#dfCustomerSortedMostToLeast.to_csv('dfCustomerSortedMostToLeast.csv')




'''
dfAges
this provides a dataframe of sentiment across various age groups. Suited for a histogram
Because of its more demographic view, it is better suited for the marketing dashboard
'''
Ages=[]
Sentiment=[]
Count=[]
Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    DOB=customers.loc[customers["Customer ID"]==customerID,"Birthday"].iloc[0]
    last2=DOB[-2:]
    last2=str(last2)
    
    if last2[0] == "0":
        year="20"+last2
        year=int(year)
    
    else:
        year="19"+last2
        year=int(year)
    age=2020-year
    
    if age not in Ages:
        Ages.append(age)
        Sentiment.append(sentiment)
        Count.append(1)
    else:
        AgeIndex=Ages.index(age)
        Sentiment[AgeIndex]=Sentiment[AgeIndex]+sentiment
        Count[AgeIndex]=Count[AgeIndex]+1
    
for item in Ages:
    AgeIndex=Ages.index(item)
    Average.append(Sentiment[AgeIndex]/Count[AgeIndex])

minAgeAc=min(Ages)
maxAgeAc=max(Ages)

if min(Ages)%5 != 0:
    minAge=min(Ages)-min(Ages)%5
else:
    minAge=min(Ages)

if max(Ages)%5 != 0:
    maxAge=(5-max(Ages)%5)+max(Ages)+5
else:
    maxAge=max(Ages)+5


bins=list(range(minAge,maxAge,5))



dfAges=pd.DataFrame(data={"Age":Ages,"Average_Sentiment":Sentiment,"Count":Count})
dfAges["Binned"]=pd.cut(dfAges["Age"],bins=bins)
dfAges["Bin Sentiment"]=dfAges.Binned.map(dfAges.groupby(["Binned"]).Average_Sentiment.mean())
dfAges=dfAges[["Binned","Bin Sentiment"]]
dfAges=dfAges.drop_duplicates()
dfAges=dfAges.rename(columns={"Binned":"Age Group","Bin Sentiment":"Average Sentiment"})
#dfAges.to_csv('dfAges.csv')

        

    
'''
dfGender
this provides a dataframe of sentiment across gender groups. 
Because of its more demographic view, it is better suited for the marketing dashboard
'''
Genders=[]
Sentiment=[]
Count=[]
Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    gender=customers.loc[customers["Customer ID"]==customerID,'Gender'].iloc[0]
    if gender==0:
        gender="Female"
    else:
        gender="Male"
    
    if gender not in Genders:
        Genders.append(gender)
        Sentiment.append(sentiment)
        Count.append(1)
    else:
        GenderIndex=Genders.index(gender)
        Sentiment[GenderIndex]=Sentiment[GenderIndex]+sentiment
        Count[GenderIndex]=Count[GenderIndex]+1

for item in Genders:
    GenderIndex=Genders.index(item)
    Average.append(Sentiment[GenderIndex]/Count[GenderIndex])
dfGender=pd.DataFrame(data={"Gender":Genders,"Average Sentiment":Average})
#dfGender.to_csv('dfGender.csv')


'''
dfProduct
This provides an overview of what sentiment is across all products
This information is more lower level and is better suited for the CMO
'''
#Sentiment by Product
Products=[]
Sentiment=[]
Count=[]
Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    productID=orderItems.loc[orderItems["Order Number"]==item,"Product_ID"].iloc[0]
    
    if productID not in Products:
        Products.append(productID)
        Sentiment.append(sentiment)
        Count.append(1)
    else:
        productIndex=Products.index(productID)
        Sentiment[productIndex]=Sentiment[productIndex]+sentiment
        Count[productIndex]=Count[productIndex]+1
for item in Products:
    productIndex=Products.index(item)
    Average.append( Sentiment[productIndex]/ Count[productIndex])

dfProduct=pd.DataFrame(data={"Products":Products,"Average Sentiment":Average})
#dfProduct.to_csv('dfProduct.csv')

   
    

'''
dfProductType
This provides an overview of what sentiment is across different product types (hat, sweater, shirt....)
This information is more lower level and think it is better suited for the CMO
'''
ProductTypes=["hat", "sweater", "shirt", "jacket", "pant", "shoe", "blouse"]
Sentiment=[0,0,0,0,0,0,0]
Count=[0,0,0,0,0,0,0]
Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    productID=orderItems.loc[orderItems["Order Number"]==item,"Product_ID"].iloc[0]
    des=products.loc[products["Product_ID"]==productID,"Description"].iloc[0]
    des=des.lower()
    
    for p in ProductTypes:
        if p in des:
            index=ProductTypes.index(p)
            Count[index]=Count[index]+1
            Sentiment[index]=Sentiment[index]+sentiment


for item in ProductTypes:
    index=ProductTypes.index(item)
    if Count[index]!=0:
        index=ProductTypes.index(item)
        Average.append(Sentiment[index]/Count[index])
    else:
        Average.append(0)

dfProductType=pd.DataFrame(data={"Product Type":ProductTypes,"Average Sentiment":Average})
dfProductType = dfProductType[~(dfProductType == 0).any(axis=1)]            
#dfProductType.to_csv('dfProductType.csv')
    
    

'''
dfProductStyle
This provides an overview of what sentiment is across different product types (funny and bold, party prices, contemporary pieces)
This information is more lower level and think it is better suited for the CMO
'''
#Sentiment by Product Style (funky and bold, party peices, etc)
ProductStyle=["funky and bold", "party pieces", "contemporary and clean", "hipster trend", "dress to impress", "runway inspired", "romantic and bohemian","athletic","utility and basics","relaxed and casual"]

Sentiment=[0,0,0,0,0,0,0,0,0,0]
Count=[0,0,0,0,0,0,0,0,0,0]

Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    productID=orderItems.loc[orderItems["Order Number"]==item,"Product_ID"].iloc[0]
    des=products.loc[products["Product_ID"]==productID,"Description"].iloc[0]
    des=des.lower()
    
    for p in ProductStyle:
        if p in des:
            index=ProductStyle.index(p)
            Count[index]=Count[index]+1
            Sentiment[index]=Sentiment[index]+sentiment


for item in ProductStyle:
    index=ProductStyle.index(item)
    if Count[index]!=0:
        index=ProductStyle.index(item)
        Average.append(Sentiment[index]/Count[index])
    else:
        Average.append(0)

dfProductStyle=pd.DataFrame(data={"Product Style":ProductStyle,"Average Sentiment":Average})
dfProductStyle= dfProductStyle[~(dfProductStyle == 0).any(axis=1)]     
#dfProductStyle.to_csv('dfProductStyle.csv')
    
        


'''
dfCustomerSentimentChange
This provides information on what customer's sentiment has changed the most over time
This is lower level data and better suited for the CMO
'''
#Change in sentiment over time per customer
FirstandLast={}
for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    keys=list(FirstandLast.keys())
    if customerID not in keys:
        FirstandLast[customerID]=[sentiment,0]
    else:
        FirstandLast[customerID][1]=sentiment

customerIDs=list(FirstandLast.keys())
sentimentChange=[]
for item in customerIDs:
    if len(FirstandLast[item])>1:
        first=FirstandLast[item][0]
        last=FirstandLast[item][1]
        change=last-first
        sentimentChange.append(change)


dfCustomerSentimentChange=pd.DataFrame(data={"CustomerID":customerIDs,"Change":sentimentChange})
#dfCustomerSentimentChage.to_csv('dfCustomerSentimentChage.csv')
dfAscending5 = dfCustomerSentimentChange.nsmallest(5, "Change")



'''
dfComplaints - Shows complaints and their frequency
dfComplaints=pd.Series(feedback["Category"]).value_counts() 
dfComplaints=dfComplaints.to_frame()
dfComplaints["Category"]=dfComplaints.index
dfComplaints[~dfComplaints.commentType.str.contains("praise")]
#dfComplaints.to_csv('dfComplaints.csv')
'''
'''
dfPraise - shows praises and their frequency
dfPraise=pd.Series(feedback["Category"]).value_counts() 
dfPraise=dfPraise.to_frame()
dfPraise["Category"]=dfComplaints.index
dfPraise[~dfPraise.commentType.str.contains("complaint")]
#dfPraise.to_csv('dfPraise.csv')
'''


'''
dfComplaints - Shows complaints and their frequency
'''
dfComplaints=pd.Series(feedback["Category"]).value_counts() 
dfComplaints=dfComplaints.to_frame()
dfComplaints["commentType"]=dfComplaints.index
dfComplaints=dfComplaints[~dfComplaints.commentType.str.contains("praise")]
dfComplaints.columns = ["Occurences", "Category"]
dfComplaints.to_csv('dfComplaints.csv')



'''
dfPraise - shows praises and their frequency
'''
dfPraise=pd.Series(feedback["Category"]).value_counts() 
dfPraise=dfPraise.to_frame()
dfPraise["commentType"]=dfPraise.index
dfPraise=dfPraise[~dfPraise.commentType.str.contains("complaint")]
dfPraise.columns = ["Occurences", "Category"]
#dfPraise.to_csv('dfPraise.csv')
















